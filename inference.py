import os
import time
import yaml
import numpy as np

from peft import PeftModel, LoraConfig, get_peft_model
from text import split_text, Text2ID
from models import *
from plbert import load_plbert
from modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ModelsManager:
    def __init__(self, base_model):
        self.model = base_model
        self.active_lora_dir = None

    def get_model(self, lora_dir=None):
        if lora_dir == self.active_lora_dir:
            return self.model

        for module_name, module_instance in self.model.items():
            if isinstance(module_instance, PeftModel):
                if self.active_lora_dir is not None:
                    module_instance.disable_adapter_layers()

                if lora_dir is not None:
                    adapter_module_path = os.path.join(lora_dir, module_name)
            
                    if adapter_module_path not in module_instance.peft_config:
                        module_instance.load_adapter(adapter_module_path, adapter_name=adapter_module_path)
                        module_instance.cuda()
                    
                    module_instance.enable_adapter_layers()
                    module_instance.set_adapter(adapter_module_path)
    
        self.active_lora_dir = lora_dir
        
        return self.model
    

class LoraInference:
    def __init__(self, base_model_checkpoint_path, base_model_config_path, lora_config_path):
        self.text2id = Text2ID()

        base_model, self.decoder_type = self.load_base_model(base_model_checkpoint_path, base_model_config_path)
        base_model = self.prepare_model_for_lora(base_model, lora_config_path)
        base_model = recursive_munch({key: module.eval().cuda() for key, module in base_model.items()})

        self.sampler = self.load_diffusion_sampler(base_model)
        self.models_manager = ModelsManager(base_model)
    
    def load_base_model(self, base_model_checkpoint_path, base_model_config_path):
        config = recursive_munch(yaml.safe_load(open(base_model_config_path)))
        decoder_type = config.model_params.decoder.type
        print("Loading base model: '{}'".format(base_model_checkpoint_path))
            
        plbert = load_plbert(config.PLBERT_dir) 
        base_model = build_model(config.model_params, len(self.text2id.symbol2index), plbert)
        
        state_dicts = torch.load(base_model_checkpoint_path, map_location='cpu')['net']
        for key, module in base_model.items():
            state_dict = state_dicts[key]
            state_dict = {k[7:]: v for k, v in state_dict.items()} if list(state_dict.keys())[0].startswith('module.') else state_dict
            module.load_state_dict(state_dict, strict=False)
            
        return base_model, decoder_type
    
    def prepare_model_for_lora(self, model, lora_config_path):
        print('Preparing model for Lora')
        if lora_config_path is not None:
            lora_config = recursive_munch(yaml.safe_load(open(lora_config_path)))
            target_layer_mapping = {
                'Linear': nn.Linear,
                'Conv1d': nn.Conv1d,
                'Conv2d': nn.Conv2d
                }
            TARGET_LAYER_TYPES = tuple(target_layer_mapping[name] for name in lora_config.target_layers)

            for module_name in lora_config.target_modules:
                module_to_modify = model[module_name]
                target_modules = [name for name, module in module_to_modify.named_modules() if isinstance(module, TARGET_LAYER_TYPES)]  

                LORA_CONFIG = LoraConfig(
                    r=lora_config.r,
                    lora_alpha=lora_config.alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_config.dropout,
                    bias=lora_config.bias)
                
                peft_model = get_peft_model(module_to_modify, LORA_CONFIG)
                model[module_name] = peft_model
            
        return model

    def load_diffusion_sampler(self, model):
        if hasattr(model, 'diffusion'):
            sampler = DiffusionSampler(
                    model.diffusion.diffusion,
                    sampler=ADPM2Sampler(),
                    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                    clamp=False)
            sampler = sampler.eval().cuda()
        else: 
            sampler = None

        return sampler

    def gen_wav(self, text, ref_embedding, ref_text='', speed=1, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, model=None, lora_dir=None):   
        if model is None:
            model = self.models_manager.get_model(lora_dir)

        tokens = self.text2id(text)
        assert len(tokens) <= 1024
        tokens = torch.LongTensor(tokens).to('cuda').unsqueeze(0)
        
        if ref_text:
            ref_tokens = self.text2id(ref_text)
            assert len(ref_tokens) <= 1024
            ref_tokens = torch.LongTensor(ref_tokens).to('cuda').unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to('cuda')
            text_mask = length_to_mask(input_lengths).to('cuda')

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            if ref_text:
                ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to('cuda')
                ref_text_mask = length_to_mask(ref_input_lengths).to('cuda')
                ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
                bert_dur = ref_bert_dur

            if self.sampler is not None and (alpha > 0 or beta > 0):    
                s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to('cuda'), 
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                            features=ref_embedding,
                                            num_steps=diffusion_steps).squeeze(1)      
                ref = s_pred[:, :128]
                s = s_pred[:, 128:]
                ref = alpha * ref + (1 - alpha)  * ref_embedding[:, :128]
                s = beta * s + (1 - beta)  * ref_embedding[:, 128:]
                s_pred = torch.cat([ref, s], dim=-1)
            else:        
                ref = ref_embedding[:, :128]
                s =  ref_embedding[:, 128:]
                s_pred = ref_embedding

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x) 

            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to('cuda'))
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to('cuda'))
            if self.decoder_type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            wav = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            wav = wav.squeeze().to('cpu').numpy()[..., 4000:]

        return wav, s_pred

    def gen_long_wav(self, text, ref_embedding, ref_text='', speed=1, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1, t=0.7, max_len=512, lora_dir=None):
        model = self.models_manager.get_model(lora_dir)
        wavs = []
        s_prev = ref_embedding

        for sentence in split_text(text, max_len):
            sentence = sentence.strip()
            wav, s_pred = self.gen_wav(sentence, s_prev, ref_text, speed, alpha, beta, diffusion_steps, embedding_scale, model)
            s_prev = t * s_pred + (1 - t) * ref_embedding
            wavs.append(wav)
 
        torch.cuda.empty_cache()
        del s_prev

        wavs = np.concatenate(wavs)     
        return wavs
