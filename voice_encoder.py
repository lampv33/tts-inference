import librosa
from librosa.util import normalize
import torchaudio
import torch
import yaml
import os
from IPython.display import Audio, display
from models import recursive_munch, build_voice_encoder


def load_wav(wav_path, sr):
    wav, _ = librosa.load(wav_path, sr=sr)
    return wav


def norm_wav(wav):
    wav = normalize(wav) * 0.95
    wav, _ = librosa.effects.trim(wav, top_db=30)
    return wav


class VoiceEncoder:
    def __init__(self, checkpoint_path, config_path):
        self.model_name = os.path.basename(os.path.dirname(checkpoint_path))
        self.model, self.sr = self.load_model(checkpoint_path, config_path)
        self.melspec_process = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

    def load_model(self, checkpoint_path, config_path):
        print("Loading voice encoder: '{}'".format(checkpoint_path))

        config = recursive_munch(yaml.safe_load(open(config_path)))
        sr = config.preprocess_params.sr
        model = build_voice_encoder(config.model_params)
        
        state_dicts = torch.load(checkpoint_path, map_location='cpu')['net']
        for key, module in model.items():
            state_dict = state_dicts[key]
            state_dict = {k[7:]: v for k, v in state_dict.items()} if list(state_dict.keys())[0].startswith('module.') else state_dict
            module.load_state_dict(state_dict, strict=False)
            module.eval().cuda()

        return model, sr

    def wav2mel(self, wav):
        wav_tensor = torch.from_numpy(wav).float()
        mel_tensor = self.melspec_process(wav_tensor)

        mean, std = -4, 4
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

        return mel_tensor

    def compute_ref_emb(self, ref_speaker_audio_path, ref_prosody_audio_path=''):
        wav_s = load_wav(ref_speaker_audio_path, sr=self.sr)
        wav_s = norm_wav(wav_s)
        melspec_s = self.wav2mel(wav_s).cuda()

        if ref_prosody_audio_path and ref_prosody_audio_path != ref_speaker_audio_path:
            wav_p = load_wav(ref_prosody_audio_path, sr=self.sr)
            wav_p = norm_wav(wav_p)
            melspec_p = self.wav2mel(wav_p).cuda()
        else:
            ref_prosody_audio_path = ref_speaker_audio_path
            melspec_p = melspec_s

        with torch.no_grad():
            ref_s = self.model.style_encoder(melspec_s.unsqueeze(1))
            ref_p = self.model.predictor_encoder(melspec_p.unsqueeze(1))

        ref_embedding = torch.cat([ref_s, ref_p], dim=1)

        display(Audio(ref_speaker_audio_path))
        display(Audio(ref_prosody_audio_path))

        return ref_embedding
    
    def compute_ref_emb_mix(self, ref_speaker_audio_path, ref_prosody_audio_path='', ref_speaker_audio_path_2='', ref_prosody_audio_path_2='', ratio=2):
        if not ref_speaker_audio_path_2:
            ref_speaker_audio_path_2 = ref_prosody_audio_path
        if not ref_prosody_audio_path_2:
            ref_prosody_audio_path_2 = ref_speaker_audio_path

        ref_embedding_1 = self.compute_ref_emb(ref_speaker_audio_path, ref_prosody_audio_path)
        ref_embedding_2 = self.compute_ref_emb(ref_speaker_audio_path_2, ref_prosody_audio_path_2)
        ref_embedding = (ref_embedding_1 * ratio + ref_embedding_2) / (ratio + 1)
        return ref_embedding

