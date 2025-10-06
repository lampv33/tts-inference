import requests
import os
import time
import numpy as np
import IPython.display as ipd
from scipy.io import wavfile
from prepare_voices import prepare_voices

os.environ['http_proxy'] = 'http://10.60.28.99:81'
os.environ['https_proxy'] = 'http://10.60.28.99:81'


def normalize_text_api(text):
    try:
        response = requests.post(url='https://kiki-tts-engine.tts.zalo.ai/norm-text',
                                headers={'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'},
                                data={'text': text})
        response.raise_for_status()
        json_response = response.json()
        return json_response.get("result") if json_response.get("error_code") == 0 else None
    except Exception as e:
        return None


def text2speech(voice_name, style, model, voice_encoder, lora_dir=None, text='', alpha=0, beta=0, speed=1, max_len=512, save_embedding=False, save_audio=True):
    ref_embedding, text_demo = prepare_voices(voice_name, style, voice_encoder)
    if not text:
        text = text_demo.lower()
        save_embedding = save_audio = True
    else:
        text = normalize_text_api(text).lower()
    print(text)
    
    start_time = time.time()
    wav = model.gen_long_wav(text, ref_embedding, alpha=alpha, beta=beta, speed=speed, max_len=max_len, lora_dir=lora_dir)
    print('\nrtf:', round((time.time()-start_time)/(len(wav)/24000), 3))

    display(ipd.Audio(wav, rate=voice_encoder.sr, normalize=True))

    if save_embedding:
        outdir = f'ref_embeddings/{voice_encoder.model_name}'
        os.makedirs(outdir, exist_ok=True) 
        filename = f'{voice_name}_{style}.npy'
        filepath = os.path.join(outdir, filename)
        np.save(filepath, ref_embedding.cpu().numpy())
        
    if save_audio:
        outdir = f'demo_audios/{voice_encoder.model_name}'
        os.makedirs(outdir, exist_ok=True) 
        filename = f'{voice_name}_{style}.wav'
        filepath = os.path.join(outdir, filename)
        wavfile.write(filepath, voice_encoder.sr, wav)