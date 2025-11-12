import requests
import os
#os.environ['http_proxy'] = 'http://10.60.28.99:81'
#os.environ['https_proxy'] = 'http://10.60.28.99:81'
import time
import numpy as np
import IPython.display as ipd
from scipy.io import wavfile
from prepare_voices import prepare_voices
from CocCocTokenizer import PyTokenizer

T = PyTokenizer(load_nontone_data=True)


def tokenize(text):
    text = text.replace('_', '#')
    text = ' '.join(w for w in T.word_tokenize(text))
    text = text.replace('#', '_').replace(' _ ', '_').replace(' - ', '-')
    return text


def normalize_text_api(text):
    """
    Gửi một yêu cầu POST đến API normalize.

    Args:
        text: Nội dung văn bản cần gửi.

    Returns:
        Nội dung phản hồi từ server nếu thành công, ngược lại trả về None.
    """
    url = 'http://0.0.0.0:7779/normalize'

    payload = {'content': text}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        norm_text = data.get('normText')
        return norm_text
    except requests.exceptions.RequestException as e:
        print(f"Đã xảy ra lỗi khi gửi yêu cầu: {e}")
        return text


def text2speech(voice_name, style, model, voice_encoder, lora_dir=None, text='', alpha=0, beta=0, diffusion_steps=5, embedding_scale=1, speed=1, max_len=512, save_embedding=False, save_audio=True):
    ref_embedding, text_demo = prepare_voices(voice_name, style, voice_encoder)
    if not text:
        text = text_demo.lower()
        save_embedding = save_audio = True
    else:
        text = normalize_text_api(text).lower()
    print(text)
    
    start_time = time.time()
    wav = model.gen_long_wav(text, ref_embedding, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, speed=speed, max_len=max_len, lora_dir=lora_dir)
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