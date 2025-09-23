import librosa
import torchaudio
import torch
from librosa.util import normalize

melspec_process = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def load_wav(wav_path, sr):
    wav, _ = librosa.load(wav_path, sr=sr)
    return wav


def norm_wav(wav):
    wav = normalize(wav) * 0.95
    wav, _ = librosa.effects.trim(wav, top_db=30)
    return wav


def wav2mel(wav):
    wav_tensor = torch.from_numpy(wav).float()
    mel_tensor = melspec_process(wav_tensor)

    mean, std = -4, 4
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std

    return mel_tensor
