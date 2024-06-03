import soundfile as sf
import torch
import fire
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models.ecapa_tdnn import ECAPA_TDNN_SMALL
import librosa

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')


    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def verification(model_name,  wav1, wav2, use_gpu=True, checkpoint=None, wav1_start_sr=0, wav2_start_sr=0, wav1_end_sr=-1, wav2_end_sr=-1, model=None, wav2_cut_wav1=False, device="cuda:0"):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint) if model is None else model

    wav1, sr1 = librosa.load(wav1, sr=None, mono=False)

    # wav1, sr1 = sf.read(wav1)
    if len(wav1.shape) == 2:
        wav1 = wav1[:,0]
    # wav2, sr2 = sf.read(wav2)
    wav2, sr2 = librosa.load(wav2, sr=None, mono=False)
    if len(wav2.shape) == 2:
        wav2 = wav2[0,:]    # wav2.shape: [channels, T]

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)
    # print(f'origin wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')
    if wav2_cut_wav1:
        wav2 = wav2[...,wav1.shape[-1]:]
    else:
        wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
        wav2 = wav2[...,wav2_start_sr:wav2_end_sr if wav2_end_sr > 0 else wav2.shape[-1]]
    # print(f'cutted wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')

    if use_gpu:
        model = model.cuda(device)
        wav1 = wav1.cuda(device)
        wav2 = wav2.cuda(device)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    # print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return sim, model


def extract_embedding(model_name,  wav1, use_gpu=True, checkpoint=None, wav1_start_sr=0, wav1_end_sr=-1, model=None, device="cuda:0"):

    assert model_name in MODEL_LIST, 'The model_name should be in {}'.format(MODEL_LIST)
    model = init_model(model_name, checkpoint) if model is None else model

    wav1, sr1 = sf.read(wav1)
    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    wav1 = resample1(wav1)
    # print(f'origin wav1 sr: {wav1.shape}, wav2 sr: {wav2.shape}')
    wav1 = wav1[...,wav1_start_sr:wav1_end_sr if wav1_end_sr > 0 else wav1.shape[-1]]
    if use_gpu:
        model = model.cuda(device)
        wav1 = wav1.cuda(device)

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
    # print("The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(sim[0].item()))
    return emb1, model

if __name__ == "__main__":
    fire.Fire(verification)

