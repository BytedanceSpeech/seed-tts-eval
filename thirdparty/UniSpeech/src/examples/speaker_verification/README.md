## Pre-training Representations for Speaker Verification

### Pre-trained models

| Model                                                        | Fix pre-train | Vox1-O    | Vox1-E    | Vox1-H   |
| ------------------------------------------------------------ | ------------- | --------- | --------- | -------- |
| [ECAPA-TDNN](https://drive.google.com/file/d/1kWmLyTGkBExTdxtwmrXoP4DhWz_7ZAv3/view?usp=sharing) | -             | 1.080     | 1.200     | 2.127    |
| [HuBERT large](https://drive.google.com/file/d/1cQAPIzg8DJASZyAYdaBN0gRa8piPQTMo/view?usp=sharing) | Yes           | 0.888     | 0.912     | 1.853    |
| [Wav2Vec2.0 (XLSR)](https://drive.google.com/file/d/1FiGokGtF2d7rkD9OpqLiQxKSqppTSXbl/view?usp=sharing) | Yes           | 0.915     | 0.945     | 1.895    |
| [**UniSpeech-SAT large**](https://drive.google.com/file/d/1W6KRt5Ci2T7_xPVdlE3JGdQG2KTrZ750/view?usp=sharing) | Yes           | 0.771     | 0.781     | 1.669    |
| [HuBERT large](https://drive.google.com/file/d/1nit9Z6RyM8Sdb3n8ccaglOQVNnqsjnui/view?usp=sharing) | No            | 0.585     | 0.654     | 1.342    |
| [Wav2Vec2.0 (XLSR)](https://drive.google.com/file/d/1TgKro9pp197TCgIF__IlE_rMVQOk50Eb/view?usp=sharing) | No            | 0.564     | 0.605     | 1.23     |
| [**UniSpeech-SAT large**](https://drive.google.com/file/d/10o6NHZsPXJn2k8n57e8Z_FkKh3V4TC3g/view?usp=sharing) | No            | **0.564** | **0.561** | **1.23** |

### How to use?

#### Environment Setup

1. `pip install -r requirements.txt`
2. Install fairseq code
   - For HuBERT_Large and Wav2Vec2.0 (XLSR), we should install the official [fairseq](https://github.com/pytorch/fairseq).
   - For UniSpeech-SAT large, we should install the [Unispeech-SAT](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) fairseq code.

#### Example

Take `unispeech_sat ` and `ecapa_tdnn` for example:

1. First, you should download the pre-trained model in the above table to `checkpoint_path`.
2. Then, run the following codes:
   - The wav files are sampled from [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).

```bash
python verification.py --model_name unispeech_sat --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.0317 (-1.0, 1.0).

python verification.py --model_name unispeech_sat --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav --checkpoint --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.5389 (-1.0, 1.0).

python verification.py --model_name ecapa_tdnn --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/Josh_Gad/HXUqYaOwrxA_0000015.wav --checkpoint $checkpoint_path
# output: The similarity score between two audios is 0.2053 (-1.0, 1.0).

python verification.py --model_name ecapa_tdnn --wav1 vox1_data/David_Faustino/hn8GyCJIfLM_0000012.wav --wav2 vox1_data/David_Faustino/xTOk1Jz-F_g_0000015.wav --checkpoint --checkpoint $checkpoint_path
# output: he similarity score between two audios is 0.5302 (-1.0, 1.0).
```

