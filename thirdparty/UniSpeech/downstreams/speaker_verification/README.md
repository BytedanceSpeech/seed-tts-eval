## Pre-training Representations for Speaker Verification

### Pre-trained models

| Model                                                        | Fix pre-train | Vox1-O    | Vox1-E    | Vox1-H   |
| ------------------------------------------------------------ | ------------- | --------- | --------- | -------- |
| [ECAPA-TDNN](https://drive.google.com/file/d/1kWmLyTGkBExTdxtwmrXoP4DhWz_7ZAv3/view?usp=sharing) | -             | 1.080     | 1.200     | 2.127    |
| [HuBERT large](https://drive.google.com/file/d/1njofuGpidjy_jdbq7rIbQMIDyyPLoAjb/view?usp=sharing) | Yes           | 0.888     | 0.912     | 1.853    |
| [Wav2Vec2.0 (XLSR)](https://drive.google.com/file/d/1izV48ebxs6re252ELiksuk6-RQov-gvE/view?usp=sharing) | Yes           | 0.915     | 0.945     | 1.895    |
| [UniSpeech-SAT large](https://drive.google.com/file/d/1sOhutb3XG7_OKQIztqjePDtRMrxjOdSf/view?usp=sharing) | Yes           | 0.771     | 0.781     | 1.669    |
| [WavLM Base](https://drive.google.com/file/d/1qVKHG7OzltELgkoAdFT1xXzu_hHXj3e8/view?usp=sharing) | Yes             | 0.84     | 0.928     | 1.758    |
| [**WavLM large**](https://drive.google.com/file/d/1D-dPa5H6Y2ctb4SJ5n21kRkdR6t0-awD/view?usp=sharing) | Yes           | 0.75     | 0.764     | 1.548    |
| [HuBERT large](https://drive.google.com/file/d/1nit9Z6RyM8Sdb3n8ccaglOQVNnqsjnui/view?usp=sharing) | No            | 0.585     | 0.654     | 1.342    |
| [Wav2Vec2.0 (XLSR)](https://drive.google.com/file/d/1TgKro9pp197TCgIF__IlE_rMVQOk50Eb/view?usp=sharing) | No            | 0.564     | 0.605     | 1.23     |
| [UniSpeech-SAT large](https://drive.google.com/file/d/10o6NHZsPXJn2k8n57e8Z_FkKh3V4TC3g/view?usp=sharing) | No            | 0.564 | 0.561 | 1.23 |
| [**WavLM large**](https://drive.google.com/file/d/18rekjal9NPo0VquVtali-80yy63252RX/view?usp=sharing) | No            | **0.431** | **0.538** | **1.154** |

### How to use?

#### Environment Setup

1. `pip install --require-hashes -r requirements.txt`
2. Install fairseq code
   - For HuBERT_Large and Wav2Vec2.0 (XLSR), we should install the official [fairseq](https://github.com/pytorch/fairseq).
   - For UniSpeech-SAT large, we should install the [Unispeech-SAT](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) fairseq code.
   - For WavLM, we should install the latest s3prl: `pip install s3prl@git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3#egg=s3prl`

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

#### Example 2

```bash
git clone https://github.com/Sanyuan-Chen/UniSpeech.git -b t-schen/asv_eval
cd UniSpeech/downstreams/speaker_verification 
pip install scipy==1.7.1 fire==0.4.0 sklearn==0.0 s3prl==0.3.1 torchaudio==0.9.0 sentencepiece==0.1.96 
pip install s3prl@git+https://github.com/s3prl/s3prl.git@7ab62aaf2606d83da6c71ee74e7d16e0979edbc3#egg=s3prl 
wget "https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/sv_finetuned/wavlm_large_finetune.pth?sv=2020-08-04&st=2022-12-02T09%3A48%3A45Z&se=2024-12-03T09%3A48%3A00Z&sr=b&sp=r&sig=jQPnEO9I5JqtoWylCvHIU0IvUxZ8jzC%2F64%2B6%2Fa1%2FKE4%3D" -O wavlm_large_finetune.pth
 
python verification_tsv.py $tsv1 $tsv2 --model_name wavlm_large --checkpoint wavlm_large_finetune.pth --scores $score_file --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1

```

If an error in hubconf.py raised, replace the file with utils/hubconf.py

tsv file example
```bash
root_path
wav1
wav2
...
```
