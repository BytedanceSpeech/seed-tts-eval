## Pre-training Representations for Speaker Diarization

### Downstream Model

[EEND-vector-clustering](https://arxiv.org/abs/2105.09040)

### Pre-trained models

- It should be noted that the diarization system is trained on 8k audio data.

| Model                                                        | 2 spk DER | 3 spk DER | 4 spk DER | 5 spk DER | 6 spk DER | ALL spk DER |
| ------------------------------------------------------------ | --------- | --------- | --------- | --------- | --------- | ----------- |
| EEND-vector-clustering                                       | 7.96      | 11.93     | 16.38     | 21.21     | 23.1      | 12.49       |
| [**UniSpeech-SAT large**](https://drive.google.com/file/d/16OwIyOk2uYm0aWtSPaS0S12xE8RxF7k_/view?usp=sharing) | 5.93      | 10.66     | 12.90     | 16.48     | 23.25     | 10.92       |

### How to use?

#### Environment Setup

1. `pip install --require-hashes -r requirements.txt`
2. Install fairseq code
   - For UniSpeech-SAT large, we should install the [Unispeech-SAT](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech-SAT) fairseq code.

#### Example

1. First, you should download the pre-trained model in the above table to `checkpoint_path`.
2. Then, run the following codes:
   - The wav file is the multi-talker simulated speech from Librispeech corpus.
3. The output will be written in `out.rttm` by  default.

```bash
python diarization.py --wav_path tmp/mix_0000496.wav --model_init $checkpoint_path
```

