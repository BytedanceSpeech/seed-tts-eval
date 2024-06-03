
# ILS-SSL

> [**ILS-SSL**](https://arxiv.org/pdf/2112.08778.pdf): Self-Supervised Learning for Speech Recognition with Intermediate Layer Supervision

The data preparation and pre-training for the first iteration follow the same pipeline as Hubert. We give example scripts for ILS-Hubert pre-training and fine-tuning in src/examples/hubert/scripts

## Pre-Trained and Fine-tuned Models
Model | Pretraining Dataset | Finetuning Dataset | Model
|---|---|---|---
ILS-Base | 960h LibriSpeech | - | [Download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/el_hubert_4_12/checkpoint_best.pt?st=2022-01-04T08%3A05%3A24Z&se=2024-01-05T08%3A05%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=JI8ZOgBhrrKUY4DE2ommnKpyAUuX6OrHfWgdjAT2Xnc%3D)
ILS-Large | 60k hrs Libri-Light | - | [Download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/ils_hubert_large/checkpoint_fixed.pt?st=2022-01-04T08%3A24%3A37Z&se=2025-01-05T08%3A24%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=Dv6svAaI7Td%2BZWUTjTFkhChFbpnAAU6xKNjPbPQnIKM%3D)
ILS-Large | 60k hrs Libri-Light | 960h LibriSpeech | [Download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/ils_hubert_large/checkpoint_ft.pt?st=2022-01-04T08%3A40%3A17Z&se=2025-01-05T08%3A40%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=GKIe%2F1kz%2F1fjGTsQsakJy68jlsFDbKmIVYjH61dhrwA%3D)


## Results on Librispeech
Base Model | Finetuning set|  LM | test-clean | test-other
|---|---|---|---|---
wav2vec2.0  | 1 hour | None |  24.5 | 29.7
Hubert  | 1 hour | None| 20.9 | 27.5
ILS-SSL  | 1 hour | None | 17.9 | 23.1
wav2vec2.0  | 1 hour | 4-gram | 5.5 | 11.3
Hubert  | 1 hour | 4-gram | 6.1 | 11.3
ILS-SSL  | 1 hour | 4-gram | 5.4 | 10.2
wav2vec2.0  | 10 hour | None | 11.1 | 17.6
Hubert  | 10 hour | None| 10.1 | 16.8
ILS-SSL  | 10 hour | None | 8.3 | 13.6
wav2vec2.0  | 10 hour | 4-gram | 4.3 | 9.5
Hubert  | 10 hour | 4-gram | 4.3 | 9.4
ILS-SSL  | 10 hour | 4-gram | 3.8 | 8.1
wav2vec2.0  | 100 hour | None | 6.1 | 13.3
Hubert  | 100 hour | None| 6.3 | 13.2
ILS-SSL  | 100 hour | None | 4.7 | 10.1
wav2vec2.0  | 100 hour | 4-gram | 3.4| 8.0
Hubert  | 100 hour | 4-gram | 3.4 | 8.1
ILS-SSL  | 100 hour | 4-gram | 3.0 | 6.9

Large Model | Finetuning set|  LM | test-clean | test-other
|---|---|---|---|---
wav2vec2.0  | 1 hour | None |  17.2 | 20.3
Hubert  | 1 hour | None| 17.4 | 20.3
ILS-SSL  | 1 hour | None | 14.3 | 16.9 
wav2vec2.0  | 1 hour | Transf | 2.9 | 5.8
Hubert  | 1 hour | Transf | 2.9 | 5.4
ILS-SSL  | 1 hour | Transf | 2.8 | 5.3
wav2vec2.0  | 10 hour | None | 6.3 | 10.0
Hubert  | 10 hour | None | 6.2 | 9.6
ILS-SSL  | 10 hour | None | 6.1 | 9.1
wav2vec2.0  | 10 hour | Transf | 2.6 | 4.9
Hubert  | 10 hour | Transf |  2.4 | 4.6
ILS-SSL  | 10 hour | Transf | 2.5 | 4.5
wav2vec2.0  | 100 hour | None | 3.1 | 6.3
Hubert  | 100 hour | None| 2.9 | 6.0
ILS-SSL  | 100 hour | None | 2.9 | 5.8
wav2vec2.0  | 100 hour | Transf | 2.0 | 4.0
Hubert  | 100 hour | Transf | 2.1 | 3.9
ILS-SSL  | 100 hour | Transf | 2.0 | 4.0
wav2vec2.0  | 960 hour | None | 2.2 | 4.5
Hubert  | 960 hour | None | 2.1 | 4.3
ILS-SSL  | 960 hour | None | 1.9 | 3.8
wav2vec2.0  | 960 hour | Transf | 1.8 | 3.3
Hubert  | 960 hour | Transf | 1.9 | 3.3
ILS-SSL  | 960 hour | Transf | 1.8 | 3.2
