# UniSpeech

<!--**Pre-trained models for speech related tasks**-->

The family of UniSpeech:
> [**WavLM**](https://arxiv.org/pdf/2110.13900.pdf) (```arXiv```): **WavLM: Large-Scale Self-Supervised  Pre-training   for Full Stack Speech Processing**

> [**UniSpeech**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech) (```ICML 2021```): **Unified Pre-training for Self-Supervised Learning and Supervised Learning for ASR**

> [**UniSpeech-SAT**](https://arxiv.org/pdf/2110.05752.pdf) (```ICASSP 2022 Submission```): **Universal Speech Representation Learning with  Speaker Aware Pre-Training**

> [**ILS-SSL**](https://arxiv.org/pdf/2112.08778.pdf) (```ICASSP 2022 Submission```): **Self-Supervised Learning for Speech Recognition with Intermediate Layer Supervision**

Model introductions, evaluation results, and model inference instructions are located in their corresponding folders. The source code is here [https://github.com/microsoft/UniSpeech/tree/main/src].

## Update
- [HuggingFace Integration] Dec 23, 2021: [**WavLM**](https://huggingface.co/models?other=wavlm)  models are on [HuggingFace](https://huggingface.co/models?other=wavlm) . 
- [HuggingFace Integration] Octorber 26, 2021: [**UniSpeech-SAT**](https://huggingface.co/microsoft/unispeech-sat-large)  models are on [HuggingFace](https://huggingface.co/models?other=unispeech-sat) . 
- [Model Release] Octorber 13, 2021: [**UniSpeech-SAT**](https://arxiv.org/pdf/2110.05752.pdf) models are releaseed.
- [HuggingFace Integration] Octorber 11, 2021: [**UniSpeech**](https://huggingface.co/microsoft/unispeech-large-1500h-cv)  models are on [HuggingFace](https://huggingface.co/models?other=unispeech) . 
- [Model Release] June, 2021: [**UniSpeech v1**](https://github.com/microsoft/UniSpeech/tree/main/UniSpeech) models are released.
## Pre-trained models
We strongly suggest using our UniSpeech-SAT model for speaker related tasks, since it shows very powerful performance on various speaker related benchmarks.
Model | Pretraining Dataset | Finetuning Dataset | Model
|---|---|---|-----
UniSpeech Large EN |  [Labeled: 1350 hrs en](https://commonvoice.mozilla.org/) | - |  [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceEnglishPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A07Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=5sxvEwVRoGtkazNQYkOuFLlPYau8nl5Ng%2FfRJa0Vnc4%3D)
UniSpeech Large Multilingual |  [Labeled: 1350 hrs en + 353 hrs fr + 168 hrs es + 90 hrs it](https://commonvoice.mozilla.org/) | - | [download](https://releasemodel.blob.core.windows.net/models/CommonVoicePretrainedModel/CommonVoiceMultilingualPretrainedModel/checkpoint_best.pt?sv=2019-12-12&st=2021-07-14T09%3A00%3A39Z&se=2022-07-15T09%3A00%3A00Z&sr=b&sp=r&sig=y%2Fd3rqtbyqW0ZCwR7Czho5any90khA%2Ft3w9PTZ6N9vU%3D)
Unispeech Large+ | [Labeled: 1350 hrs en, Unlabeled: 353 hrs fr](https://commonvoice.mozilla.org/) | - | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/pt_fr353.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A44%3A54Z&se=2023-10-26T06%3A44%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=7tYuYMxVFfM2Vgi%2BoqUh%2ByJXD4hSuoafHgBP5VZApw0%3D)
UniSpeech Large+ | [Labeld: 1350 hrs en, Unlabeled: 168 hrs es](https://commonvoice.mozilla.org/) | - | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/pt_es168.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A39%3A37Z&se=2023-10-26T06%3A39%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=T2B5%2BlOI6v64TNdLSe9rdp3R%2B9Q2E35taUOigGW0nsQ%3D)
UniSpeech Large+ | [Labeled: 1350 hrs en, Unlabeld: 90 hrs it](https://commonvoice.mozilla.org/) | -| [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/pt_it90.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A52%3A08Z&se=2023-10-26T06%3A52%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=kXsSJXK9r8UEYlUr2LaJxtPf8m9J2G23MfG725k2DBk%3D)
UniSpeech Large Multilingual |  [Labeled: 1350 hrs en + 353 hrs fr + 168 hrs es + 90 hrs it, Unlabeled: 17 hrs ky](https://commonvoice.mozilla.org/) | - | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/pt_ky17.large.many2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A53%3A00Z&se=2022-10-26T06%3A53%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=oCQecalXzC5daaurLLJGQdFNtfYwsBM6pNQrDAsf5i0%3D)
UniSpeech Large+ | [Labeled: 1350 hrs en, Unlabeled: 353 hrs fr](https://commonvoice.mozilla.org/) | 1 hr fr | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/ft_fr-pt_fr353.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A27%3A53Z&se=2023-10-26T06%3A27%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=9vEa3xqzWu7SYkACn9TQqDtcm%2BKmUcOHhabjbjZuPys%3D)
UniSpeech Large+ | [Labeld: 1350 hrs en, Unlabeled: 168 hrs es](https://commonvoice.mozilla.org/) | 1 hr es | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/ft_es-pt_es168.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A21%3A34Z&se=2024-10-26T06%3A21%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=G%2B0RddgOh653UzXG95Ljuwv7aG3tu9gXtPXn1ixCiug%3D)
UniSpeech Large+ | [Labeled: 1350 hrs en, Unlabeld: 90 hrs it](https://commonvoice.mozilla.org/) | 1 hr it | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/ft_it-pt_it90.large.one2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A36%3A17Z&se=2023-10-26T06%3A36%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=e1WD9uOCo9sCAdH%2FPZQ4wCD30aCDpZvvu43kJrqq2HE%3D)
UniSpeech Large Multilingual |  [Labeled: 1350 hrs en + 353 hrs fr + 168 hrs es + 90 hrs it, Unlabeled: 17 hrs ky](https://commonvoice.mozilla.org/) | 1 hr ky | [download](https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-chengw/models/pt_ky17.large.many2one_unispeech/checkpoint_best.pt?st=2021-10-25T06%3A54%3A04Z&se=2023-10-26T06%3A54%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=2K3VjMcsbKfBkLVyDlqGhVpIX%2B2ZcA5DTlMhjdkXo3g%3D)
UniSpeech-SAT Base |  [960 hrs LibriSpeech](http://www.openslr.org/12) | - | [download](https://drive.google.com/file/d/1l5etRW6W2aP_8I2Fs_8ailGZqEzdrAPz/view?usp=sharing)
UniSpeech-SAT Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | - | [download](https://drive.google.com/file/d/1Q1MLVfyOHkSzTjyD-mzSZVjhndEmCvef/view?usp=sharing)
UniSpeech-SAT Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main) | - | [download](https://drive.google.com/file/d/12ScE1G2W-AHcccyBb_0uVI6qpFVQ0PaI/view?usp=sharing)
WavLM Base |  [960 hrs LibriSpeech](http://www.openslr.org/12)| -  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-04-08&st=2021-11-05T00%3A35%3A31Z&se=2022-11-06T00%3A35%3A00Z&sr=b&sp=r&sig=JljnRVzyHY6AjHzhVmHV5KyQQCvvGfgp9D2M02oGJBU%3D) <br> [Google Drive](https://drive.google.com/file/d/19-C7SMQvEFAYLG5uc47NX_MY03JCbI4x/view?usp=sharing)
WavLM Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  |  [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-04-08&st=2021-11-05T00%3A34%3A47Z&se=2022-10-06T00%3A34%3A00Z&sr=b&sp=r&sig=Gkf1IByHaIn1t%2FVEd9D6WHjZ3zu%2Fk5eSdoj21UytKro%3D) <br> [Google Drive](https://drive.google.com/file/d/1PlbT_9_B4F9BsD_ija84sUTVw7almNX8/view?usp=sharing) 
WavLM Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  | [Azure Storage](https://msranlcmtteamdrive.blob.core.windows.net/share/wavlm/WavLM-Large.pt?sv=2020-08-04&st=2021-11-22T10%3A03%3A53Z&se=2022-11-23T10%3A03%3A00Z&sr=b&sp=r&sig=3kB8dwTCyIS8YQ7gW5oXmDrXV%2FAaLmoxBS37oPpFsz4%3D) <br> [Google Drive](https://drive.google.com/file/d/1rMu6PQ9vz3qPz4oIm72JDuIr5AHIbCOb/view?usp=sharing) 

## Universal Representation Evaluation on SUPERB 
![alt text](WavLM/WavLM_SUPERB_Results.png)

## Downstream Task Performance 
We also evaluate our models on typical speaker related benchmarks.
### Speaker Verification
Finetune the model with VoxCeleb2 dev data, and evaluate it on the [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/#:~:text=VoxCeleb%20is%20an%20audio%2Dvisual,interview%20videos%20uploaded%20to%20YouTube)
| Model         |Fix pre-train| Vox1-O | Vox1-E     | Vox1-H         |
| ------------- |------------- | ---------- | ---------- | ---------- |
| ECAPA-TDNN   | - | 0.87     | 1.12  | 2.12   |
| HuBERT large  | Yes|  0.888	|0.912|	1.853 |
| Wav2Vec2.0 (XLSR)| Yes | 0.915|	0.945	|1.895|
| UniSpeech-SAT large | Yes | 0.771	| 0.781|	1.669|
| WavLM large | Yes | 0.59	| 0.65|	1.328|
| WavLM large | No | 0.505	| 0.579|	1.176|
|+Large Margin Finetune and Score Calibration|
| HuBERT large | No| 0.585|	0.654	|1.342|   
| Wav2Vec2.0 (XLSR) | No| 0.564|	0.605	|1.23|   
| UniSpeech-SAT large | No | 0.564 | 0.561| 1.23 |
| **WavLM large (New)** | No | **0.33** | **0.477**| **0.984** |

[Large-scale Self-Supervised Speech Representation Learning for Automatic Speaker Verification](https://arxiv.org/pdf/2110.05777.pdf)



### Speech Separation

Evaluation on [LibriCSS](https://github.com/chenzhuo1011/libri_css)
| Model         |0S | 0L | OV10     |      OV20     |OV30 |OV40 |
| ---------------- |------| ------ | ------ | ------ | ------ | ------ |
| [Conformer](https://ieeexplore.ieee.org/abstract/document/9413423/) (SOTA)   | 4.5	| 4.4	|6.2	|8.5|	11	|12.6|
| UniSpeech-SAT base | 4.4|	4.4	|5.4|	7.2|	9.2	|10.5|
| UniSpeech-SAT large | 4.3|	4.2	|5.0	|6.3|	8.2|	8.8|
| WavLM base+ | 4.5|	4.4	|5.6|	7.5|	9.4	|10.9|
| **WavLM large** | 4.2| 4.1	| 4.8	| 5.8 |	7.4|	8.5|


### Speaker Diarization

Evaluation on CALLHOME
| Model         |spk_2	|spk_3|	spk_4|	spk_5|	spk_6|	spk_all |
| ---------------- |------| ------ | ------ | ------ | ------ | ------ |
| [EEND-vector clustering](https://arxiv.org/pdf/2105.09040.pdf)   | 7.96|	11.93	|16.38|	21.21|	23.1	|12.49||
| [EEND-EDA clustering](https://arxiv.org/abs/2107.01545) (SOTA)  | 7.11|	11.88 |14.37|	25.95|	21.95	|11.84||
| UniSpeech-SAT large | 5.93|	10.66|	12.9	|16.48|	23.25|	10.92|
| WavLM Base| 6.99|	11.12|	15.20	|16.48|	21.61|	11.75|
| **WavLm large** | 6.46|	10.69|	11.84	|12.89|	20.70|	10.35|


## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [FAIRSEQ](https://github.com/pytorch/fairseq) project.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


### Reference
If you find our work is useful in your research, please cite the following paper:
``` latex
@inproceedings{Wang2021UniSpeech,
  author    = {Chengyi Wang and Yu Wu and Yao Qian and Kenichi Kumatani and Shujie Liu and Furu Wei and Michael Zeng and Xuedong Huang},
  editor    = {Marina Meila and Tong Zhang},
  title     = {UniSpeech: Unified Speech Representation Learning with Labeled and
               Unlabeled Data},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {10937--10947},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {http://proceedings.mlr.press/v139/wang21y.html},
  timestamp = {Thu, 21 Oct 2021 16:06:12 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/0002WQK0WZ021.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

``` latex
@article{Chen2021WavLM,
  title   = {WavLM: Large-Scale Self-Supervised  Pre-training   for Full Stack Speech Processing},
  author  = {Sanyuan Chen and Chengyi Wang and Zhengyang Chen and Yu Wu and Shujie Liu and Zhuo Chen and Jinyu Li and Naoyuki Kanda and Takuya Yoshioka and Xiong Xiao and Jian Wu and Long Zhou and Shuo Ren and Yanmin Qian and Yao Qian and Jian Wu and Michael Zeng and Furu Wei},
  eprint={2110.13900},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2021}
}
```

``` latex
@article{Chen2021UniSpeechSAT,
  title   = {UniSpeech-SAT: Universal Speech Representation Learning with  Speaker Aware Pre-Training},
  author  = {Sanyuan Chen and Yu Wu and Chengyi Wang and Zhengyang Chen and Zhuo Chen and Shujie Liu and   Jian Wu and Yao Qian and Furu Wei and Jinyu Li and  Xiangzhan Yu},
  eprint={2110.05752},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  year={2021}
}
```


### Contact Information

For help or issues using UniSpeech models, please submit a GitHub issue.

For other communications related to UniSpeech, please contact Yu Wu (`yuwu1@microsoft.com`).
