# Learning the Beauty in Songs: Neural Singing Voice Beautifier
---
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/NeuralSVB)](https://github.com/MoonInTheRiver/NeuralSVB)
![visitors](https://visitor-badge.glitch.me/badge?page_id=moonintheriver/NeuralSVB)

<div align="center">
    <a href="https://neuralsvb.github.io" target="_blank">Demo&nbsp;Page</a>
</div>


This repository is the official PyTorch implementation of our ACL-2022 [paper](https://arxiv.org/abs/2202.13277). 


## 0. Dataset (PopBuTFy) Acquirement
### Audio samples
- See in [apply_form](resources/apply_form.md).
- Dataset [preview](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/PopBuTFy-preview.zip).

### Text labels
NeuralSVB does not need text as input, but the ASR model to extract PPG needs text. Thus we also provide the [text labels](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/text_labels.zip) of PopBuTFy. 
<!-- We recommend mixing [LibriTTS](https://www.openslr.org/60/) with PopBuTFy to train the ASR model. -->

## 1. Preparation

### Environment Preparation
WIP.

### Data Preparation


1. Extract embeddings of vocal timbre:
    ```sh 
    CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config egs/datasets/audio/PopBuTFy/save_emb.yaml
    ```
2. Pack the dataset:
    ```sh 
    CUDA_VISIBLE_DEVICES=0 python data_gen/tts/bin/binarize.py --config egs/datasets/audio/PopBuTFy/para_bin.yaml
    ```


### Vocoder Preparation
We provide the pre-trained model of [HifiGAN-Singing](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/1012_hifigan_all_songs_nsf.zip) which is specially designed for SVS with NSF mechanism.

Please unzip pre-trained vocoder into `checkpoints` before training your acoustic model.

This singing vocoder is trained on 100+ hours singing data (including Chinese and English songs). 

### PPG Extractor Preparation
We provide the pre-trained model of [PPG Extractor](https://github.com/MoonInTheRiver/NeuralSVB/releases/download/pre-release/1009_pretrain_asr_english.zip).

Please unzip pre-trained PPG extractor into `checkpoints` before training your acoustic model.


After the instructions above, the directory structure should be as follows:

```
.
|--data
    |--processed
        |--PopBuTFy (unzip PopBuTFy.zip)
            |--data
                |--directories containing wavs
    |--binary
        |--PopBuTFyENSpkEM
|--checkpoints
    |--1009_pretrain_asr_english
        |--
        |--config.yaml
    |--1012_hifigan_all_songs_nsf
        |--
        |--config.yaml
```


## 2. Training Example

```sh
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml --exp_name exp_name --reset
```

## 3. Inference
### Inference from packed test set

```sh
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config egs/datasets/audio/PopBuTFy/vae_global_mle_eng.yaml --exp_name exp_name --reset --infer
```
Inference results will be saved in `./checkpoints/EXP_NAME/generated_` by default.

We will also provide:
 - the pre-trained model of NSVB (WIP);

Remember to put the pre-trained models in `checkpoints` directory.

### Inference from raw inputs
WIP.

## Limitations
See Appendix D "Limitations and Solutions" in our [paper](https://aclanthology.org/2022.acl-long.549.pdf).

## Citation
If this repository helps your research, please cite:

    @inproceedings{liu-etal-2022-learning-beauty,
    title = "Learning the Beauty in Songs: Neural Singing Voice Beautifier",
    author = "Liu, Jinglin  and
      Li, Chengxi  and
      Ren, Yi  and
      Zhu, Zhiying  and
      Zhao, Zhou",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.549",
    pages = "7970--7983",}


## Issues
 - Before raising a issue, please check our Readme and other issues for possible solutions.
 - We will try to handle your problem in time but we could not guarantee a satisfying solution.
 - Please be friendly.

## Acknowledgements
* r9y9's [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
* Po-Hsun-Su's [ssim](https://github.com/Po-Hsun-Su/pytorch-ssim)
* descriptinc's [melgan](https://github.com/descriptinc/melgan-neurips)
* Official [espnet](https://github.com/espnet/espnet)
* Official [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)

The framework of this repository is based on [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger), 
and is a predecessor of [NATSpeech](https://github.com/NATSpeech/NATSpeech/). 
