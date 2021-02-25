# Transformer-UVOS
This is a repo offers a strong baseline for Unsupervised Video Object Segmentation(UVOS) using Transformer

## Accuracy
With Res50 as backbone and image size (640x352) during training, using only single-scale for testing
On **DAVIS 2016(val)**:

Method | J_mean | J_recall | J_decay | F_mean | F_recall | F_decay
-- | -- | -- | -- | -- | -- | -- 
Ours(Res50-FPN, 640x352) | 0.777 | 0.915 | 0.066 | 0.766 | 0.859 | 0.043
Anchor-Diffusion(Res101-Deeplabv3, 854x480) | 0.782 | --- | --- | 0.771 | --- | ---

## Architecture
ResNet50-FPN + Transformer + Simple Decoder

## Training
You need to first download the [DAVIS dataset](https://davischallenge.org/index.html)

```
cd model/transformer/davis.transformer.fpn.R50.random_sample/
sh run.sh
```

## Acknowledgement 
Idea inspired by [Anchor-Diffusion](https://arxiv.org/abs/1910.10895).

Codes based on [TorchSeg](https://github.com/ycszen/TorchSeg), [Detectron2](https://github.com/facebookresearch/detectron2) and [Detr](https://github.com/facebookresearch/detr)

## Contact
Feel free to contact me if you have any questions : yuming.du@enpc.fr
