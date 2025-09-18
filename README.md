# FastPoint
This is an implementation of our paper "FastPoint: Accelerating 3D Point Cloud Model Inference via Sample Point Distance Prediction".

## Install
```
source install.sh
```
Note:

   We recommend using CUDA 11.x; check your CUDA version by: `nvcc --version` before using the bash file;

## Prepare Pretrained Models
Model weights used in our experiments are available through Google Drive [link](https://drive.google.com/drive/folders/1obELZpNd6U5XEn08eHPK0tMDy_rzfj8H).

## Prepare Dataset

#### S3DIS
```
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3disfull.tar
```

#### ScanNet
```
cd data
gdown https://drive.google.com/uc?id=1uWlRPLXocqVbJxPvA2vcdQINaZzXf1z_
tar -xvf ScanNet.tar
```

#### SemanticKITTI
Download [SemanticKITTI](https://www.semantic-kitti.org/dataset.html#download) dataset.


## Experiments
#### Inference Speed (val mode)
```
# Run baseline
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l, pointmetabase-l].yaml wandb.use_wandb=False mode=val --pretrained_path [path]

# Run with only MDPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l-mdps, pointmetabase-l-mdps].yaml wandb.use_wandb=False mode=val --pretrained_path [path]

# Run with FastPoint
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l-fastpoint, pointmetabase-l-fastpoint].yaml wandb.use_wandb=False mode=val --pretrained_path [path]
```

#### Model Accuracy (test mode)
```
# Run baseline
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l, pointmetabase-l].yaml wandb.use_wandb=False mode=test --pretrained_path [path]

# Run with only MDPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l-mdps, pointmetabase-l-mdps].yaml wandb.use_wandb=False mode=test --pretrained_path [path]

# Run with FastPoint
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/[s3dis, scannet, semantickitti]/[pointvector-l-fastpoint, pointmetabase-l-fastpoint].yaml wandb.use_wandb=False mode=test --pretrained_path [path]
```

## Training Minimum Distance Curve Estimator
We provide example training script for S3DIS estimator. Detailed hyperparameters for other datasets are explained in Appendix A.3.
```
CUDA_VISIBLE_DEVICES=0 bash script/train_estimator.sh cfgs/s3dis/pointmetabase-l.yaml wandb.use_wandb=False
```

## Acknowledgment
This repository is built on reusing codes of [Frugal\_PN\_Training](https://github.com/SNU-ARC/Frugal_PN_Training.git), [PointMetaBase](https://github.com/linhaojia13/PointMetaBase), [OpenPoints](https://github.com/guochengqian/openpoints) and [PointNeXt](https://github.com/guochengqian/PointNeXt).

## Citation
```tex
@inproceedings {fastpoint,
    title={FastPoint: Accelerating 3D Point Cloud Model Inference via Sample Point Distance Prediction},
    author={Donghyun Lee and Dawoon Jeong and Jae W. Lee and Hongil Yoon},
    booktitle = {IEEE/CVF International Conference on Computer Vision ({ICCV} 25)},
    year={2025},
}
```
