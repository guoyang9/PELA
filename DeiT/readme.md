# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Parameter preparation
Our work requires the checkpoints of pretrained models. Specifically, download the ImageNet-pretrained checkpoints in to `params` directory:
```
mkdir params && cd params
wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
wget https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth
```

## Training
To train PELA with Deit-Base, simply run
```
bash scripts/train_base.sh
```

To train PELA with Deit-III-Large, run
```
bash scripts/train_large.sh
```


