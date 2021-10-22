# Form segmentation and processing framework


## Installation 

Follow instructions from `pytorch` website

```sh
https://pytorch.org/get-started/locally/
```

```sh
$ pip install -r requirements.txt
```


To install latest segmentaion models


```sh
$ pip install git+https://github.com/qubvel/segmentation_models.pytorch
```


ICR Accuracy : 97.87234

## Generating form layout 

## HICFA Segmentation Hyperparams

Data prep

```sh
python ./datasets/prepare_patches_dataset.py  --input_dir ./datasets/hicfa --output_dir ./datasets/hicfa/ready
```

Training

```
python train.py --dataroot ./datasets/hicfa/ready --name hicfa_pix2pix --model pix2pix --direction AtoB --gpu_ids 0,1 --no_flip --batch_size 6 --display_freq 100  --preprocess none  --display_freq 100 --lr 0.0002 --save_epoch_freq 1 --load_size 1024  --crop_size 1024 --output_nc 3 --input_nc 3 --save_epoch_freq 1  --save_latest_freq 2000 --save_epoch_freq 1 --lr .0002 --display_env hicfa --n_epochs 300 --netG unet_1024 
```

Test 

```
python test.py --dataroot ./datasets/hicfa/eval_1024 --name hicfa_pix2pix --model test --netG unet_1024 --direction AtoB --dataset_mode single --gpu_id -1 --norm batch  --load_size 1024 --crop_size 1024
```

# Ref

one hot encoding
https://www.kaggle.com/balraj98/deeplabv3-resnet101-for-segmentation-pytorch