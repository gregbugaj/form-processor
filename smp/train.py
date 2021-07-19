from __future__ import print_function

import os
import sys
import argparse
import torch
import numpy as np
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
import albumentations as albu

from torch.utils.data import DataLoader

from dataset import Dataset

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
from adabound.adabound import AdaBound

# basic constants

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['foreground']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

ALPHA=.7
BETA=.3
GAMMA=2

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        self.focal_loss = FocalTverskyLoss()
        self.__name__ = 'custom_loss'

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        criterion = dice_loss + (1 * focal_loss)
                       
        return criterion        


def get_training_augmentation(pad_size, crop_size):
    """Training augmentation 
        Args:
            pad_size  (w, h)
            crop_size (w, h)
    """

    train_transform = [
        albu.PadIfNeeded(min_height=pad_size[1], min_width=pad_size[0], always_apply=True, border_mode=0),
        albu.RandomCrop(height=crop_size[1], width=crop_size[0], always_apply=True),
    ]
    
    return albu.Compose(train_transform)

def get_validation_augmentation(pad_size):
    """Add paddings to make image shape divisible by 32"""

    test_transform = [
        albu.PadIfNeeded(min_height=pad_size[1], min_width=pad_size[0], always_apply=True, border_mode=0), # HCFA04
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
]

def build_model(args, device, device_ids=[0], ckpt=None):
    print('==> Building model..')

    net = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
        decoder_attention_type='scse',
        # in_channels=1,
    )

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True

    if ckpt:
        net.load_state_dict(ckpt['net'])

    return net

def build_dataset(data_dir, pad_size, crop_size):
    print(f'==> Preparing data..')    

    x_train_dir = os.path.join(data_dir, 'train/image')
    y_train_dir = os.path.join(data_dir, 'train/mask')

    x_valid_dir = os.path.join(data_dir, 'test/image')
    y_valid_dir = os.path.join(data_dir, 'test/mask')

    print(f'==> dir  : {data_dir}')
    print(f'==> size : {pad_size}')
    print(f'==> crop : {crop_size}')
    
    # Lets look at data we have
    if True:
        dataset = Dataset(x_train_dir, y_train_dir, size=pad_size)
        image, mask = dataset[3] # get some sample
        visualize(image=image, mask=mask.squeeze())

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(pad_size=pad_size, crop_size=crop_size), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        size=pad_size
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(pad_size=pad_size), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
        size=pad_size
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # #### Visualize resulted augmented images and masks
    if True:
        augmented_dataset = Dataset(
            x_train_dir, 
            y_train_dir, 
            augmentation=get_training_augmentation(pad_size=pad_size, crop_size=crop_size),
            size=pad_size
        )

        # same image with different random transforms
        for i in range(3):
            idx = np.random.randint(len(augmented_dataset))
            image, mask = augmented_dataset[idx]
            visualize(image=image, mask=mask.squeeze(-1))

    return train_loader, valid_loader


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch training')
    parser.add_argument('--model', default='unetplusplus', type=str, help='model',
                        choices=['unetplusplus'])
    parser.add_argument('--optim', default='sgd', type=str, help='optimizer',
                        choices=['sgd', 'adagrad', 'adam', 'amsgrad', 'adabound', 'amsbound'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--final_lr', default=0.1, type=float,
                        help='final learning rate of AdaBound')
    parser.add_argument('--gamma', default=1e-3, type=float,
                        help='convergence speed term of AdaBound')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum term')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay for optimizers')
    return parser

def get_ckpt_name(model='resnet', optimizer='adabound', lr=0.1, final_lr=0.1, momentum=0.9,
                  beta1=0.9, beta2=0.999, gamma=1e-3):
    name = {
        'sgd': 'lr{}-momentum{}'.format(lr, momentum),
        'adagrad': 'lr{}'.format(lr),
        'adam': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adamw': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'amsgrad': 'lr{}-betas{}-{}'.format(lr, beta1, beta2),
        'adabound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
        'amsbound': 'lr{}-betas{}-{}-final_lr{}-gamma{}'.format(lr, beta1, beta2, final_lr, gamma),
    }[optimizer]
    return '{}-{}-{}.pth'.format(model, optimizer, name)
    
def load_checkpoint(ckpt_name):
    print('==> Resuming from checkpoint..')
    path = os.path.join('checkpoint', ckpt_name)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    assert os.path.exists(path), 'Error: checkpoint {} not found'.format(ckpt_name)
    return torch.load(path)

def create_optimizer(args, model_params):
    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(model_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)    
    elif args.optim == 'adamw':
        return optim.AdamW(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(model_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'adabound':
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return AdaBound(model_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay, amsbound=True)

def main():
    parser = get_parser()
    args = parser.parse_args()

    args.optim = 'adabound'
    args.optim = 'adamw'
    args.lr = 1e-4
    args.final_lr = 0.1
    args.gamma = 0.001
    args.resume = False

    # HCFA04
    data_dir = '/home/greg/dev/unet-denoiser/data_HCFA04_finetune/'
    pad_size = (1024, 192)
    crop_size = (256, 160)

    train_loader, test_loader = build_dataset(data_dir, pad_size, crop_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device  : {device}')
    
    ckpt_name = get_ckpt_name(model=args.model, optimizer=args.optim, lr=args.lr,
                              final_lr=args.final_lr, momentum=args.momentum,
                              beta1=args.beta1, beta2=args.beta2, gamma=args.gamma)
    if args.resume:
        ckpt =  load_checkpoint(ckpt_name)
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
    else:
        ckpt = None
        best_acc = 0
        start_epoch = -1

    net = build_model(args, device, device_ids=[0], ckpt=ckpt)

    loss = CustomLoss()
    loss._name = 'custom_loss'

    print(f'start_epoch : {start_epoch}')
    start_epoch = -1

    optimizer = create_optimizer(args, net.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, last_epoch=-1)

    # # prevent : KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
    # for i in range(start_epoch):
    #     scheduler.step()

    train_accuracies = []
    test_accuracies = []

    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        net, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    test_epoch = smp.utils.train.ValidEpoch(
        net, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    best_acc = 0
    
    for epoch in range(start_epoch+1, 120):
        print('\nEpoch: {}'.format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = test_epoch.run(test_loader)
        
        scheduler.step()
        train_acc = train_logs['iou_score']
        test_acc = valid_logs['iou_score']

        # Save checkpoint.
        if test_acc > best_acc:
            print(f'Saving.. : {test_acc}')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join('checkpoint', ckpt_name))
            torch.save(net, './best_model.pth')
            # torch.save(net, os.path.join('checkpoint', 'best_model.pth'))
            best_acc = test_acc

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if not os.path.isdir('curve'):
            os.mkdir('curve')
            
        torch.save({'train_acc': train_accuracies, 'test_acc': test_accuracies},
                   os.path.join('curve', ckpt_name))

if __name__ == '__main__':
    main()        