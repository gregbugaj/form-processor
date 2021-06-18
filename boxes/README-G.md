
## Training steps

CUDA_VISIBLE_DEVICES=1 python train.py --dataset=ICDAR2015 --batch_size=3 --multi_scale=True --logdir=logs/multi_step1/ --save_folder=weights/multi_step1/ --num_workers=4  --input_size 608


-- STEP 1 608x608
CUDA_VISIBLE_DEVICES=1 python train.py --dataset=ICDAR2015 --batch_size=3 --logdir=logs/multi_step1/ --save_folder=weights/multi_step1/ --num_workers=4  --resume=weights/multi_step1/ckpt_500.pth --input_size 608


-- STEP 2 960x960
CUDA_VISIBLE_DEVICES=1 python train.py --dataset=ICDAR2015 --batch_size=3 --logdir=logs/multi_step2/ --save_folder=weights/multi_step2/ --num_workers=4 --input_size 960  --resume=weights/multi_step1/best_segmenter.pth


CUDA_VISIBLE_DEVICES=1 python train.py --dataset=ICDAR2015 --batch_size=3 --logdir=logs/multi_step3/ --save_folder=weights/multi_step3/ --num_workers=4 --input_size 1280  --resume=weights/multi_step2/best_segmenter.pth

