import os

# path to the data directories
data_dir = 'data-patches-2'
train_dir = 'train'

val_dir = 'test'
imgs_dir = 'mask'

noisy_dir = 'image'
debug_dir = 'debug'

patch_dir = './assets-gen/patches/forms'
txt_file_dir = './assets/text.txt'

# maximun number of synthetic words to generate
num_synthetic_imgs = 100
train_percentage = 0.8

test_dir = os.path.join(data_dir, val_dir, noisy_dir)

