## Setup

Clone the repo
```sh
git clone git@github.com:ez2rok/meru.git
cd meru
```

Install the packages via conda
```sh
conda create -n meru python pytorch torchvision
conda activate meru
```
and with python/pip
```sh
pip3 install torch torchvision tensorflow wandb
python -m pip install --pre timm
python -m pip install -r requirements.txt
python setup.py develop
```

Download the small and large meru and clip models
```sh
wget https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth -P checkpoints
wget https://dl.fbaipublicfiles.com/meru/clip_vit_s.pth -P checkpoints
```
It will take ~1, ~5 minutes to download the small, large models respectivally.

## The Basics

Now let's try several commands.

1. Image traversal
   
```sh
python scripts/image_traversals.py --image-path assets/taj_mahal.jpg \
    --checkpoint-path checkpoints/meru_vit_s.pth --train-config configs/train_meru_vit_s.py
```

2. Zero-shot image classification
   ```sh
   python scripts/evaluate.py --config configs/demo/zero_shot_classification.py \
    --checkpoint-path checkpoints/meru_vit_s.pth \
    --train-config configs/train_meru_vit_s.py
   ```

3. Linear probe classification
   ```sh
   python scripts/evaluate.py --config configs/demo/linprobe_classification.py \
    --checkpoint-path checkpoints/meru_vit_s.pth \
    --train-config configs/train_meru_vit_s.py 
    ```
    
4. Zero-shot image and text retrieval
   
   Let's first download the coco dataset (~5 min):
   ```sh
   chmod 777 ./meru/data/download_coco.sh
    ./meru/data/download_coco.sh
   ```
   and run the retrieval script
   ```sh
   python scripts/evaluate.py --config configs/demo/zero_shot_retrieval.py \
    --checkpoint-path checkpoints/meru_vit_s.pth \
    --train-config configs/train_meru_vit_s.py
   ```

## Training

How to get a symlink to work:
[stackoverflow](https://superuser.com/questions/511900/why-doesnt-my-symbolic-link-work).

To train, run the command:
```sh
python scripts/train.py \
   --config configs/train_meru_vit_s.py \
   --num-gpus 2 \
   train.total_batch_size=128
```

To finetune the last layer with a new dimension of 64, run the command:
```sh
python scripts/train.py \
   --config configs/train_meru_vit_s.py \
   --resume \
   --proj-layer-only 64 \
   --num-gpus 2 \
   --save \
   --output-dir ./output/
   train.num_iterations=125000
```