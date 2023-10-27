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
pip3 install torch torchvision tensorflow          
python -m pip install --pre timm
python -m pip install -r requirements.txt
python setup.py develop
```

Download the small meru model
```sh
wget https://dl.fbaipublicfiles.com/meru/meru_vit_s.pth -P checkpoints
```

Now let's try several commands.

1. Image traversal
   
```sh
python scripts/image_traversals.py --image-path assets/taj_mahal.jpg \
    --checkpoint-path checkpoints/meru_vit_s.pth --train-config configs/train_meru_vit_s.py
```

2. Zero-shot image classification
   ```sh
   python scripts/evaluate.py --config configs/small/eval_zero_shot_classification_small.py \
    --checkpoint-path checkpoints/meru_vit_s.pth \
    --train-config configs/train_meru_vit_s.py
   ```

3. Linear probe classification
   ```sh
   python scripts/evaluate.py --config configs/small/eval_linprobe_classification_small.py \
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
   python scripts/evaluate.py --config configs/small/eval_zero_shot_retrieval_small.py \
    --checkpoint-path checkpoints/meru_vit_s.pth \
    --train-config configs/train_meru_vit_s.py
   ```