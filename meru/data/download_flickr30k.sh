#! /bin/bash

mkdir -p datasets

############################################
#### Download the test/train captions. #####
############################################

# Download the captions.
wget https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip -P datasets

# Unzip the captions.
unzip datasets/flickr30k.zip -d datasets

# Reorganize the directories.
mv datasets/flickr30k/dataset.json datasets/flickr30k/dataset_flickr30k.json

# Clean up the directory.
rm datasets/flickr30k.zip datasets/flickr30k/vgg_feats.mat datasets/flickr30k/readme.txt

#############################################
#### Download the images.                ####
#############################################

# Download the dataset.
wget https://uofi.app.box.com/shared/static/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl

# Unzip the file.
mv 1cpolrtkckn4hxr1zhmfg0ln9veo6jpl flickr30k.tar.gz
gunzip flickr30k.tar.gz
tar xvf flickr30k.tar -C datasets/flickr30k

# Reorganize the directories.
mv datasets/flickr30k/flickr30k-images datasets/flickr30k/flickr30k_images
mv datasets/flickr30k/flickr30k_images/readme.txt datasets/flickr30k/readme.txt

# Clean up the directory.
rm -rf flickr30k.tar