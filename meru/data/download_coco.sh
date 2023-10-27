#! /bin/bash

mkdir -p datasets/coco

wget http://images.cocodataset.org/zips/train2017.zip -P datasets/coco
unzip datasets/coco/train2017.zip -d datasets/coco
rm datasets/coco/train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip -P datasets/coco
unzip datasets/coco/val2017.zip -d datasets/coco
rm datasets/coco/val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P datasets/coco
unzip datasets/coco/annotations_trainval2017.zip -d datasets/coco
rm datasets/coco/annotations_trainval2017.zip datasets/coco/annotations/instances_train2017.json datasets/coco/annotations/instances_val2017.json datasets/coco/annotations/person_keypoints_train2017.json datasets/coco/annotations/person_keypoints_val2017.json
