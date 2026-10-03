# Face Recognition with ArcFace (MATLAB)

Capstone project implementing a deep learning face recognition pipeline in MATLAB, built around a ResNet-50 backbone and an ArcFace (Additive Angular Margin Loss) head for learning discriminative face embeddings.

## Overview

Standard softmax classification doesn't push embeddings of the same person close together or different people far apart in feature space. ArcFace fixes this by adding an angular margin penalty directly on the classification boundary, which produces embeddings that are much better suited for face verification (comparing two faces and deciding if they're the same person) rather than just classification.

The pipeline here:

1. Load and organize a face dataset into per-person folders
2. Detect and crop faces, balance the dataset across classes, and apply augmentation
3. Fine-tune a pretrained ResNet-50 with a plain softmax head (warmup stage)
4. Replace the head with an embedding layer + ArcFace transform and continue training
5. Evaluate with a confusion matrix and a cosine-similarity based verification function

## Files

| File | Description |
|---|---|
| `face_recognition.m` | Main script: dataset loading, preprocessing, warmup training, ArcFace training, evaluation, and a `verifyFace` helper for comparing two face images |
| `ArcFaceTransformLayer.m` | Custom layer applying the ArcFace angular margin transform to normalized embeddings before the softmax |
| `ArcFaceLossLayer.m` | Alternative formulation of the ArcFace margin loss as a standalone layer |
| `CenterLossLayer.m` | Optional center loss layer that pulls embeddings of the same class toward a learned class center, used alongside softmax/ArcFace to tighten clusters |
| `l2NormalizationLayer.m` | Simple L2-normalization layer used before computing cosine similarity between embeddings |
| `organize_ds.m` | Utility script that sorts a flat folder of images into per-person subfolders based on filename |
| `faceRecognitionNet_final.mat` | Trained network weights (tracked with Git LFS) |

## Requirements

- MATLAB with Deep Learning Toolbox
- Computer Vision Toolbox (used for `vision.CascadeObjectDetector` face detection)
- A GPU is recommended for training (`ExecutionEnvironment = 'gpu'` is set in the training options)

## Usage

1. Point `organize_ds.m` at your raw image folder to group images by person
2. Update `datasetPath` in `face_recognition.m` to the organized dataset
3. Run `face_recognition.m` section by section — it runs the warmup stage first, then trains the ArcFace head on top of it
4. Use `verifyFace(net, img1, img2, threshold)` to compare two face images by embedding cosine similarity

## Note on the model file

`faceRecognitionNet_final.mat` is tracked with [Git LFS](https://git-lfs.com/). Make sure you have Git LFS installed before cloning, otherwise you'll just get a pointer file instead of the actual weights.
