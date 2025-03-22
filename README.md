# Tennis Analysis

## Introduction
This project tracks tennis players in a video clip to measure their speed, ball shot speed and number of shots. The tennis ball and players are detected using YOLO, while also uisng   Neural Network (CNNs) for extracting the court keypoints.

## Output Videos
Here is a screenshot from  the output video:

<img width="1440" alt="Screen Shot 2025-01-05 at 10 52 49 AM" src="https://github.com/user-attachments/assets/c64d1b42-939b-4d0a-9ec4-0701418fc229" />


## Models Used
* YOLO for player detection
* Fine Tuned YOLO for tennis ball detection
* Tennis Court Key point extraction



## Training
* Tennis ball detetcor with YOLO: training/tennis_ball_detector_training.ipynb
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## TechStack
* python3.8
* ultralytics
* pytorch
* pandas
* numpy 
* opencv
