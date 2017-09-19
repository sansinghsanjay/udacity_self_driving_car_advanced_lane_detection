# Udacity Self Driving Car Nanodegree: Advanced Lane Detection
 [sample_input.gif] [technologies_used.png] [sample_output.png]

## Objective
This project is for marking lanes on road so that Self Driving Cars can run on correct path. 

In this project, some hard "Computer Vision" issues, such as camera calibration, are considered and an attempt is made to solve them.

Above GIF images are showing sample input (at left side) and sample output (at right side) of this project, obtained by using Python and OpenCV (Open Computer Vision).

## Introduction

### Self Driving Cars
Self Driving Cars are unmanned ground vehicles, also known as Autonomus Cars, Driverless Cars, Robotic Cars.
[Self Driving Car Image]

### Technologies Used
Following are the technologies used by these Self Driving Cars to navigate:

1. Computer Vision and Machine Learning (AI) to find path, to classify traffic signs, etc.
2. Sensor Fusion to sense the surrounding moving vehicles.
3. Machine Learning (AI) for decision making.

### Why "Lane Detection"?
A simple traffic rule says that every vehicle should run in its own (directional) lane. If it goes in other lane then chances of accident increases. Thus, lane marking is an important task which prevents these autonomus vehicles from entering in other lane and hence prevents from accidents.

Self Driving Cars uses Computer Vision to find the path on which these cars/vehicles have to navigate. Computer Vision uses symbolic information from image using techniques from geometry, physics, statistics and information theory.


## Programming Language
In this project, Python-3.5.2 is used with following packages:
1. numpy - 1.13.0
2. moviepy - 0.2.3.2
3. cv2 - 3.0.0 (Computer Vision)


## Algorithm
1. First, we perform camera calibration (estimating parameters of lens and image sensor to correct lens distortion):
	i. For camera calibration, it is proven to use classical black-white chessboard image.
	ii. We read various black-white chessboard images and calculate "image points" and "object points" by using OpenCV functions like: cv2.findChessboardCorners()
	iii. After getting "image points" and "object points" from above step, we perform camera calibration on a black-white chessboard image by using OpenCV function: cv2.calibrateCamera().
	iv. The estimated parameters in above step are used to undistort images by using an OpenCV function: cv2.undistort()
	v. Following is a sample of given image (left) and undistorted image (right) after camera calibration step: [image]

2. We read an image: [image]

3. Undistort the above image by using camera calibration parameters obtained in (1.iii): [image]

4. On the undistorted image, we perform perspective transform to generate bird's eye view: [image]

5. Then, we transform above obtained image into HLS and LAB color space: [image] [image]

6. HLS and LAB color space image are combined by putting 1 at all position where both HLS and LAB have pixel value 1: [image]

7. Then, we perform various mathematical operations to draw mark lanes on input image: [image]

8. At last, we write data about lanes (final output): [image]


## How To Use?
To use this project:
1. Clone this github repository.
2. Make "scripts" sub-directory as present-working-directory.
3. Run this command in terminal: ```python main.py```

Or simply use ipython notebook under "scripts" sub-directory.


## Limitations
1. On turns, this code is giving weired output.
2. If the position of lane gets changed (like width of lanes), then this code may not be able to detect the lanes even if lanes are straight. This is because region of interest is hard coded here, its not generic.
3. This technique of lane detection would not work if light from sun or headlight of another vehicle falls on camera of self driving car. 
4. In this technique, its an assumption that roads are flat and have clearly visible lane marks.
