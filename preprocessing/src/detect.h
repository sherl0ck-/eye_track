/* Header file for calculating orientation and colour maps based on 
 * Walter and Koch (2006): Modeling attention to salient proto-objects
 *
 * Developed for Foundations of Machine Learning 2016, NYU */
#ifndef DETECT_H
#define DETECT_H

// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Image size constants
const int kEyeWidth  = 70;
const int kEyeHeight = 60;

// // Preprocessing
// const bool kSmoothFaceImage = false;
// const float kSmoothFaceFactor = 0.005;

// // Algorithm Parameters
// const int kFastEyeWidth = 50;
// const int kWeightBlurSize = 5;
// const bool kEnableWeight = true;
// const float kWeightDivisor = 1.0;
// const double kGradientThreshold = 50.0;

// // Postprocessing
// const bool kEnablePostProcess = true;
// const float kPostProcessThreshold = 0.97;

// // Eye Corner
// const bool kEnableEyeCorner = false;

int detectSetup();
int detectFace(cv::Mat greyFrame, cv::Rect &faceFrame);
void detectEyeRegions(cv::Mat frame_gray, cv::Rect face, cv::Rect &leftEye, cv::Rect &rightEye);

#endif
