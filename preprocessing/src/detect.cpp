#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <vector>

#include "detect.h"
#include "findEyeCenter.h"

class DetectException: public std::exception {
	virtual const char* what() const throw() {
    	return "Please run detectSetup() in detect.cpp before starting any detections.";
  	}
} detectExp;

int setupRun = 0;
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
cv::String faceCascadeName = "../res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier faceCascade;


int detectSetup() {
	// Load the cascades
    if (!faceCascade.load(faceCascadeName)) {
    	throw detectExp;
    }
    return setupRun = 1;
}

void detectCheck() {
	if (!setupRun) 
		throw detectExp;
}

/* Detects the face using the face cascade algorithm
 * from open cv. */
int detectFace(cv::Mat greyFrame, cv::Rect &faceFrame) {
	detectCheck();

    // Detect faces
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(greyFrame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150));
    if (faces.size() > 0)
        faceFrame = faces[0];
    return faces.size();
}

/* Find the eye regions of a given face (statically). */
void detectEyeRegions(cv::Mat greyFrame, cv::Rect face, cv::Rect &leftEye, cv::Rect &rightEye) {
	detectCheck();

    cv::Mat faceROI = greyFrame(face);
    cv::Mat debugFace = faceROI;

    // Find eye regions 
    int ers = face.width * (kEyePercentSide / 100.0); // Side
    int erw = face.width * (kEyePercentWidth / 100.0); // Width
    int erh = face.width * (kEyePercentHeight / 100.0); // Height
    int ert = face.height * (kEyePercentTop / 100.0); // Top
    cv::Rect leftEyeRegion(ers, ert, erw, erh);
    cv::Rect rightEyeRegion(face.width - erw - ers, ert, erw, erh);

    // Set eye regions
    leftEye = leftEyeRegion;
    rightEye = rightEyeRegion; 
}