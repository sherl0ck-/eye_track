#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <vector>

#include "saliency.h"
#include "findEyeCenter.h"
#include "helpers.h"
#include "detect.h"
#include "haar.h"
#include "maps.h"

/* Constants. */
std::string basePath = "/Users/Brinck/Documents/data";
std::string outputPath = basePath + "/preprocessed";
std::string dataPath = basePath + "/points";
std::fstream outputFile;

/* Function headers. */
void setup();
void endSetup();

/* Global variables. */
cv::Mat debugImage;

cv::Rect alignRect(cv::Rect frame, int width, int height) {
    int wdiff = frame.width - width;
    int hdiff = frame.height - height;
    cv::Rect r = frame;
    r.width -= wdiff;
    r.height -= hdiff;
    r.x += wdiff/2;
    r.y += hdiff/2;
    return r;
}

/**
 * @function main
 */
int main( int argc, const char** argv ) {
    cv::Mat frame;


    // Load the cascades
    detectSetup();
    
    // setup(); 
    std::cout << outputPath + "/data" << std::endl;
    outputFile.open(outputPath + "/data", std::fstream::out | std::fstream::trunc);
    int c = 0;
    std::ifstream dataFile(dataPath);
    std::string line;

    // Move to arbitrary line in data
    // std::srand(std::time(NULL));
    // int randomLine = std::rand() % 50000;
    // for (int i = 0; i < randomLine; ++i) 
    //     std::getline(dataFile, line);
    // std::cout << "Moved to line " << randomLine << std::endl;

    // Gabor filters
    // std::vector<cv::Mat> filters0;
    // std::vector<cv::Mat> filterspi2;
    // getGaborFilters(filters0, filterspi2);

    int counter = 0;
    for (; std::getline(dataFile, line); ){       
        float glanceX, glanceY;
        std::string imageName;
        
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        imageName = token.substr(0, token.size() - 1);
        iss >> token;
        glanceX = std::stof(token.substr(0, token.size() - 1));
        iss >> token;
        glanceY = std::stof(token);

        cv::Mat frame = cv::imread(basePath + "/" + imageName);
        cv::flip(frame, frame, 1);
        frame.copyTo(debugImage);

        std::vector<cv::Mat> rgbChannels(3);
        cv::split(frame, rgbChannels);
        cv::Mat greyFrame = rgbChannels[2];

        if(!frame.empty()) {
            cv::Rect faceFrame;
            int faces = detectFace(greyFrame, faceFrame);
            cv::Mat face = frame(faceFrame);
            cv::Mat greyFace = greyFrame(faceFrame);

            if (faces > 0) {
                cv::Rect leftEyeRect,
                         rightEyeRect;
                detectEyeRegions(greyFrame, faceFrame, leftEyeRect, rightEyeRect);
                cv::Mat leftEye = face(leftEyeRect),
                        rightEye = face(rightEyeRect);

                cv::Mat eyePrint = leftEye.clone();

                cv::Point leftPupil = findEyeCenter(greyFace, leftEyeRect, "Left Eye"),
                          rightPupil = findEyeCenter(greyFace, rightEyeRect, "Right Eye");
                cv::Point tlLeftEye(leftPupil.x-kEyeWidth/2,leftPupil.y-kEyeHeight/2),
                          brLeftEye(leftPupil.x+kEyeWidth/2,leftPupil.y+kEyeHeight/2),
                          tlRightEye(rightPupil.x-kEyeWidth/2,rightPupil.y-kEyeHeight/2),
                          brRightEye(rightPupil.x+kEyeWidth/2,rightPupil.y+kEyeHeight/2);

                cv::Rect adjLeftEyeRect(tlLeftEye, brLeftEye),
                         adjRightEyeRect(tlRightEye, brRightEye);
                adjLeftEyeRect.x+=leftEyeRect.x;
                adjLeftEyeRect.y+=leftEyeRect.y;
                adjRightEyeRect.x+=rightEyeRect.x;
                adjRightEyeRect.y+=rightEyeRect.y;
                leftEyeRect = adjLeftEyeRect;
                rightEyeRect = adjRightEyeRect;
                if (!rectInImage(adjLeftEyeRect, face) || !rectInImage(adjRightEyeRect, face))
                    continue;

                leftEye = face(leftEyeRect);
                rightEye = face(rightEyeRect);

                size_t lastIndex = imageName.find_last_of("."); 
                string rawName = imageName.substr(0, lastIndex);

                // Pupil features (4 -> 0-4)
                outputFile << leftPupil.x << " " << leftPupil.y << " ";
                outputFile << rightPupil.x << " " << rightPupil.y << " ";

                // Haar features (8 -> 4-12)
                float decimation = 2;
                cv::Mat haarLeftEye = greyFace(leftEyeRect);
                cv::Mat integralLeft;
                downSample(haarLeftEye, cv::Size(3,3), decimation, 1);
                cv::integral(haarLeftEye, integralLeft);
                std::vector<Haar> haarRectsLeftEye = getHaarRects(haarLeftEye);

                for (int i = 0; i < haarRectsLeftEye.size(); ++i) {
                    cv::rectangle(haarLeftEye, haarRectsLeftEye[i].hp, 200);
                    cv::rectangle(haarLeftEye, haarRectsLeftEye[i].hn, 200);
                    outputFile << calculateHaar(integralLeft, haarRectsLeftEye[i]) << " ";
                }

                cv::Mat haarRightEye = greyFace(rightEyeRect);
                cv::Mat integralRight;
                downSample(haarRightEye, cv::Size(3,3), decimation, 1);
                cv::integral(haarRightEye, integralRight);
                std::vector<Haar> haarRectsRightEye = getHaarRects(haarRightEye);
                
                for (int i = 0; i < haarRectsRightEye.size(); ++i) {
                    cv::rectangle(haarRightEye, haarRectsRightEye[i].hp, 200);
                    cv::rectangle(haarRightEye, haarRectsRightEye[i].hn, 200);
                    outputFile << calculateHaar(integralRight, haarRectsRightEye[i]) << " ";
                }

                debugWindow("Face", eyePrint);
                cv::waitKey(0);


                // std::cout << "Reading " + basePath + "/saliency/" + rawName + "saliency.png" << std::endl;
                cv::Mat saliencyLeft = cv::imread(basePath + "/saliency/" + rawName + "leftsaliency.png", 1);
                cv::Mat saliencyRight = cv::imread(basePath + "/saliency/" + rawName + "saliency.png", 1);
                cvtColor(saliencyLeft, saliencyLeft, CV_BGR2GRAY);
                cvtColor(saliencyRight, saliencyRight, CV_BGR2GRAY);

                for (int i = 0; i < saliencyLeft.cols; ++i)
                    for (int j = 0; j < saliencyRight.rows; ++j)
                        outputFile << (float)saliencyLeft.at<uchar>(i,j) << " " << (float)saliencyRight.at<uchar>(i,j) << " ";

                // //-- Intensity Map (1050 -> 12-1062)
                // // Downsample first.
                // cv::Mat intLeftEye = leftEye.clone();
                // cv::Mat intRightEye = rightEye.clone();                
                // downSample(intLeftEye, cv::Size(5,5), 2, 1);
                // downSample(intRightEye, cv::Size(5,5), 2, 1);

                // cv::Mat leftIntensity(intLeftEye.size(), CV_32FC1);
                // cv::Mat leftRG(intLeftEye.size(), CV_32FC1);
                // cv::Mat leftBY(intLeftEye.size(), CV_32FC1);
                // getIntensityMaps(intLeftEye, leftIntensity, leftRG, leftBY, outputFile);
                // cv::Mat leftOrientation = leftIntensity.clone();
                // cv::normalize(leftIntensity, leftIntensity, 0, 1, cv::NORM_MINMAX);
                // cv::normalize(leftRG, leftRG, 0, 1, cv::NORM_MINMAX);
                // cv::normalize(leftBY, leftBY, 0, 1, cv::NORM_MINMAX);

                // cv::Mat rightIntensity(intLeftEye.size(), CV_32FC1);
                // cv::Mat rightRG(intLeftEye.size(), CV_32FC1);
                // cv::Mat rightBY(intLeftEye.size(), CV_32FC1);
                // getIntensityMaps(intRightEye, rightIntensity, rightRG, rightBY, outputFile);
                // cv::Mat rightOrientation = rightIntensity.clone();
                // cv::normalize(rightIntensity, rightIntensity, 0, 1, cv::NORM_MINMAX);
                // cv::normalize(rightRG, rightRG, 0, 1, cv::NORM_MINMAX);
                // cv::normalize(rightBY, rightBY, 0, 1, cv::NORM_MINMAX);
                // //-- Orientation maps
                // // Left eye
                // std::vector<cv::Mat> leftOrientations  = getOrientationMaps(leftOrientation, filters0, filterspi2, outputFile);
                // // Right eye
                // std::vector<cv::Mat> rightOrientations  = getOrientationMaps(rightOrientation, filters0, filterspi2, outputFile);

                // cv::Mat leftIntensityDiff = getCenterDifference<float>(leftIntensity, 3, 7, 4);
                // cv::Mat leftRGDiff = getCenterDifference<float>(leftRG, 3, 7, 4);
                // cv::Mat leftBYDiff = getCenterDifference<float>(leftBY, 3, 7, 4);
                // cv::Mat leftFeature = leftIntensityDiff + leftRGDiff + leftBYDiff;
                // for (int i = 0; i < leftOrientations.size(); i++) {
                //     debugWindow("Orientation " + std::to_string(i), leftOrientations[i]);
                //     leftOrientations[i].convertTo(leftOrientations[i], CV_32FC1);
                //     leftFeature += 0.005 * leftOrientations[i];
                // }
                // cv::normalize(leftFeature, leftFeature, 0, 1, cv::NORM_MINMAX); 

                outputFile << glanceX << " " << glanceY << std::endl;


                // cv::circle(leftEye, cv::Point(kEyeWidth/2,kEyeHeight/2), 3, 1234);
                // debugWindow("Left Eye", leftEye);
                // cv::circle(rightEye, cv::Point(kEyeWidth/2,kEyeHeight/2), 3, 1234);
                // debugWindow("Right Eye", rightEye);
                // debugWindow("Haar Left Eye", haarLeftEye);
                // debugWindow("Haar Right Eye", haarRightEye);
                // debugWindow("Intensity Window", leftIntensity);
                // debugWindow("Diff Intensity Window", leftIntensityDiff);
                // debugWindow("LeftRG Window", leftRG);
                // debugWindow("LeftRG Window Diff", leftRGDiff);
                // debugWindow("LeftBY Window", leftBY);
                // debugWindow("LeftBY Window Diff", leftBYDiff);
                // debugWindow("Left Feature", leftFeature);
                // debugWindow("Sal Left", salLeft);
                // debugWindow("Sal Right", salRight);
            } else {
                std::cout << "No face found for " << imageName << std::endl;
            }
        } else {
            std::cout << "No frame found for " << imageName << std::endl;
        }

        // sleep(1);
        if (c++ % 50 == 0) std::cout << ".";
        if (c % 1000 == 0) std::cout << "#\n";
        std::cout.flush();
        // imshow(main_window_name,debugImage);
    }

    outputFile.close();
    return 0;
}

void setup() {
  // boost::filesystem::path dir(outputPath);
  // if(!boost::filesystem::create_directory(dir))
  //     std::cout << "Failed to create dir " << outputPath << std::endl; 
}