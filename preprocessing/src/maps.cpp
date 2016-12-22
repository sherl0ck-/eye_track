/* File for calculating orientation and colour maps based on 
 * Walter and Koch (2006): Modeling attention to salient proto-objects
 * and Xie et al. (2014) Small target detection based on accumulated 
 * center-surround difference measure.
 *
 * Developed for Foundations of Machine Learning 2016, NYU */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>

#include "helpers.h"
#include "maps.h"

class GreyScaleException: public std::exception {
    virtual const char* what() const throw() {
        return "The matrix must be gray scale.";
    }
} greyScaleExp;

class CenterSurroundException: public std::exception {
    virtual const char* what() const throw() {
        return "The center must be smaller then the surround rectangle.";
    }
} centerSurroundExp;

/* Compute the attenuation function for the 
   center surround differences. */
float getAttenuation(cv::Point target, cv::Point surround) {
    const float cVal = 0.1;
    std::vector<cv::Point> points;
    points.push_back(target - surround);
    // Return the attenuation factor 
    // 1 - e^{-c||target-surround||_2^2}.
    return  1.0 - std::exp(-cVal * cv::norm(points, cv::NORM_L2SQR));
}

/* Calculate center surround difference based on
   Xie et al. (2014) Small target detection based 
   on accumulated center-surround difference measure. */
template <typename T>
cv::Mat getCenterDifference(cv::Mat target, uint center, uint surround, uint pixels) {
    if (target.channels() != 1)
        throw greyScaleExp;
    if (surround <= center)
        throw centerSurroundExp;

    // Get surround difference, note
    // that surround > center
    int offset = surround - center;
    int shift = offset / pixels;

    cv::Mat differenceMap = target.clone();
    // Iterate over pixels
    for (int y = 0; y < target.rows; ++y) {
        for (int x = 0; x < target.cols; ++x) {
            cv::Point tp(x, y);
            T minDiff = 255;
            // Iterate over difference orientations
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if ((i - 1) == 0 && (j - 1) == 0) continue;
                    // Calculate center difference
                    T diff = 0;
                    bool discard = false;
                    for (int k = 0; k < offset; k += shift) {
                        cv::Point sp((i - 1) * center + (i - 1) * k, (j - 1) * center + (j - 1) * k);
                        // Discard direction if not all points
                        // are in the image
                        if (!pointInImage(tp + sp, target)) {
                            discard = true;
                            continue;
                        }
                        diff += getAttenuation(tp, sp) * std::abs(target.at<T>(tp) - target.at<T>(sp));
                    }

                    if (diff < minDiff && !discard)
                        minDiff = diff;
                }
            }
            // std::cout << "Point (" << (tp).x << "," << (tp).y << ") --------- diff " << (uint)minDiff << std::endl;
            differenceMap.at<T>(tp) = (T)minDiff;
        }
    }
    cv::normalize(differenceMap, differenceMap, 0, 1, cv::NORM_MINMAX);
    return differenceMap;
    //return normalizeDifferences(differenceMap);
}
template cv::Mat getCenterDifference<uchar>(cv::Mat target, uint center, uint surround, uint pixels);
template cv::Mat getCenterDifference<float>(cv::Mat target, uint center, uint surround, uint pixels);

/* Calculate center surround addition to
 * do across-scale addition. */
cv::Mat getCenterAddition(cv::Mat target, uint center, uint surround, uint pixels, double ease) {
    if (target.channels() != 1)
        throw greyScaleExp;
    if (surround <= center)
        throw centerSurroundExp;

    // Get surround difference, note
    // that surround > center
    int offset = surround - center;
    int shift = offset / pixels;

    cv::Mat additionMap = target.clone();
    // Iterate over pixels
    for (int y = 0; y < target.rows; ++y) {
        for (int x = 0; x < target.cols; ++x) {
            cv::Point tp(x, y);
            // Iterate over difference orientations
            float totalDiff = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if ((i - 1) == 0 && (j - 1) == 0) continue;
                    // Calculate center addition
                    bool discard = false;
                    float diff = 0.0;
                    for (int k = 0; k < offset; k += shift) {
                        cv::Point sp((i - 1) * center + (i - 1) * k, (j - 1) * center + (j - 1) * k);
                        // Discard direction if not all points
                        // are in the image
                        if (!pointInImage(tp + sp, target)) {
                            discard = true;
                            continue;
                        }
                        diff += ease * std::abs(target.at<float>(sp));
                    }

                    if (!discard)
                        totalDiff += diff;
                }
            }
            std::cout << totalDiff << std::endl;
            additionMap.at<float>(tp) = totalDiff;
        }
    }
    return additionMap;
}

cv::Mat normalizeDifferences(cv::Mat map) {
    // Set intensity range to 0 and 1
    cv::Mat mapf(map.size(), CV_32FC1);
    map.convertTo(mapf, CV_32FC1);
    cv::normalize(mapf, mapf, 0.0, 1.0, cv::NORM_MINMAX);

    // Create normalisation kernel
    double sigmaEx2 = 0.0004 * map.cols * map.cols,
           sigmaInf2 = 0.0625 * map.cols * map.cols;
    cv::Mat kernel(map.size(), CV_32FC1);
    for (int i = 0; i < kernel.rows; ++i) {
        for (int j = 0; j < kernel.cols; ++j) {
            // Following Itti and Koch (2001)
            float ft = 0.25 / (2.0 * M_PIl * sigmaEx2) * std::exp(-(i*i + j*j) / (2.0 * M_PIl * sigmaEx2)); // First term
            float st = 2.25 / (2.0 * M_PIl * sigmaInf2) * std::exp(-(i*i + j*j) / (2.0 * M_PIl * sigmaInf2)); // Second term
            kernel.at<double>(i, j) = ft - st;
        }
    }

    // Convolve followed by the (4) transformation
    // in Itti and Kock (2001)
    int times = 0;
    cv::Mat convolvedMap = mapf.clone();
    cv::filter2D(convolvedMap, convolvedMap, -1, kernel);
    for (int k = 0; k < times; ++k) 
        for (int i = 0; i < mapf.rows; ++i)
            for (int j = 0; j < mapf.cols; ++j)
                mapf.at<float>(i, j) = std::max(0.0, mapf.at<float>(i, j) + convolvedMap.at<double>(i, j) - 0.02);
    
    return mapf;
}

/* Return the gabor filters. */
void getGaborFilters(std::vector<cv::Mat>& filters0, std::vector<cv::Mat>& filterspi2) {
    for (double i = 0; i < 3*M_PIl/4; i+=M_PIl/4) {
        cv::Mat psi0 = cv::getGaborKernel(cv::Size(19,19), 3, i, 10, 1, 0);
        cv::Mat psipi2 = cv::getGaborKernel(cv::Size(19,19), 3, i, 10, 1, M_PIl/2);
        filters0.push_back(psi0);
        filterspi2.push_back(psipi2);
    }
}

/* Compute intensity maps. */
void getIntensityMaps(const cv::Mat artifact, cv::Mat& ints, cv::Mat& rg, cv::Mat& by, std::fstream& outputFile) {
    for (int i = 0; i < artifact.rows; i++) {
        for (int j = 0; j < artifact.cols; j++) {
            // Retrieve color intensities
            cv::Vec3b intensity = artifact.at<cv::Vec3b>(i, j);
            uchar b = intensity.val[0];
            uchar g = intensity.val[1];
            uchar r = intensity.val[2];
            ints.at<float>(i, j) = (b + g + r) / 3.0;
            
            // Calculate RG and BY maps
            float rgValue = r - g;
            float byValue = b - std::min<uchar>(r,g);
            rg.at<float>(i,j) = rgValue;
            by.at<float>(i,j) = byValue;
            outputFile << (float)(b+g+r)/3.0 << " " << (float)rgValue << " " << (float)byValue << " "; 
        }
    }

    cv::blur(ints, ints, cv::Size(2, 2));
    cv::blur(rg, rg, cv::Size(2, 2));
    cv::blur(by, by, cv::Size(2, 2));
}

/* Calculate orientation maps by applying  
   gabor filters. */ 
std::vector<cv::Mat> getOrientationMaps(cv::Mat artifact, std::vector<cv::Mat> filters0, std::vector<cv::Mat> filterspi2, std::fstream& outputFile) {
    std::vector<cv::Mat> orientations;
    for (double i = 0; i < 4; i++) {
        cv::Mat orientation;
        cv::Mat filter0;
        cv::Mat filterpi2;
        cv::filter2D(artifact, filter0, -1, filters0[i]);
        cv::filter2D(artifact, filterpi2, -1, filterspi2[i]);
        cv::convertScaleAbs(filter0, filter0);
        cv::convertScaleAbs(filterpi2, filterpi2);
        orientation = filter0.clone() + filterpi2;
        // Print features
        for (int i = 0; i < orientation.rows; i++) {
            for (int j = 0; j < orientation.cols; j++) {
                outputFile << (float)orientation.at<uchar>(i,j) << " ";                            
            }
        }
        orientations.push_back(orientation);
    }
    return orientations;
}
    