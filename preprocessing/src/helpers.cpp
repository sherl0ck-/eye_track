#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>
#include <unordered_map>
#include <utility>

int accX = 0;
int accY = 0;
int expand = 0;
int screenH = 800;
std::unordered_map<std::string, bool> windows;
void debugWindow(std::string name, cv::Mat image) {
    if (windows.find(name) != windows.end()) {
        imshow(name, image);

        return;
    } else 
        windows.insert(std::make_pair<std::string, bool>(name, true));

    cv::namedWindow(name,CV_WINDOW_NORMAL);
    if (accY + image.rows > screenH) {
        accY = 0;
        accX += expand;
        expand = 0;
    }
    cv::moveWindow(name, accX, accY);
    accY += image.rows + 40;
    std::cout << "setup " << name << " " << accX << " " << accY << std::endl;
    if (expand < image.cols)
        expand = image.cols;
    imshow(name, image);
}

void downSample(cv::Mat& target, cv::Size blur, const uint decimation, uint times) {
    while(times-- > 0) {
        cv::GaussianBlur(target, target, blur, 0);
        cv::pyrDown(target, target, cv::Size(target.cols/decimation, target.rows/decimation));    
    }
}

bool rectInImage(cv::Rect rect, cv::Mat image) {
    return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
           rect.y+rect.height < image.rows;
}

bool pointInImage(cv::Point point, cv::Mat image) {
    return point.x > 0 && point.y > 0 && point.x < image.cols && point.y < image.rows;
}

bool inMat(cv::Point p,int rows,int cols) {
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

cv::Mat matrixMagnitude(const cv::Mat& matX, const cv::Mat& matY) {
    cv::Mat mags(matX.rows, matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
        const double* Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double* Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}

double computeDynamicThreshold(const cv::Mat& mat, double stdDevFactor) {
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows * mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}