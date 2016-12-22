#ifndef HELPERS_H
#define HELPERS_H

void debugWindow(std::string name, cv::Mat image);
void downSample(cv::Mat& target, cv::Size blur, const uint decimation, uint times);
bool rectInImage(cv::Rect rect, cv::Mat image);
bool pointInImage(cv::Point point, cv::Mat image);
bool inMat(cv::Point p,int rows,int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

#endif