/* Header file for calculating orientation and colour maps based on 
 * Walter and Koch (2006): Modeling attention to salient proto-objects
 *
 * Developed for Foundations of Machine Learning 2016, NYU */
#ifndef MAPS_H
#define MAPS_H

#define M_PIl 3.141592653589793238462643383279502884L /* pi */

float getAttenuation(cv::Point target, cv::Point surround);
void getGaborFilters(std::vector<cv::Mat>& filters0, std::vector<cv::Mat>& filterspi2);
void getIntensityMaps(const cv::Mat artifact, cv::Mat& ints, cv::Mat& rg, cv::Mat& by, std::fstream& outputFile);
std::vector<cv::Mat> getOrientationMaps(cv::Mat artifact, std::vector<cv::Mat> filters0, std::vector<cv::Mat> filterspi2, std::fstream& outputFile);

template <typename T> 
cv::Mat getCenterDifference(cv::Mat target, uint center, uint surround, uint pixels);
cv::Mat getCenterAddition(cv::Mat target, uint center, uint surround, uint pixels, double ease);
cv::Mat normalizeDifferences(cv::Mat map);

#endif
