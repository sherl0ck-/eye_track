/* Header file for calculating haar features based on Viola and Jones (2001):
 * Rapid Object Detection using a Boosted Cascade of Simple Features
 *
 * Developed for Foundations of Machine Learning 2016, NYU */
#ifndef HAAR_H
#define HAAR_H

struct Haar {
    // Positive and negative haar areas
    cv::Rect hp; 
    cv::Rect hn;
};

std::vector<cv::Point> getPoints(cv::Rect r);
std::vector<Haar> getHaarHorizontal(const cv::Rect area, int recurse);
std::vector<Haar> getHaarVertical(const cv::Rect area, int recurse);
std::vector<Haar> getHaarRects(const cv::Mat eye);
int calculateHaar(const cv::Mat integral, const struct Haar haar);

#endif
