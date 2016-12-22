/* File for calculating haar features based on Viola and Jones (2001):
 * Rapid Object Detection using a Boosted Cascade of Simple Features
 *
 * Developed for Foundations of Machine Learning 2016, NYU */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "haar.h"
#include <vector>

/* Get rectangle corner points. */
std::vector<cv::Point> getPoints(cv::Rect r) {
    // Get all rectangle corners
    cv::Point tl = r.tl();
    cv::Point tr = cv::Point(tl.x + r.width, tl.y);
    cv::Point br = cv::Point(tl.x + r.width, tl.y + r.height);
    cv::Point bl = cv::Point(tl.x, tl.y + r.height);

    // Add corners to vector
    std::vector<cv::Point> points;
    points.push_back(tl);  
    points.push_back(tr);
    points.push_back(br);
    points.push_back(bl);
    return points;
}

/* Get's horizontal haar regions by subdividing the given
 * area recurse many times. */
std::vector<Haar> getHaarHorizontal(const cv::Rect area, int recurse) {
    if (recurse == 0) return std::vector<Haar>();
    // Calculate the initial haar rectangles
    cv::Rect hp = area;
    hp.width = area.width/2;
    cv::Rect hn = hp;
    hn.x += hn.width;

    // Get haar rectangles recursively
    recurse--;
    std::vector<Haar> left = getHaarHorizontal(hp, recurse);
    std::vector<Haar> right = getHaarHorizontal(hn, recurse);
    std::vector<Haar> result;
    result.reserve(left.size() + right.size());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    
    // Push to result array
    struct Haar h = { hp, hn };
    result.push_back(h);
    return result;
}

/* Get's vertical haar regions -- should be merged with the 
 * horizontal function. */
std::vector<Haar> getHaarVertical(const cv::Rect area, const int recurse) {
    if (recurse == 0) return std::vector<Haar>();
    // Calculate the initial haar rectangles
    cv::Rect hp = area;
    hp.height = area.height/2;
    cv::Rect hn = hp;
    hn.y += hn.height;

    // Get haar rectangles recursively
    std::vector<Haar> left = getHaarVertical(hp, recurse - 1);
    std::vector<Haar> right = getHaarVertical(hn, recurse - 1);
    std::vector<Haar> result;
    result.reserve(left.size() + right.size());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    
    struct Haar h = { hp, hn };
    result.push_back(h);

    return result;
}

/* Function that gets the haar rectangles from
 * a specific eye image. */
float wScale = 4;
float hScale = 7;
std::vector<Haar> getHaarRects(const cv::Mat eye) {
    cv::Point tll(eye.cols/2 - eye.cols/wScale, eye.rows/2 - eye.rows/hScale);
    cv::Point brl(eye.cols/2 + eye.cols/wScale, eye.rows/2 + eye.rows/hScale);
    cv::Rect hhAreaLeft(tll, brl);
    std::vector<Haar> hor = getHaarHorizontal(hhAreaLeft, 2);
    std::vector<Haar> ver = getHaarVertical(hhAreaLeft, 1);
    std::vector<Haar> result;
    result.reserve(hor.size() + ver.size());
    result.insert(result.end(), hor.begin(), hor.end());
    result.insert(result.end(), ver.begin(), ver.end());
    return result;
}

/* Calculates the haar feature given a haar
 * rectangle struct and an integral map. */
int calculateHaar(const cv::Mat integral, const struct Haar haar) {
    cv::Rect positive = haar.hp;
    cv::Rect negative = haar.hn;
    // Get points to calculate pixel sum difference
    std::vector<cv::Point> pos = getPoints(positive);
    std::vector<cv::Point> neg = getPoints(negative);

    return  integral.at<int>(pos.at(0)) - integral.at<int>(pos.at(1)) + integral.at<int>(pos.at(2)) - integral.at<int>(pos.at(3))
            - (integral.at<int>(neg.at(0)) - integral.at<int>(neg.at(1)) + integral.at<int>(neg.at(2)) - integral.at<int>(neg.at(3)));
}