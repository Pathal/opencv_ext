#include "core.hpp"

float cvx::getPixelAsFloat(const cv::Mat& inp, int x, int y) {
	switch (inp.depth()) {
	//case CV_32F:
	case CV_32FC1:
		return inp.at<float>(x, y);
	case CV_64FC1:
		return (float)inp.at<double>(x, y);
	//case CV_8U:
	case CV_8UC1:
		return (float)inp.at<uchar>(x, y);
	case CV_16UC1:
		return (float)inp.at<ushort>(x, y);
	default:
		break;
	}
}

void cvx::matlab::quatmultiply() {
	//
}
