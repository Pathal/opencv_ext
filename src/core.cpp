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
	return -1;
}

std::array<double, 4> cvx::matlab::quatmultiply(std::array<double, 4> q, std::array<double, 4> r) {
	return {
		(r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]),
		(r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]),
		(r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]),
		(r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0])
	};
}

cv::Mat cvx::matlab::strel(STREL_SHAPE shape, cv::Size& dims, double radius) {
	cv::Mat element;
	cv::Size local_dims = dims;

	switch (shape) {
	case STREL_SHAPE::RECTANGLE:
	case STREL_SHAPE::SQUARE:
		// If dims are empty, just make the radius into a square
		if (local_dims.area() == 0) local_dims = cv::Size(radius, radius);
		element = cv::Mat::ones(dims, CV_8UC1);
		break;
	case STREL_SHAPE::DISK:
		element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
		break;
	case STREL_SHAPE::OCTAGON:
	case STREL_SHAPE::DIAMOND:
	case STREL_SHAPE::LINE:
	default:
		CV_Error(cv::Error::StsBadArg, "Currently Unsupported STREL shape.");
		break;
	}

	return element;
}
