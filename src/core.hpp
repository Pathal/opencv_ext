#pragma once

#include <opencv2/core.hpp>

namespace cvx {
	float getPixelAsFloat(const cv::Mat& inp, int x, int y);
};
