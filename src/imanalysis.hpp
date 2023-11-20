#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace cvx {
	std::vector<cv::KeyPoint> detectSURFFeatures(const cv::Mat& inp,
		float MetricThreshold = 1000,
		int NumOctaves = 3,
		int NumScaleLevels = 4,
		cv::Rect ROI = cv::Rect());
};
