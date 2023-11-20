#include "imanalysis.hpp"

std::vector<cv::KeyPoint> cvx::detectSURFFeatures(const cv::Mat& inp, float MetricThreshold, int NumOctaves, int NumScaleLevels, cv::Rect ROI) {
	CV_Assert(MetricThreshold > 0);
	CV_Assert(NumOctaves > 1);
	CV_Assert(NumScaleLevels >= 3);

	if (ROI.empty()) {
		ROI = cv::Rect{ 0, 0, inp.cols, inp.rows };
	}

	const cv::Mat sub_img = inp(ROI);
	
	auto surf_ptr = cv::xfeatures2d::SURF::create(MetricThreshold, NumOctaves, NumScaleLevels-2);
	std::vector<cv::KeyPoint> keypoints;
	surf_ptr->detect(sub_img, keypoints);
	return keypoints;
}
