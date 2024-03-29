#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	namespace matlab {
		// detectSURFFeatures
		std::pair<std::vector<cv::KeyPoint>, cv::Mat> detectSURFFeatures(
			const cv::Mat& inp,
			float MetricThreshold = 1000,
			int NumOctaves = 3,
			int NumScaleLevels = 4,
			cv::Rect ROI = cv::Rect());
		void detectSURFFeatures(
			const cv::Mat& inp,
			std::vector<cv::KeyPoint>& output,
			cv::Mat& descriptors,
			float MetricThreshold = 1000,
			int NumOctaves = 3,
			int NumScaleLevels = 4,
			cv::Rect ROI = cv::Rect());

		// matchFeatures
		std::vector<cv::DMatch> matchFeatures(
			const cv::Mat& descriptors1,
			const cv::Mat& descriptors2,
			float ratio_thresh = 0.3,
			int max_matches = 2);
		void matchFeatures(
			const cv::Mat& descriptors1,
			const cv::Mat& descriptors2,
			std::vector<cv::DMatch>& output,
			float ratio_thresh = 0.3,
			int max_matches = 2);

		// Gradiant
		void gradiant1D(const cv::Mat& inp, cv::Mat& dst);
		cv::Mat gradiant1D(const cv::Mat& inp);

		void gradiant2D(const cv::Mat& inp, cv::Mat& gx, cv::Mat& gy);
		std::array<cv::Mat, 2> gradiant2D(const cv::Mat& inp);

		// estimateGeometricTransform
		void estimateGeometricTransform(
			std::vector<cv::Point2f>& obj,
			std::vector<cv::Point2f>& scene,
			std::vector<cv::DMatch>& matches,
			double maxDistance = 1.5,
			double confidence = 99.0, // matlab defaults to 99.0, opencv defaults to 0.995
			int maxNumTrials = 1000); // matlab defaults to 1000, opencv defaults to 2000

		// semanticseg
		void semanticSeg(cv::Mat& inp);
	}

	namespace common {
		// Median
		std::optional<float> median(const cv::Mat&);
	}
};
