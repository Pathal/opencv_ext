#pragma once

#include <array>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	namespace matlab {
		void imsharpen(const cv::Mat& inp,
			cv::Mat& dst,
			double ratio,
			cv::Size kSize,
			double sigmaX,
			double sigmaY = 0,
			int borderType = cv::BORDER_DEFAULT,
			cv::Mat swap = cv::Mat());
		cv::Mat imsharpen(const cv::Mat& inp,
			double ratio,
			cv::Size kSize,
			double sigmaX,
			double sigmaY = 0,
			int borderType = cv::BORDER_DEFAULT,
			cv::Mat swap = cv::Mat());


		cv::Mat imerode(cv::Mat& inp, cv::Mat& strel);
		cv::Mat imdilate(cv::Mat& inp, cv::Mat& strel);
		// imopen is a erosion followed by a dilation
		cv::Mat imopen(cv::Mat& inp, cv::Mat& strel);
		// imclose is a dilation followed by an erosion
		cv::Mat imclose(cv::Mat& inp, cv::Mat& strel);

	}

	namespace common {
		void clamp(cv::Mat& inp, cv::Scalar min, cv::Scalar max);
		void clamp(cv::Mat& inp, cv::Mat& dest, cv::Scalar min, cv::Scalar max);
	}
};
