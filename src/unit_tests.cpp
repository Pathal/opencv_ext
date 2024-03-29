#include "imtrans.hpp"
#include "imanalysis.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


const bool TEST_MEDIAN = true;
const bool TEST_GRADIANT = true;
const bool TEST_FEATURE_EXTRACT = true;
const bool TEST_CLAMP = true;
const bool STEP_THROUGH_UNIT_TESTS = false;

bool compare_float(float x, float y, float epsilon = 0.0001f) {
	if (fabs(x - y) < epsilon)
		return true; //they are same
	return false; //they are not same
}

int main() {
	if (TEST_MEDIAN) {			// Median value
		std::cout << "Testing Median " << std::endl;
		cv::Mat mat(800, 800, CV_8UC1);
		cv::randu(mat, cv::Scalar(0), cv::Scalar(256));
		auto median_val = cvx::common::median(mat);
		//std::cout << median_val.value() << std::endl;
		assert(median_val.has_value());
		assert(compare_float(median_val.value(), 127.0, 1.1));

		cv::randn(mat, cv::Scalar(40.0), cv::Scalar(20.0));
		median_val = cvx::common::median(mat);
		//std::cout << median_val.value() << std::endl;
		assert(median_val.has_value());
		assert(compare_float(median_val.value(), 40.0, 1.1));
	}


	if (TEST_GRADIANT) {		// Gradiant 1D
		std::cout << "Testing Gradiant1D" << std::endl;
		cv::Size row_dim(10, 1);
		uint8_t row_data[] = { 0, 4, 4, 5, 9, 14, 20, 40, 100, 101 };
		cv::Mat single_row(1, 10, CV_8UC1, row_data);
		cv::Mat single_col(10, 1, CV_8UC1, row_data);
		float row_data_res[] = { 4.0000, 2.0000, 0.5000, 2.5000, 4.5000, 5.5000, 13.0000, 40.0000, 30.5000, 1.0000 };

		cv::Mat dst = cvx::matlab::gradiant1D(single_row);
		assert(dst.size().area() == 10);
		for (int i = 0; i < 10; i++) {
			float pix_value = *(float*)(void*)&dst.data[i * dst.elemSize()]; // this can't be right...
			assert(compare_float(pix_value, row_data_res[i]));
		}

		dst = cvx::matlab::gradiant1D(single_col);
		assert(dst.size().area() == 10);
		for (int i = 0; i < 10; i++) {
			float pix_value = *(float*)(void*)&dst.data[i * dst.elemSize()]; // this can't be right...
			assert(compare_float(pix_value, row_data_res[i]));
		}
	}


	if (TEST_GRADIANT) {		// Gradiant 2D
		std::cout << "Testing Gradiant2D" << std::endl;
		cv::Size inpSize(6, 6);
		uint8_t inpData[] = { 0,  1,  4,  7,  8, 10,
							 5, 10, 15, 20, 25, 30,
							10, 10, 10, 10, 10, 10,
							 4,  8, 16, 32, 64,128,
							 0,  1,  2,  3,  4,  5,
							 6, 12, 18, 24, 30, 36 };
		cv::Mat inp(inpSize, CV_8UC1, inpData);
		auto [gx, gy] = cvx::matlab::gradiant2D(inp);
		assert(gx.size().area() == 36);
		assert(gy.size().area() == 36);
	}


	if (TEST_FEATURE_EXTRACT) {	// SURF feature extraction
		std::cout << "Testing Feature Extraction and Matching" << std::endl;
		cv::Mat inp = cv::Mat::zeros(cv::Size(500, 500), CV_8UC1);
		inp(cv::Rect{50, 50, 80, 80}).setTo(cv::Scalar(255));
		//std::cout << inp << std::endl;
		auto [keypoints, descriptors] = cvx::matlab::detectSURFFeatures(inp);
		cv::Mat img_keypoints;
		cv::drawKeypoints(inp, keypoints, img_keypoints);
		if (STEP_THROUGH_UNIT_TESTS) {
			cv::imshow("keypoints", img_keypoints);
		}

		auto matches = cvx::matlab::matchFeatures(descriptors, descriptors);

		cv::Mat img_matches;
		cv::drawMatches(inp, keypoints, inp, keypoints, matches, img_matches, cv::Scalar::all(-1),
			cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		if (STEP_THROUGH_UNIT_TESTS) {
			cv::imshow("matched features", img_matches);
			cv::waitKey(0);
		}
	}

	if (TEST_CLAMP) {			// Clamping values to a range
		std::cout << "Testing Clamp" << std::endl;
		cv::Mat inp(800, 800, CV_32FC1);
		cv::randu(inp, cv::Scalar(0.0), cv::Scalar(256.0));
		cvx::common::clamp(inp, 100, 100);
		if (STEP_THROUGH_UNIT_TESTS) {
			cv::imshow("clamp", inp);
			cv::waitKey(0);
		}
		for (int i = 0; i < 10; i++) {
			float pix_value = *(float*)(void*)&inp.data[i * inp.elemSize()]; // this can't be right...
			assert(compare_float(pix_value, 100));
		}

	}

	return 0;
}
