#include "imtrans.hpp"

#include <iostream>

void cvx::imsharpen(const cv::Mat& inp, cv::Mat& dst, double ratio, cv::Size kSize, double sigmaX, double sigmaY, int borderType, cv::Mat swap) {
	cv::GaussianBlur(inp, swap, kSize, sigmaX, sigmaY, borderType);
	cv::addWeighted(inp, 1.0+ratio, swap, -ratio, 0, dst);
}
cv::Mat cvx::imsharpen(const cv::Mat& inp, double ratio, cv::Size kSize, double sigmaX, double sigmaY, int borderType, cv::Mat swap) {
	cv::Mat dst;
	cvx::imsharpen(inp, dst, ratio, kSize, sigmaX, sigmaY, borderType, swap);
	return dst;
}

void cvx::gradiant1D(const cv::Mat& inp, cv::Mat& dst) {
	auto inpSize = inp.size();
	if (dst.empty() || dst.type() != CV_32FC1) {
		dst = cv::Mat(inpSize, CV_32FC1);
	}

	// make sure the image is 1D in one direction or the other
	CV_Assert((inp.cols == 1 && inp.rows > 1) || (inp.cols > 1 && inp.rows == 1));

	if (inpSize.width > 1 && inpSize.height == 1) {
		// horizontal case
		cv::Mat_<float> kernel(1, 3, CV_32FC1); kernel << -0.5, 0.0, 0.5;
		float first_val = getPixelAsFloat(inp, 0, 1) - getPixelAsFloat(inp, 0, 0);
		float last_val  = getPixelAsFloat(inp, 0, inp.cols-1) - getPixelAsFloat(inp, 0, inp.cols-2);
		cv::filter2D(inp(cv::Rect{ 1,0, inp.cols - 2, 1 }),
			dst(cv::Rect{ 1,0, inp.cols - 2, 1 }),
			CV_32FC1,
			kernel,
			cv::Point(-1, -1),
			0,
			cv::BORDER_DEFAULT);
		dst.at<float>(0, 0) = first_val;
		dst.at<float>(0, dst.cols-1) = last_val;

	} else if (inpSize.width == 1 && inpSize.height > 1) {
		// vertical case
		cv::Mat_<float> kernel(3, 1, CV_32FC1); kernel << -0.5, 0.0, 0.5;
		float first_val = getPixelAsFloat(inp, 1, 0) - getPixelAsFloat(inp, 0, 0);
		float last_val = getPixelAsFloat(inp, inp.rows - 1, 0) - getPixelAsFloat(inp, inp.rows - 2, 0);
		cv::filter2D(inp(cv::Rect{ 0,1, 1, inp.rows - 2 }),
			dst(cv::Rect{ 0,1, 1, inp.rows - 2 }),
			CV_32FC1,
			kernel,
			cv::Point(-1, -1),
			0,
			cv::BORDER_DEFAULT);
		dst.at<float>(0, 0) = first_val;
		dst.at<float>(dst.rows - 1, 0) = last_val;
	}
}

cv::Mat cvx::gradiant1D(const cv::Mat& inp) {
	cv::Mat dst(inp.size(), CV_32FC1);
	gradiant1D(inp, dst);
	return dst;
}

void cvx::gradiant2D(const cv::Mat& inp, cv::Mat& gx, cv::Mat& gy) {
	CV_Assert(inp.cols > 1 && inp.rows > 1);

	auto inpSize = inp.size();
	if (gx.empty() || gx.type() != CV_32FC1) {
		gx = cv::Mat(inpSize, CV_32FC1);
	}
	if (gy.empty() || gy.type() != CV_32FC1) {
		gy = cv::Mat(inpSize, CV_32FC1);
	}

	// horizontal gradiant
	cv::Mat_<float> gx_kernel(1, 3, CV_32FC1); gx_kernel << -0.5, 0.0, 0.5;
	// first column 
	cv::Mat first_column;
	cv::subtract(inp(cv::Rect{1, 0, 1, inp.rows}), inp(cv::Rect{0,0, 1, inp.rows}), first_column);
	// last column
	cv::Mat last_column;
	cv::subtract(inp(cv::Rect{ inp.cols-1, 0, 1, inp.rows }), inp(cv::Rect{ inp.cols-2,0, 1, inp.rows }), last_column);
	cv::filter2D(inp(cv::Rect{ 1,0, inp.cols - 2, inp.rows }),
		gx(cv::Rect{ 1,0, inp.cols - 2, inp.rows }),
		CV_32FC1,
		gx_kernel,
		cv::Point(-1, -1),
		0,
		cv::BORDER_DEFAULT);
	first_column.copyTo(gx(cv::Rect{ 0,0, 1, inp.rows }));
	last_column.copyTo( gx(cv::Rect{ inp.cols - 1, 0, 1, inp.rows }));

	// vertical gradiant
	cv::Mat_<float> gy_kernel(3, 1, CV_32FC1); gy_kernel << -0.5, 0.0, 0.5;
	// first column 
	cv::Mat first_row;
	cv::subtract(inp(cv::Rect{ 0, 1, inp.cols, 1 }), inp(cv::Rect{ 0,0, inp.cols, 1 }), first_row);
	// last column
	cv::Mat last_row;
	cv::subtract(inp(cv::Rect{ 0, inp.rows-1, inp.cols, 1 }), inp(cv::Rect{ 0, inp.rows-2, inp.cols, 1 }), last_row);
	cv::filter2D(inp(cv::Rect{ 0, 1, inp.cols, inp.rows-2 }),
		gy(cv::Rect{ 0, 1, inp.cols, inp.rows - 2 }),
		CV_32FC1,
		gy_kernel,
		cv::Point(-1, -1),
		0,
		cv::BORDER_DEFAULT);
	first_row.copyTo(gy(cv::Rect{ 0,0, inp.cols, 1 }));
	last_row.copyTo( gy(cv::Rect{ 0, inp.rows - 1, inp.cols, 1 }));
}

std::array<cv::Mat, 2> cvx::gradiant2D(const cv::Mat& inp) {
	cv::Mat gx(inp.size(), CV_32FC1);
	cv::Mat gy(inp.size(), CV_32FC1);
	cvx::gradiant2D(inp, gx, gy);
	return {gx, gy};
}

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