#include "imanalysis.hpp"

/**
* Feature Extraction
*/
std::pair<std::vector<cv::KeyPoint>, cv::Mat> cvx::matlab::detectSURFFeatures(const cv::Mat& inp, float MetricThreshold, int NumOctaves, int NumScaleLevels, cv::Rect ROI) {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	cvx::matlab::detectSURFFeatures(inp, keypoints, descriptors, MetricThreshold, NumOctaves, NumScaleLevels, ROI);
	return { keypoints, descriptors };
}

void cvx::matlab::detectSURFFeatures(const cv::Mat& inp, std::vector<cv::KeyPoint>& output, cv::Mat& descriptors, float MetricThreshold, int NumOctaves, int NumScaleLevels, cv::Rect ROI) {
	CV_Assert(MetricThreshold > 0);
	CV_Assert(NumOctaves > 1);
	CV_Assert(NumScaleLevels >= 3);

	if (ROI.empty()) {
		ROI = cv::Rect{ 0, 0, inp.cols, inp.rows };
	}

	const cv::Mat sub_img = inp(ROI);

	auto surf_ptr = cv::xfeatures2d::SURF::create(MetricThreshold, NumOctaves, NumScaleLevels - 2);
	surf_ptr->detectAndCompute(sub_img, cv::noArray(), output, descriptors);
}


/**
* Feature Matching
*/
std::vector<cv::DMatch> cvx::matlab::matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, float ratio_thresh, int max_matches) {
	std::vector<cv::DMatch> good_matches;
	cvx::matlab::matchFeatures(descriptors1, descriptors2, good_matches, ratio_thresh, max_matches);
	return good_matches;
}
void cvx::matlab::matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2, std::vector<cv::DMatch>& output, float ratio_thresh, int max_matches) {
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<cv::DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, max_matches);

	for (size_t i = 0; i < knn_matches.size(); i++) {
		if (knn_matches[i][0].distance < 1.0-ratio_thresh * knn_matches[i][1].distance) {
			output.push_back(knn_matches[i][0]);
		}
	}
}


/**
* Gradiants
*/
void cvx::matlab::gradiant1D(const cv::Mat& inp, cv::Mat& dst) {
	auto inpSize = inp.size();
	if (dst.empty() || dst.type() != CV_32FC1) {
		dst = cv::Mat(inpSize, CV_32FC1);
	}

	// make sure the image is 1D in one direction or the other
	CV_Assert((inp.cols == 1 && inp.rows > 1) || (inp.cols > 1 && inp.rows == 1));

	if (inpSize.width > 1 && inpSize.height == 1) {
		// horizontal case
		cv::Mat_<float> kernel(1, 3, CV_32FC1); kernel << -0.5, 0.0, 0.5;
		// first column 
		cv::Mat first_column;
		cv::subtract(inp(cv::Rect{ 1, 0, 1, inp.rows }), inp(cv::Rect{ 0,0, 1, inp.rows }), first_column);
		// last column
		cv::Mat last_column;
		cv::subtract(inp(cv::Rect{ inp.cols - 1, 0, 1, inp.rows }), inp(cv::Rect{ inp.cols - 2,0, 1, inp.rows }), last_column);
		cv::filter2D(inp(cv::Rect{ 1,0, inp.cols - 2, 1 }),
			dst(cv::Rect{ 1,0, inp.cols - 2, 1 }),
			CV_32FC1,
			kernel,
			cv::Point(-1, -1),
			0,
			cv::BORDER_DEFAULT);
		first_column.copyTo(dst(cv::Rect{ 0,0, 1, inp.rows }));
		last_column.copyTo(dst(cv::Rect{ inp.cols - 1, 0, 1, inp.rows }));

	} else if (inpSize.width == 1 && inpSize.height > 1) {
		// vertical case
		cv::Mat_<float> kernel(3, 1, CV_32FC1); kernel << -0.5, 0.0, 0.5;
		// first column 
		cv::Mat first_row;
		cv::subtract(inp(cv::Rect{ 0, 1, inp.cols, 1 }), inp(cv::Rect{ 0,0, inp.cols, 1 }), first_row);
		// last column
		cv::Mat last_row;
		cv::subtract(inp(cv::Rect{ 0, inp.rows - 1, inp.cols, 1 }), inp(cv::Rect{ 0, inp.rows - 2, inp.cols, 1 }), last_row);
		cv::filter2D(inp(cv::Rect{ 0,1, 1, inp.rows - 2 }),
			dst(cv::Rect{ 0,1, 1, inp.rows - 2 }),
			CV_32FC1,
			kernel,
			cv::Point(-1, -1),
			0,
			cv::BORDER_DEFAULT);
		first_row.copyTo(dst(cv::Rect{ 0,0, inp.cols, 1 }));
		last_row.copyTo(dst(cv::Rect{ 0, inp.rows - 1, inp.cols, 1 }));
	}
}

cv::Mat cvx::matlab::gradiant1D(const cv::Mat& inp) {
	cv::Mat dst(inp.size(), CV_32FC1);
	cvx::matlab::gradiant1D(inp, dst);
	return dst;
}

void cvx::matlab::gradiant2D(const cv::Mat& inp, cv::Mat& gx, cv::Mat& gy) {
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
	cv::subtract(inp(cv::Rect{ 1, 0, 1, inp.rows }), inp(cv::Rect{ 0,0, 1, inp.rows }), first_column);
	// last column
	cv::Mat last_column;
	cv::subtract(inp(cv::Rect{ inp.cols - 1, 0, 1, inp.rows }), inp(cv::Rect{ inp.cols - 2,0, 1, inp.rows }), last_column);
	cv::filter2D(inp(cv::Rect{ 1,0, inp.cols - 2, inp.rows }),
		gx(cv::Rect{ 1,0, inp.cols - 2, inp.rows }),
		CV_32FC1,
		gx_kernel,
		cv::Point(-1, -1),
		0,
		cv::BORDER_DEFAULT);
	first_column.copyTo(gx(cv::Rect{ 0,0, 1, inp.rows }));
	last_column.copyTo(gx(cv::Rect{ inp.cols - 1, 0, 1, inp.rows }));

	// vertical gradiant
	cv::Mat_<float> gy_kernel(3, 1, CV_32FC1); gy_kernel << -0.5, 0.0, 0.5;
	// first column 
	cv::Mat first_row;
	cv::subtract(inp(cv::Rect{ 0, 1, inp.cols, 1 }), inp(cv::Rect{ 0,0, inp.cols, 1 }), first_row);
	// last column
	cv::Mat last_row;
	cv::subtract(inp(cv::Rect{ 0, inp.rows - 1, inp.cols, 1 }), inp(cv::Rect{ 0, inp.rows - 2, inp.cols, 1 }), last_row);
	cv::filter2D(inp(cv::Rect{ 0, 1, inp.cols, inp.rows - 2 }),
		gy(cv::Rect{ 0, 1, inp.cols, inp.rows - 2 }),
		CV_32FC1,
		gy_kernel,
		cv::Point(-1, -1),
		0,
		cv::BORDER_DEFAULT);
	first_row.copyTo(gy(cv::Rect{ 0,0, inp.cols, 1 }));
	last_row.copyTo(gy(cv::Rect{ 0, inp.rows - 1, inp.cols, 1 }));
}

std::array<cv::Mat, 2> cvx::matlab::gradiant2D(const cv::Mat& inp) {
	cv::Mat gx(inp.size(), CV_32FC1);
	cv::Mat gy(inp.size(), CV_32FC1);
	cvx::matlab::gradiant2D(inp, gx, gy);
	return { gx, gy };
}


/**
* Median
*/
std::optional<float> cvx::matlab::median8U(const cv::Mat& inp) {
	// surely there's a more performant way of doing this...
	if (!inp.isContinuous()) { return std::nullopt; }

	std::vector<char> res;

	res.assign((char*)inp.data, (char*)inp.data + inp.total());
	auto m = res.begin() + res.size() / 2;
	std::nth_element(res.begin(), m, res.end());

	return res[res.size() / 2];
}

std::optional<float> cvx::matlab::median16U(const cv::Mat& inp) {
	// surely there's a more performant way of doing this...
	if (!inp.isContinuous()) { return std::nullopt; }

	std::vector<uint16_t> res;

	res.assign((uint16_t*)inp.data, (uint16_t*)inp.data + inp.total());
	auto m = res.begin() + res.size() / 2;
	std::nth_element(res.begin(), m, res.end());

	return res[res.size() / 2];
}

std::optional<float> cvx::matlab::median32F(const cv::Mat& inp) {
	// surely there's a more performant way of doing this...
	if (!inp.isContinuous()) { return std::nullopt; }

	std::vector<float> res;

	res.assign((float*)inp.data, (float*)inp.data + inp.total());
	auto m = res.begin() + res.size() / 2;
	std::nth_element(res.begin(), m, res.end());

	return res[res.size() / 2];
}

std::optional<float> cvx::matlab::median64F(const cv::Mat& inp) {
	// surely there's a more performant way of doing this...
	if (!inp.isContinuous()) { return std::nullopt; }

	std::vector<double> res;

	res.assign((double*)inp.data, (double*)inp.data + inp.total());
	auto m = res.begin() + res.size() / 2;
	std::nth_element(res.begin(), m, res.end());

	return res[res.size() / 2];
}