#pragma once

#include <array>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx {
	float getPixelAsFloat(const cv::Mat& inp, int x, int y);

	namespace matlab {
		std::array<double, 4> quatmultiply(std::array<double, 4> q, std::array<double, 4> r);

		/**
		* We need our own STREL shapes enum because OpenCV has a very limited number of shapes it normally supports
		* otherwise we would use cv::MORPH_type
		*/
		enum class STREL_SHAPE {
			// 2d
			DIAMOND, // 2*r+1 width and height
			DISK, // nearly identical to octagon, not actually rounded
			OCTAGON,
			LINE, // numbers don't make sense? Length of 10 is 7 elements?
			RECTANGLE,
			SQUARE,
			// 3d
			CUBE,
			CUBOID,
			SPHERE
		};

		/**
		* https://www.mathworks.com/help/images/ref/strel.html
		* The rules of the parameters vary depending on the shape given.
		* RECTANGLE & SQUARE - Use the dims, if empty dims is set to a square of length "radius"
		* DISK - Uses the CV Ellipse structuring element generator with a size of radius*2+1 in both directions
		* Others - Ignored for now
		*/
		cv::Mat strel(STREL_SHAPE shape, cv::Size& dims, double radius);
	};
};
