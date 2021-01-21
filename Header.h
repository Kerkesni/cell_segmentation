#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <iostream>

// White cells dark background
#define FRST_MODE_BRIGHT 1
// Dark cells bright background
#define FRST_MODE_DARK 2
// A mix of both
#define FRST_MODE_BOTH 3

/**
	Calculate vertical gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void grady(const cv::Mat& input, cv::Mat &output)
{
	output = cv::Mat::zeros(input.size(), CV_64FC1);
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 1; x < input.cols - 1; x++)
		{
			*((double*)output.data + y * output.cols + x) = (double)(*(input.data + y * input.cols + x + 1) - *(input.data + y * input.cols + x - 1)) / 2;
		}
	}
}

/**
	Calculate horizontal gradient for the input image

	@param input Input 8-bit image
	@param output Output gradient image
*/
void gradx(const cv::Mat& input, cv::Mat &output)
{
	output = cv::Mat::zeros(input.size(), CV_64FC1);
	for (int y = 1; y < input.rows - 1; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			*((double*)output.data + y * output.cols + x) = (double)(*(input.data + (y + 1)*input.cols + x) - *(input.data + (y - 1)*input.cols + x)) / 2;
		}
	}
}

/**
	Applies Fast radial symmetry transform to image
	Check paper Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for
	detecting points of interest. Computer Vision, ECCV 2002.

	@param inputImage The input grayscale image (8-bit)
	@param outputImage The output image containing the results of FRST
	@param radii Gaussian kernel radius
	@param alpha Strictness of radial symmetry
	@param stdFactor Standard deviation factor
	@param mode Transform mode ('bright', 'dark' or 'both')
*/
void frst2d(const cv::Mat& inputImage, cv::Mat& outputImage, const int radii, const double alpha, const double stdFactor, const int mode)
{
	int width = inputImage.cols;
	int height = inputImage.rows;

	cv::Mat gx, gy;
	gradx(inputImage, gx);
	grady(inputImage, gy);

	std::cout << gx.at<double>(50, 50) << std::endl;


	// set dark/bright mode
	bool dark = false;
	bool bright = false;

	if (mode == FRST_MODE_BRIGHT)
		bright = true;
	else if (mode == FRST_MODE_DARK)
		dark = true;
	else if (mode == FRST_MODE_BOTH) {
		bright = true;
		dark = true;
	}
	else {
		throw std::exception("invalid mode!");
	}

	outputImage = cv::Mat::zeros(inputImage.size(), CV_64FC1);

	cv::Mat S = cv::Mat::zeros(inputImage.rows + 2 * radii, inputImage.cols + 2 * radii, outputImage.type());

	cv::Mat O_n = cv::Mat::zeros(S.size(), CV_64FC1);
	cv::Mat M_n = cv::Mat::zeros(S.size(), CV_64FC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			cv::Point p(i, j);

			cv::Vec2d g = cv::Vec2d(gx.at<double>(i, j), gy.at<double>(i, j));

			double gnorm = std::sqrt(g.val[0] * g.val[0] + g.val[1] * g.val[1]);

			if (gnorm > 0) {

				cv::Vec2i gp;
				gp.val[0] = (int)std::round((g.val[0] / gnorm) * radii);
				gp.val[1] = (int)std::round((g.val[1] / gnorm) * radii);

				if (bright) {
					cv::Point ppve(p.x + gp.val[0] + radii, p.y + gp.val[1] + radii);

					O_n.at<double>(ppve.x, ppve.y) = O_n.at<double>(ppve.x, ppve.y) + 1;
					M_n.at<double>(ppve.x, ppve.y) = M_n.at<double>(ppve.x, ppve.y) + gnorm;
				}

				if (dark) {
					cv::Point pnve(p.x - gp.val[0] + radii, p.y - gp.val[1] + radii);

					O_n.at<double>(pnve.x, pnve.y) = O_n.at<double>(pnve.x, pnve.y) - 1;
					M_n.at<double>(pnve.x, pnve.y) = M_n.at<double>(pnve.x, pnve.y) - gnorm;
				}
			}
		}
	}

	double min, max;

	O_n = cv::abs(O_n);
	cv::minMaxLoc(O_n, &min, &max);
	O_n = O_n / max;

	M_n = cv::abs(M_n);
	cv::minMaxLoc(M_n, &min, &max);
	M_n = M_n / max;

	cv::pow(O_n, alpha, S);
	S = S.mul(M_n);

	int kSize = std::ceil(radii / 2);
	if (kSize % 2 == 0)
		kSize++;

	cv::GaussianBlur(S, S, cv::Size(kSize, kSize), radii * stdFactor);

	outputImage = S(cv::Rect(radii, radii, width, height));
}

/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit). The resulting image overwrites the input
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(cv::Mat& inputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 3, const int iterations = 1)
{
	int _mSize = (mSize % 2) ? mSize : mSize + 1;

	cv::Mat element = cv::getStructuringElement(mShape, cv::Size(_mSize, _mSize));
	cv::morphologyEx(inputImage, inputImage, operation, element, cv::Point(-1, -1), iterations);
}

/**
Perform the specified morphological operation on input image with structure element of specified type and size
@param inputImage Input image of any type (preferrably 8-bit)
@param outputImage Output image of the same size and type as the input image
@param operation Name of the morphological operation (MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE)
@param mShape Shape of the structure element (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
@param mSize Size of the structure element
@param iterations Number of iterations, how many times to perform the morphological operation
*/
void bwMorph(const cv::Mat& inputImage, cv::Mat& outputImage, const int operation, const int mShape = cv::MORPH_RECT, const int mSize = 3, const int iterations = 1)
{
	inputImage.copyTo(outputImage);

	bwMorph(outputImage, operation, mShape, mSize, iterations);
}

/**
Returns the centers in an array
@param contours found contours in image
@param centerOfMass get the center of masse of the blobs
@param contourIdx number of elements (contours) in vector
*/
std::vector<cv::Point> bhContoursCenter(const std::vector<std::vector<cv::Point>>& contours, bool centerOfMass, int contourIdx = -1)
{
	std::vector<cv::Point> result;
	if (contourIdx > -1)
	{
		if (centerOfMass)
		{
			cv::Moments m = moments(contours[contourIdx], true);
			result.push_back(cv::Point(m.m10 / m.m00, m.m01 / m.m00));
		}
		else
		{
			cv::Rect rct = boundingRect(contours[contourIdx]);
			result.push_back(cv::Point(rct.x + rct.width / 2, rct.y + rct.height / 2));
		}
	}
	else
	{
		if (centerOfMass)
		{
			for (int i = 0; i < contours.size(); i++)
			{
				cv::Moments m = moments(contours[i], true);
				result.push_back(cv::Point(m.m10 / m.m00, m.m01 / m.m00));

			}
		}
		else
		{

			for (int i = 0; i < contours.size(); i++)
			{
				cv::Rect rct = boundingRect(contours[i]);
				result.push_back(cv:: Point(rct.x + rct.width / 2, rct.y + rct.height / 2));
			}
		}
	}

	return result;
}

/**
Performs non maximum suppression (removing non maximum local points)
@param image Input Image
@param neighbor local neighborhood size to consider
*/
std::vector<cv::Point> bhFindLocalMaximum(cv::Mat  _src, int neighbor = 12) {
	cv::Mat src = _src;

	cv::Mat peak_img = src.clone();
	cv::dilate(peak_img, peak_img, cv::Mat(), cv::Point(-1, -1), neighbor);
	peak_img = peak_img - src;



	cv::Mat flat_img;
	erode(src, flat_img, cv::Mat(), cv::Point(-1, -1), neighbor);
	flat_img = src - flat_img;


	threshold(peak_img, peak_img, 0, 255, cv::THRESH_BINARY);
	threshold(flat_img, flat_img, 0, 255, cv::THRESH_BINARY);
	bitwise_not(flat_img, flat_img);

	peak_img.setTo(cv::Scalar::all(255), flat_img);
	bitwise_not(peak_img, peak_img);


	std::vector<std::vector<cv::Point>> contours;
	findContours(peak_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	return bhContoursCenter(contours, true);
}

/**
returns center points of cells
@param inputImage Input image (binary grayscale 8-bit)
*/
std::vector<cv::Point> getCellCenters(const cv::Mat& inputImage) {

	// apply FRST
	cv::Mat frstImage;
	frst2d(inputImage, frstImage, 8, 2, 0.1, FRST_MODE_DARK);

	// display the image
	cv::imshow("Display window", frstImage);
	cv::waitKey(0);

	// normalising the values of the FRST algorithm to [0, 1] because they are not regular
	cv::normalize(frstImage, frstImage, 0.0, 1.0, cv::NORM_MINMAX);
	frstImage.convertTo(frstImage, CV_8U, 255.0);

	// display the image
	cv::imshow("Display window", frstImage);
	cv::waitKey(0);

	// Setting up a flat kernel of size 3x3
	cv::Mat_<float> kernel(3, 3);
	kernel << 1, 1, 1, 1, 1, 1, 1, 1, 1;

	// Dilatation of the response to remove irrelevant points (noise)
	cv::morphologyEx(frstImage, frstImage, cv::MORPH_DILATE, kernel, cv::Point(-1, -1), 1);

	// display the image
	cv::imshow("Display window", frstImage);
	cv::waitKey(0);

	// applying non max suppression and retreiving local maxima points (centers)
	std::vector<cv::Point> localMax = bhFindLocalMaximum(frstImage, 10);

	return localMax;
}
