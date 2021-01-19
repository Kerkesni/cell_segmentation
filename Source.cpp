#include "Header.h"

#include <iostream>
#include <string.h>

int main(int argc, char* argv[]) {

	cv::Mat image;

	image = cv::imread("./datasets/CUSTOM/BM_GRAZ_HE_0001_01.png");

	if (!image.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	cv::namedWindow("Display window", cv::WINDOW_NORMAL);

	// lose the Alpha channel
	if (image.channels() == 4) {
		cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
	}

	cv::Mat patch(image, cv::Rect(cv::Point2i(0, 200), cv::Point2i(200, 400)));

	// convert to grayscale
	cv::Mat grayImg;
	cv::cvtColor(patch, grayImg, cv::COLOR_BGR2GRAY);

	// Binarizing image
	cv::Mat binImage;
	cv::GaussianBlur(grayImg, binImage, cv::Size(5, 5), 0);
	cv::threshold(binImage, binImage, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

	// Morph images
	cv::Mat morphed;
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(binImage, morphed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

	// Get cell centers
	std::vector<cv::Point2f> centers = getCellCenters(morphed);

	// Draw centers
	for (cv::Point2f point : centers) {
		cv::circle(patch, point, 2, CV_RGB(0, 255, 0), -1, 2, 0);
	}
	
	// display the image
	cv::imshow("Display window", patch);
	cv::waitKey(0);

	// cv::imwrite("patch.jpg", patch);

	cv::destroyAllWindows();

	return 0;
}

