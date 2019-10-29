#include "ImageProcessing.h"

void ImageProcessing::CannyEdgeDetection(const cv::Mat& in_image, cv::Mat& out_image, int l_threshold, int h_threshold) {
	int height = in_image.rows;
	int width = in_image.cols;
	/*
		First step is blurring the image using Gaussian Filter to remove the obvious noise. We use
		sigma = 1.4 and 5x5 matrix because big values of ksize and sigma will negatively affect
		the performance of the detector. This is the best size for most of the cases.
	*/
	GaussianBlur(in_image, out_image, 3, 1.4);
	//cv::imshow("After 1 step", out_image);
	/*
		Now we need to calculate the intensity gradient of the image.
		We will use 3x3 Scharr filter because it does better than
		Sobel operator, however Prewitt, Cross or 5x5 Sobel might
		improve the quality of edge detection.

		angles is an array with gradient directions.
	*/
	double** angles = ScharrOperator(out_image, out_image);
	cv::imshow("After 2 step", out_image);
	/*
		Thirdly, we need to do non-maximum suppression to suppress all the gradient values except
		the local maxima, which indicate locations with the sharpest change of intensity value.

		Wa take our center pixel at (i, j) coordinate and compare it with the pixels in positive
		and negative gradient directions. For example, if gradient direction is 45, we compare it
		with pixels at (i+1, j-1) and (i-1, j+1). If the center pixel is lesser, we suppress it.

		As a result, we get binary image with thin edges.
	*/
	for (int i = 1; i < height - 1; ++i) {
		for (int j = 1; j < width - 1; ++j) {
			double center_value = out_image.data[i * width + j];
			double l_value = 255;
			double r_value = 255;
			double angle = angles[i][j] * 180 / PI;
			if ((angle <= 22.5 && angle >= -22.5) || (angle >= 157.5 || angle <= -157.5)) {
				l_value = out_image.data[i * width + j + 1];
				r_value = out_image.data[i * width + j - 1];
			}
			else if ((angle >= 22.5 && angle <= 67.5) || (angle >= -157.5 && angle <= -112.5)) {
				l_value = out_image.data[(i + 1) * width + j - 1];
				r_value = out_image.data[(i - 1) * width + j + 1];
			}
			else if ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5)) {
				l_value = out_image.data[(i + 1) * width + j];
				r_value = out_image.data[(i - 1) * width + j];
			}
			else {
				l_value = out_image.data[(i + 1) * width + j + 1];
				r_value = out_image.data[(i - 1) * width + j - 1];
			}

			if (center_value <= l_value || center_value <= r_value)
				out_image.data[i * width + j] = 0;
			else
				out_image.data[i * width + j] = center_value;
		}
	}
	cv::imshow("After 3 step", out_image);
	/*
		The fourth step is called Double Threshold. In this step we remove some edge
		pixels caused by noise and color variation. We need to select high and low
		threshold values. If pixel value is higher than high threshold, we mark it as
		a strong edge pixel. If it value is between low and high threshold, we mark it as
		a weak edge pixel. Otherwise, we suppress it.
	*/
	double low_threshold = 10;
	double high_threshold = 20;
	if (l_threshold != -1)
		low_threshold = l_threshold;
	if (h_threshold != -1)
		high_threshold = h_threshold;

	//We launch recursive function on the strong edges to make weak edges connected to them strong too
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (out_image.data[i * width + j] >= high_threshold) {				
				out_image.data[i * width + j] = 255;
				recursiveHysteresis(out_image, low_threshold, high_threshold, i, j);
			}
		}
	}
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (out_image.data[i * width + j] != 255)
				out_image.data[i * width + j] = 0;
		}
	}
}
void ImageProcessing::recursiveHysteresis(cv::Mat& image, int low_threshold, int high_threshold, int i, int j, int length) {
	if (length > 5)
		return;
	for (int a = -1; a <= 1; ++a) {
		for (int b = -1; b <= 1; ++b) {
			if ((i + a < 0) || (i + a >= image.rows) || (j + b < 0 )|| (j + b >= image.cols))
				continue;

			double value = image.data[(i + a) * image.cols + j + b];
			if (value >= low_threshold && value < high_threshold) {
				image.data[(i+a) * image.cols + j + b] = 255;
				recursiveHysteresis(image, low_threshold, high_threshold, i + a, j + b, length + 1);
			}
			else if(value < low_threshold) {
				image.data[(i + a) * image.cols + j + b] = 0;
			}
		}
	}
}
double ImageProcessing::OtsuThreshold(const cv::Mat& in_image)
{
	int size = 256, total = in_image.rows * in_image.cols;
	double level = 0;
	//Build the histogram
	double* histogram = new double[size] {0};
	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		histogram[in_image.data[i]]++;
	}
	double sumB = 0;
	double wB = 0;
	double maximum = 0;
	double sum1 = 0;
	for (int i = 0; i < size; ++i) {
		sum1 += (i * histogram[i]);
	}
	for (int i = 1; i < size; ++i) {
		double wF = total - wB;
		if (wB > 0 && wF > 0) {
			double mF = (sum1 = sumB) / wF;
			double val = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF);
			if (val >= maximum) {
				level = i;
				maximum = val;
			}
		}
		wB = wB + histogram[i];
		sumB = sumB + (i - 1) * histogram[i];
	}
	return level;
}

