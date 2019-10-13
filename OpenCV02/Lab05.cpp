#include "ImageProcessing.h"

void ImageProcessing::CannyEdgeDetection(const cv::Mat& in_image, cv::Mat& out_image) {
	int height = in_image.rows;
	int width = in_image.cols;
	/*
		First step is blurring the image using Gaussian Filter to remove the obvious noise. We use
		sigma = 1.4 and 5x5 matrix because big values of ksize and sigma will negatively affect
		the performance of the detector. This is the best size for most of the cases, however
		we can improve this part as shown in CannyImprovedEdgeDetection() method.
	*/
	GaussianBlur(in_image, out_image, 5, 1.4);
	cv::imshow("After 1 step", out_image);
	/*
		Now we need to calculate the intensity gradient of the image.
		We will use 3x3 Sobel filter as the most commonly chosen, however
		other filters, such as Scharr, Prewitt, Cross or 5x5 Sobel might
		improve the quality of edge detection.
	*/
	double** gr_d = SobelOperator(out_image, out_image);
	cv::imshow("After 2 step", out_image);
	/*
		Thirdly, we need to do non-maximum suppression to suppress all the gradient values except
		the local maxima, which indicate locations with the sharpest change of intensity value.

		Wa take our center pixel at (i, j) coordinate and compare it with the pixels in positive
		and negative gradient directions. For example, if gradient direction is 45, we compare it
		with pixels at (i+1, j-1) and (i-1, j+1). If the center pixel is lesser, we suppress it.

		As a result, we get binary image with thin edges.
	*/
	double pi8 = PI / 8;
	for (int i = 1; i < height - 1; ++i) {
		for (int j = 1; j < width - 1; ++j) {
			double center_value = out_image.data[i * width + j];
			double l_value = 255;
			double r_value = 255;
			if (gr_d[i][j] <= pi8 || gr_d[i][j] >= PI - pi8) {
				l_value = out_image.data[i * width + j + 1];
				r_value = out_image.data[i * width + j - 1];
			}
			else if (gr_d[i][j] >= PI / 4 - pi8 && gr_d[i][j] <= PI / 4 + pi8) {
				l_value = out_image.data[(i - 1) * width + j + 1];
				r_value = out_image.data[(i + 1) * width + j - 1];
			}
			else if (gr_d[i][j] >= PI / 2 - pi8 && gr_d[i][j] <= PI / 2 + pi8) {
				l_value = out_image.data[(i+1) * width + j];
				r_value = out_image.data[(i-1) * width + j];
			}
			else {
				l_value = out_image.data[(i - 1) * width + j - 1];
				r_value = out_image.data[(i + 1) * width + j + 1];
			}
			if (center_value < l_value || center_value < r_value)
				out_image.data[i * width + j] = 0;
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
	double high_threshold = 60;
	double low_threshold = 40;
	std::cout << high_threshold << std::endl;
	double** mark = new double* [in_image.rows];
	for (int i = 0; i < in_image.rows; ++i) {
		mark[i] = new double[in_image.cols];
		for (int j = 0; j < in_image.cols; ++j) {
			if (out_image.data[i * in_image.cols + j] < low_threshold)
				mark[i][j] = 0; //Suppress.
			else if (out_image.data[i * in_image.cols + j] >= high_threshold)
				mark[i][j] = 2; //Strong edge
			else {
				mark[i][j] = 1; //Weak edge
				out_image.data[i * width + j] = 50;
			}
		}
	}
	cv::imshow("After 4 step", out_image);
	/*
		Edge tracking by hysteresis - remove such weak edges that doesn't have strong edge near them.
	*/
	for (int i = 1; i < in_image.rows - 1; ++i) {
		for (int j = 1; j < in_image.cols - 1; ++j) {
			if (mark[i][j] == 2)
				out_image.data[i * width + j] = 255;
			if (mark[i][j] == 0)
				out_image.data[i * in_image.cols + j] = 0;
			else {
				bool has_strong = false;
				for (int a = -1; a <= 1; ++a) {
					for (int b = -1; b <= 1; ++b) {
						if (mark[i + a][j + b] == 2) {
							has_strong = true;
						}
					}
				}
				if (!has_strong)
					out_image.data[i * in_image.cols + j] = 0;
				else {
					mark[i][j] = 2;
					out_image.data[i * width + j] = 255;
				}
			}
		}
	}
}

double ImageProcessing::OtsuThreshold(const cv::Mat& in_image)
{
	int size = 256, total = in_image.rows * in_image.cols;
	double level = 0;
	//Build the histogram
	double* histogram = new double[size] {};
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

