#include "ImageProcessing.h"
#include <algorithm>

double** ImageProcessing::SobelOperator(const cv::Mat& in_image, cv::Mat& out_image) {
	double* g_y = new double[9] {1, 2, 1, 0, 0, 0, -1, -2, -1};
	double* g_x = new double[9] {-1, 0, 1, -2, 0, 2, -1, 0, 1 };
	double** g_d = new double*[in_image.rows];
	for (int i = 0; i < in_image.rows; ++i)
		g_d[i] = new double[in_image.cols];

	cv::Mat y_image(in_image.size(), in_image.type());
	cv::Mat x_image(in_image.size(), in_image.type());

	convolution(in_image, y_image, g_y, 3, 1.0);
	convolution(in_image, x_image, g_x, 3, 1.0);
	//cv::imshow("y image", y_image);
	//cv::imshow("x image", x_image);

	double g_max = 0;
	for (int i = 0; i < out_image.rows; ++i) {
		for (int j = 0; j < out_image.cols; ++j) {
			double g_y_val = y_image.data[i * in_image.cols + j];
			double g_x_val = x_image.data[i * in_image.cols + j];

			double g = sqrt(g_y_val * g_y_val + g_x_val * g_x_val);
			g = shrink(g, 0, 255);
			g_max = std::max(g, g_max);

			g_d[i][j] = atan2(g_y_val, g_x_val);
		}
	}
	for (int i = 0; i < out_image.rows; ++i) {
		for (int j = 0; j < out_image.cols; ++j) {
			double g_y_val = y_image.data[i * in_image.cols + j];
			double g_x_val = x_image.data[i * in_image.cols + j];

			double g = sqrt(g_y_val * g_y_val + g_x_val * g_x_val);

			out_image.data[i * out_image.cols + j] = (g / g_max) * 255;
		}
	}
	return g_d;
}
