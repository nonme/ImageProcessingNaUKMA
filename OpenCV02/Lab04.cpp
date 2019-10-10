#include "ImageProcessing.h"
#include <algorithm>

void ImageProcessing::SobelOperator(const cv::Mat& in_image, cv::Mat& out_image) {
	//int n = 9;
	////We use separability
	//double *g_x_vertical = new double[3];
	//g_x_vertical[0] = 1;
	//g_x_vertical[1] = 2;
	//g_x_vertical[2] = 1;
	//double* g_x_horizontal = new double[3];
	//g_x_horizontal[0] = 1;
	//g_x_horizontal[1] = 0;
	//g_x_horizontal[2] = -1;
	//double* g_y_vertical = new double[3];
	//g_y_vertical[0] = 1;
	//g_y_vertical[1] = 0;
	//g_y_vertical[2] = -1;
	//double* g_y_horizontal = new double[3];
	//g_y_horizontal[0] = 1;
	//g_y_horizontal[1] = 2;
	//g_y_horizontal[2] = 1;
	//We don't' use separability
	double* g_y = new double[9];
	g_y[0] = 1; g_y[1] = 2; g_y[2] = 1;
	g_y[3] = 0; g_y[4] = 0; g_y[5] = 0;
	g_y[6] = -1; g_y[7] = -2; g_y[8] = -1;
	double* g_x = new double[9];
	g_x[0] = -1; g_x[1] = 0; g_x[2] = 1;
	g_x[3] = -2; g_x[4] = 0; g_x[5] = 2;
	g_x[6] = -1; g_x[7] = 0; g_x[8] = 1;

	
	cv::Mat y_image(in_image.size(), in_image.type());
	cv::Mat x_image(in_image.size(), in_image.type());

	//convolution(in_image, y_image, g_y, 3, 1.0/8.0);
	for (int i = 1; i < in_image.rows - 1; ++i) {
		for (int j = 1; j < in_image.cols - 1; ++j) {
			double sum = in_image.data[(i + 1) * in_image.cols + j - 1] +
				2 * in_image.data[(i + 1) * in_image.cols + j] + in_image.data[(i + 1) * in_image.cols + j + 1];
			double minus = in_image.data[(i - 1) * in_image.cols + j - 1] +
				2 * in_image.data[(i - 1) * in_image.cols + j] + in_image.data[(i - 1) * in_image.cols + j + 1];

			y_image.data[i * in_image.cols + j] = (sum - minus) / 40.0;

			if (y_image.data[i * in_image.cols + j] < 0)
				y_image.data[i * in_image.cols + j] = 0;
		}
	}
	cv::imshow("y image", y_image);

	//convolution(in_image, x_image, g_x, 3, 1.0/4.0);
	for (int i = 1; i < in_image.rows - 1; ++i) {
		for (int j = 1; j < in_image.cols - 1; ++j) {
			double sum = in_image.data[(i - 1) * in_image.cols + j + 1] +
				2 * in_image.data[(i) * in_image.cols + j + 1] + in_image.data[(i + 1) * in_image.cols + j + 1];
			double minus = in_image.data[(i - 1) * in_image.cols + j - 1] +
				2 * in_image.data[(i) * in_image.cols + j - 1] + in_image.data[(i + 1) * in_image.cols + j - 1];

			if (x_image.data[i * in_image.cols + j] < 0)
				x_image.data[i * in_image.cols + j] = 0;
			x_image.data[i * in_image.cols + j] = (sum - minus) / 40.0;
			if (x_image.data[i * in_image.cols + j] < 0)
				x_image.data[i * in_image.cols + j] = 0;
		}
	}
	cv::imshow("x image", x_image);
	for (int i = 0; i < out_image.rows; ++i) {
		for (int j = 0; j < out_image.cols; ++j) {
			double g_y_val = y_image.data[i * in_image.cols + j];
			//std::cout << g_y_val << " ";
			double g_x_val = x_image.data[i * in_image.cols + j];
			double g = sqrt(g_y_val * g_y_val + g_x_val * g_x_val);
			out_image.data[i * out_image.cols + j] = g;
		}
	}
}