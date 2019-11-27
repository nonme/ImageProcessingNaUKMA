#include "ImageProcessing.h"
#include <algorithm>

double** ImageProcessing::SobelOperator(const cv::Mat& in_image, cv::Mat& out_image) {
	int height = in_image.rows;
	int width = in_image.cols;

	double kernelY[] {1, 2, 1, 0, 0, 0, -1, -2, -1};
	double kernelX[] {-1, 0, 1, -2, 0, 2, -1, 0, 1 };
	double** angles = new2DArray(height, width);

	double** Gy = new2DArray(height, width);
	double** Gx = new2DArray(height, width);

	convolution(in_image, Gy, kernelY, 3, 1.0 / 4.0);
	convolution(in_image, Gx, kernelX, 3, 1.0 / 4.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double G = sqrt(Gx[i][j] * Gx[i][j] + Gy[i][j] * Gy[i][j]);
			out_image.data[i * width + j] = shrink(G, 0, 255);
			angles[i][j] = atan2(Gy[i][j], Gx[i][j]);
		}
	}

	delete2DArray(Gy, height);
	delete2DArray(Gx, height);

	return angles;
}

void ImageProcessing::SobelOperator(const cv::Mat& in_image, double** g, double** gx, double** gy)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double kernelY[]{ 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	double kernelX[]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };

	convolution(in_image, gy, kernelY, 3, 1.0 / 4.0);
	convolution(in_image, gx, kernelX, 3, 1.0 / 4.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double G = sqrt(gx[i][j] * gx[i][j] + gy[i][j] * gy[i][j]);
			g[i][j] = shrink(G, 0, 255);
		}
	}
}

double** ImageProcessing::ScharrOperator(const cv::Mat& in_image, cv::Mat& out_image)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double kernelY[9] { 47, 162, 47, 0, 0, 0, -47, -162, -47 };
	double kernelX[9] { -47, 0, 47, -162, 0, 162, -47, 0, 47 };

	double** angles = new2DArray(height, width);
	double** Gy = new2DArray(height, width);
	double** Gx = new2DArray(height, width);

	convolution(in_image, Gy, kernelY, 3, 1.0 / 250.0);
	convolution(in_image, Gx, kernelX, 3, 1.0 / 250.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double G = sqrt(Gx[i][j] * Gx[i][j] + Gy[i][j] * Gy[i][j]);
			out_image.data[i * width + j] = shrink(G, 0, 255);

			angles[i][j] = atan2(Gy[i][j], Gx[i][j]);
		}
	}
	delete2DArray(Gy, height);
	delete2DArray(Gx, height);

	return angles;
}


void ImageProcessing::LaplaceOperator(const cv::Mat& in_image, cv::Mat& out_image)
{
	int height = in_image.rows;
	int width = in_image.cols;
	double** result = new2DArray(height, width);

	LaplaceOperator(in_image, result);
	arrayToMat(result, out_image, height, width);
	delete2DArray(result, height);
}

void ImageProcessing::LaplaceOperator(const cv::Mat& in_image, double** result)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double kernel[9]{ 0, 1, 0, 1, -4, 1, 0, 1, 0 };
	convolution(in_image, result, kernel, 3, 1.0);
}

void ImageProcessing::ZeroCrossOperator(const cv::Mat& in_image, cv::Mat& out_image, int threshold)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double** laplace = new2DArray(height, width);
	LaplaceOperator(in_image, laplace);
	
	for (int i = 1; i < height - 1; ++i) {
		for (int j = 1; j < width - 1; ++j) {
			out_image.data[i * width + j] = 0;
			if (laplace[i + 1][j] * laplace[i - 1][j] < 0 && abs(laplace[i + 1][j]) + abs(laplace[i-1][j]) > threshold) {
				out_image.data[i * width + j] = 255;
			}
			else if (laplace[i][j + 1] * laplace[i][j - 1] < 0 && abs(laplace[i][j + 1]) + abs(laplace[i][j - 1]) > threshold) {
				out_image.data[i * width + j] = 255;
			}
			else if (laplace[i + 1][j + 1] * laplace[i - 1][j - 1] < 0 && abs(laplace[i + 1][j + 1]) + abs(laplace[i - 1][j - 1]) > threshold) {
				out_image.data[i * width + j] = 255;
			}
			else if (laplace[i + 1][j - 1] * laplace[i - 1][j + 1] < 0 && abs(laplace[i + 1][j - 1]) + abs(laplace[i - 1][j + 1]) > threshold) {
				out_image.data[i * width + j] = 255;
			}
		}
	}
}

void ImageProcessing::RobertsOperator(const cv::Mat& in_image, cv::Mat& out_image)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double kernelX[] { 1, 0, 0, -1 };
	double kernelY[] { 0, 1, -1, 0 };

	double** Gx = new2DArray(height, width);
	double** Gy = new2DArray(height, width);

	convolution(in_image, Gx, kernelX, 2, 1.0);
	convolution(in_image, Gy, kernelY, 2, 1.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double G = sqrt(Gx[i][j] * Gx[i][j] + Gy[i][j] * Gy[i][j]);
			out_image.data[i * width + j] = shrink(G, 0, 255);
		}
	}

	delete2DArray(Gx, height);
	delete2DArray(Gy, height);
}

void ImageProcessing::PrewittOperator(const cv::Mat& in_image, cv::Mat& out_image)
{
	int height = in_image.rows;
	int width = in_image.cols;

	double kernelY[9]{ 1, 1, 1, 0, 0, 0, -1, -1, -1 };
	double kernelX[9]{ 1, 0, -1, 1, 0, -1, 1, 0, -1 };

	double** Gy = new2DArray(height, width);
	double** Gx = new2DArray(height, width);

	convolution(in_image, Gy, kernelY, 3, 1.0 / 3.0);
	convolution(in_image, Gx, kernelX, 3, 1.0 / 3.0);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double G = sqrt(Gx[i][j] * Gx[i][j] + Gy[i][j] * Gy[i][j]);
			out_image.data[i * width + j] = shrink(G, 0, 255);
		}
	}
	delete2DArray(Gy, height);
	delete2DArray(Gx, height);
}

void ImageProcessing::apply(const cv::Mat& in_image, cv::Mat& applied_image, cv::Mat& out_image, double coef)
{
	int height = in_image.rows;
	int width = in_image.cols;

	assert(in_image.size() == applied_image.size());
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double value = coef*static_cast<double>(in_image.data[i * width + j]) + static_cast<double>(applied_image.data[i * width + j]);
			out_image.data[i * width + j] = shrink(value, 0, 255);
		}
	}
}
