#include "ImageProcessing.h"

double ImageProcessing::shrink(double val, double left, double right) {
	if (val < left)
		return left;
	if (val > right)
		return right;
	return val;
}

void ImageProcessing::arrayToMat(double** arr, cv::Mat& image, int rows, int cols) {
	assert(image.rows == rows && image.cols == cols);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			image.data[i * cols + j] = shrink(arr[i][j], 0, 255);
		}
	}
}

double ImageProcessing::normalize(double value, double local_value, double max_value)
{
	return value / local_value * max_value;
}

void ImageProcessing::delete2DArray(double** arr, int rows)
{
	for (int i = 0; i < rows; ++i)
		delete[] arr[i];
	delete[] arr;
}

double** ImageProcessing::new2DArray(int height, int width, int fillValue)
{
	double** arr = new double*[height];
	for (int i = 0; i < height; ++i) {
		arr[i] = new double[width];
		if (fillValue != -1) {
			for (int j = 0; j < width; ++j) {
				arr[i][j] = fillValue;
			}
		}
	}
	return arr;
}
