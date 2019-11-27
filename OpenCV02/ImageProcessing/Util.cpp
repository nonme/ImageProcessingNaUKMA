#include "ImageProcessing.h"

double ImageProcessing::shrink(double val, double left, double right) {
	if (val < left)
		return left;
	if (val > right)
		return right;
	return val;
}

void ImageProcessing::arrayToMat(double** arr, cv::Mat& image, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			image.data[i * cols + j] = shrink(arr[i][j], 0, 255);
		}
	}
}

cv::Mat ImageProcessing::arrayToMat(double** arr, int rows, int cols)
{ 
	cv::Mat mat(cv::Size(cols, rows), 0);
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			mat.data[i * cols + j] = shrink(arr[i][j], 0, 255);
		}
	}
	return mat;
}

double ImageProcessing::normalize(double value, double local_value, double max_value)
{
	return value / local_value * max_value;
}

void ImageProcessing::normalize(double** arr, int height, int width, double alpha, double beta)
{
	double max_value = 0;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			max_value = std::max(max_value, arr[i][j]);
		}
	}
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			arr[i][j] = arr[i][j] / max_value * beta;
		}
	}
}

double ImageProcessing::maxInMat(const cv::Mat& image)
{
	uchar max_value = 0;
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			max_value = std::max(max_value, image.data[i * image.cols + j]);
		}
	}
	return max_value;
}

cv::Mat ImageProcessing::subtract(const cv::Mat& a, const cv::Mat& b)
{
	
	cv::Mat result(a.size(), a.type());
	if (a.size() != b.size())
		return result;
	for (int i = 0; i < a.rows; ++i) {
		for (int j = 0; j < a.cols; ++j) {
			result.data[i * a.cols + j] = a.data[i * a.cols + j] - b.data[i * a.cols + j];
		}
	}
	return result;
}

void ImageProcessing::plotCurvature(const std::vector<double>& plot, cv::Mat& out_image)
{
	if (plot.size() == 0)
		return;
	int height = out_image.rows;
	int width = out_image.cols;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			out_image.data[i * width + j] = 255;
		}
	}
	for (int i = 0; i < plot.size() - 1; ++i) {

		cv::Point p1 = cv::Point(i, height / 2 - plot[i] * 100);
		cv::Point p2 = cv::Point(i + 1, height / 2 - plot[i + 1] * 100);

		std::cout << plot[i] << " " << plot[i + 1] << std::endl;

		cv::line(out_image, p1, p2, cv::Scalar(94, 206, 165, 255));
	}
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

ImageProcessing::Pixel ImageProcessing::GetPixel(int i, int j, const cv::Mat& image, Direction head, Direction direction)
{
	switch (head) {
	case NORTH:
		switch (direction) {
		case NORTH: --i; break;
		case NORTH_EAST: --i; ++j; break;
		case EAST: ++j; break;
		case SOUTH_EAST: ++i; ++j; break;
		case SOUTH: ++i; break;
		case SOUTH_WEST: ++i; --j; break;
		case WEST: --j; break;
		case NORTH_WEST: --i; --j; break;
		}
		break;
	case EAST:
		switch (direction) {
		case NORTH: ++j; break;				// ->
		case NORTH_EAST: ++i; ++j; break;
		case EAST: ++i; break;				// down
		case SOUTH_EAST: ++i; --j; break;
		case SOUTH: --j; break;				// <-
		case SOUTH_WEST: --i; --j; break;
		case WEST: --i; break;				// up
		case NORTH_WEST: --i; ++j; break;
		}
		break;
	case SOUTH:
		switch (direction) {
		case NORTH: ++i; break;				// down
		case NORTH_EAST: ++i; --j; break;	// down-left
		case EAST: --j; break;				// left
		case SOUTH_EAST: --i; --j; break;	// upper-left
		case SOUTH: --i; break;				// up
		case SOUTH_WEST: --i; ++j; break;	// upper-right
		case WEST: ++j; break;				// right
		case NORTH_WEST: ++i; ++j; break;	// down-right
		}
		break;
	case WEST: 
		switch (direction) {
		case NORTH: --j; break;				// <-
		case NORTH_EAST: --i; --j; break;
		case EAST: --i; break;				// up
		case SOUTH_EAST: --i; ++j; break;
		case SOUTH: ++j; break;				// ->
		case SOUTH_WEST: ++i;  ++j; break;
		case WEST: ++i; break;				// down
		case NORTH_WEST: ++i; --j; break;
		}
		break;
	}
	if (i >= 0 && i < image.rows && j >= 0 && j < image.cols)
		return Pixel(i, j, image.data[i * image.cols + j]);
	else
		return Pixel(i, j, 0);
}

unsigned char ImageProcessing::GetPixel(int i, int j, const cv::Mat& image)
{
	return image.data[i * image.cols + j];
}

ImageProcessing::Direction ImageProcessing::turnRight(Direction direction)
{
	switch (direction) {
	case NORTH:
		return EAST;
	case EAST:
		return SOUTH;
	case SOUTH:
		return WEST;
	case WEST:
		return NORTH;
	}
}
ImageProcessing::Direction ImageProcessing::turnLeft(Direction direction)
{
	switch (direction) {
	case NORTH:
		return WEST;
	case EAST:
		return NORTH;
	case SOUTH:
		return EAST;
	case WEST:
		return SOUTH;
	}
}

void ImageProcessing::drawContours(cv::Mat& image, std::vector<std::vector<cv::Point>> contours)
{
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j)
			image.data[i * image.cols + j] = 0;
	}
	for (int i = 0; i < contours.size(); ++i) {
		for (int j = 0; j < contours[i].size(); ++j) {
			image.data[contours[i][j].y * image.cols + contours[i][j].x] = 255;
		}
	}
}

double ImageProcessing::sumArray(double* arr, int size)
{
	double sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += arr[i];
	}
	return sum;
}

double ImageProcessing::sumArray(int* arr, int size)
{
	double sum = 0;
	for (int i = 0; i < size; ++i) {
		sum += arr[i];
	}
	return sum;
}

double ImageProcessing::meanBrightness(int* histogram, int size)
{
	double sum = 0;
	int number_of_pixels = 0;
	for (int i = 0; i < size; ++i) {
		sum += histogram[i] * i;
		number_of_pixels += histogram[i];
	}
	return sum / number_of_pixels;
}
