#include "ImageProcessing.h"

void ImageProcessing::toBinary(const cv::Mat& in_image, cv::Mat& out_image, int threshold)
{
	int height = in_image.rows;
	int width =  in_image.cols;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (in_image.data[i * width + j] >= threshold)
				out_image.data[i * width + j] = 255;
			else
				out_image.data[i * width + j] = 0;
		}
	}
}

void ImageProcessing::toBinary(cv::Mat& image, int threshold)
{
	int height = image.rows;
	int width =  image.cols;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (image.data[i * width + j] >= threshold)
				image.data[i * width + j] = 255;
			else
				image.data[i * width + j] = 0;
		}
	}
}

void ImageProcessing::SemiThresholding(const cv::Mat& in_image, cv::Mat& out_image, int high_threshold, int low_threshold)
{
	int height = in_image.rows;
	int width = in_image.cols;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (in_image.data[i * width + j] >= high_threshold)
				out_image.data[i * width + j] = 150;
			else if (in_image.data[i * width + j] <= low_threshold)
				out_image.data[i * width + j] = 255;
			else
				out_image.data[i * width + j] = in_image.data[i * width + j];
		}
	}
}

int ImageProcessing::TriangleAlgorithm(const cv::Mat& in_image)
{
	int histogram_width = 256;
	int* histogram = new int[histogram_width];
	GetHistogram(in_image, histogram);

	//Find minimum brightness and maximum brightness
	int b_min = 255, b_max = 0;
	int index_min = 0, index_max = 0;
	for (int i = 0; i < histogram_width; ++i) {
		if (histogram[i] > b_max) {
			b_max = histogram[i];
			index_max = i;
		}
		if (histogram[i] < b_min && histogram[i] > 0) {
			b_min = histogram[i];
			index_min = i;
		}
	}
	//Find the maximum distance between pixels
	Line line(index_min, b_min, index_max, b_max);
	if (index_min > index_max)
		std::swap(index_min, index_max);
	double min_d = line.distance(index_min, histogram[index_min]);
	int threshold = index_min;
	for (int i = index_min + 1; i <= index_max; ++i) {
		double d = line.distance(i, histogram[i]);
		if (d < min_d) {
			min_d = d;
			threshold = d;
		}
	}
	return threshold;
}

int ImageProcessing::OptimalThreshold(const cv::Mat& in_image)
{
	int height = in_image.rows;
	int width = in_image.cols;
	int hist_size = 256;

	int* histogram = new int[hist_size];
	GetHistogram(in_image, histogram);

	int T = 0;
	int Tk = meanBrightness(histogram);

	double* sumArray = new double[hist_size];
	double* brightArray = new double[hist_size];
	sumArray[0] = histogram[0];
	brightArray[0] = 0;
	for (int i = 1; i < hist_size; ++i) {
		sumArray[i] = sumArray[i - 1] + histogram[i];
		brightArray[i] = brightArray[i - 1] + i * histogram[i];
	}

	while (std::abs(T - Tk) >= 1) {
		T = Tk;

		int u1 = brightArray[T] / sumArray[T];
		int u2 = (brightArray[hist_size - 1] - brightArray[T]) / (sumArray[hist_size - 1] - sumArray[T]);

		Tk = (u1 + u2) / 2;
	}
	return T;
}
