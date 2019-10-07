#include "ImageProcessing.h"

void ImageProcessing::decrease(const cv::Mat in_image, cv::Mat& out_image, int n) {
	for (int i = 0; i < in_image.rows; i += n) {
		for (int j = 0; j < in_image.cols; j += n) {
			int res = 0, c = 0;
			for (int k = 0; k < n; ++k) {
				for (int l = 0; l < n; ++l) {
					if (i + k < in_image.rows && j + l < in_image.cols) {
						res += in_image.data[(i + k) * in_image.cols + j + l];
						c++;
					}
				}
			}
			if (i / n < out_image.rows && j / n < out_image.cols)
				out_image.data[(i / n) * out_image.cols + j / n] = res / c;
		}
	}
}
void ImageProcessing::increase(const cv::Mat in_image, cv::Mat& out_image, int n) {
	for (int i = 0; i < in_image.rows; ++i) {
		for (int j = 0; j < in_image.cols; ++j) {
			for (int k = 0; k < n; ++k) {
				for (int l = 0; l < n; ++l) {
					out_image.data[(i * n + k) * out_image.cols + j * n + l] = in_image.data[i * in_image.cols + j];
				}
			}
		}
	}
}
void ImageProcessing::invertImage(cv::Mat& image) {
	cv::Mat bin_image = cv::Mat::zeros(image.size(), image.type());
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (image.data[i * image.cols + j] > 70)
				bin_image.data[i * image.cols + j] = 255;
			else
				bin_image.data[i * image.cols + j] = 0;
		}
	}
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (image.data[i * image.cols + j] > 20)
				image.data[i * image.cols + j] = bin_image.data[i * bin_image.cols + j] - image.data[i * image.cols + j];
		}
	}
}
