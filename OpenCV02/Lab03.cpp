#include "ImageProcessing.h"

void ImageProcessing::convolution(const cv::Mat& in_image, cv::Mat& out_image, double mask[], int ksize, double coef) {
	double sum(0.0);
	for (int i = ksize / 2; i < in_image.rows - ksize / 2; ++i) {
		for (int j = ksize / 2; j < in_image.cols - ksize / 2; ++j) {
			sum = 0.0;
			for (int a = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b) {
					sum += (in_image.data[(i + a - ksize / 2) * in_image.cols + (j + b - ksize / 2)]) * mask[a * ksize + b];
				}
			}
			out_image.data[i * in_image.cols + j] = sum * coef;
		}
	}
}
void ImageProcessing::GaussianBlur(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double sigma) {
	/* Create mask array */
	double* mask = new double[static_cast<long>(ksize) * static_cast<long>(ksize)];
	double coef(0.0);

	/* Fill array with formula Wg(x,y) = 1/(2*PI*sigma^2) * e^(-(x^2 + y^2)/(2sigma^2)) */
	for (int i = 0; i < ksize * ksize; ++i) {
		double x = i % ksize;
		double y = i / ksize;

		double denominator = 2 * ImageProcessing::PI * pow(sigma, 2.0);
		double power = -(x * x + y * y) / (2 * pow(sigma, 2.0));
		mask[i] = 1.0 / denominator * pow(ImageProcessing::E, power);

		coef += mask[i];
	}
	convolution(in_image, out_image, mask, ksize, 1.0 / coef);
}