#include "ImageProcessing.h"

void ImageProcessing::convolution(const cv::Mat& in_image, cv::Mat& out_image, double mask[], int ksize, double coef) {
	double sum(0.0);
	for (int i = ksize / 2; i < in_image.rows - ksize / 2; ++i) {
		for (int j = ksize / 2; j < in_image.cols - ksize / 2; ++j) {
			sum = 0.0;
			for (int a = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b) {
					if (i + a - ksize / 2 >= 0 && j + b - ksize / 2 >= 0)
						sum += (in_image.data[(i + a - ksize / 2) * in_image.cols + (j + b - ksize / 2)]) * mask[a * ksize + b];
				}
			}
			sum *= coef;
			//sum = shrink(sum, 0, 255);
			sum = abs(sum);
			out_image.data[i * in_image.cols + j] = sum;
		}
	}
}

void ImageProcessing::convolution(const cv::Mat& in_image, double** out_image, double mask[], int ksize, double coef)
{
	double sum(0.0);
	for (int i = 0; i < in_image.rows; ++i) {
		for (int j = 0; j < in_image.cols; ++j) {
			sum = 0.0;
			for (int a = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b) {
					if (i + a - ksize / 2 >= 0 && i + a - ksize / 2 < in_image.rows && j + b - ksize / 2 < in_image.cols && j + b - ksize / 2 >= 0)
						sum += (in_image.data[(i + a - ksize / 2) * in_image.cols + (j + b - ksize / 2)]) * mask[a * ksize + b];
				}
			}
			sum *= coef;
			out_image[i][j] = sum;
		}
	}
}

void ImageProcessing::convolution(double** image, double** out_image, int height, int width, double mask[], int ksize, double coef)
{
	double sum(0.0);
	for (int i = ksize / 2; i < height - ksize / 2; ++i) {
		for (int j = ksize / 2; j < width - ksize / 2; ++j) {
			sum = 0.0;
			for (int a = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b) {
					if (i + a - ksize / 2 >= 0 && j + b - ksize / 2 >= 0)
						sum += (image[(i + a - ksize / 2)][(j + b - ksize / 2)]) * mask[a * ksize + b];
				}
			}
			sum *= coef;
			out_image[i][j] = sum;
		}
	}
}

void ImageProcessing::convolution(double** image, int height, int width, double mask[], int ksize, double coef)
{
	double** tempArray = new2DArray(height, width);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			tempArray[i][j] = image[i][j];
		}
	}
	double sum(0.0);
	for (int i = ksize / 2; i < height - ksize / 2; ++i) {
		for (int j = ksize / 2; j < width - ksize / 2; ++j) {
			sum = 0.0;
			for (int a = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b) {
					if (i + a - ksize / 2 >= 0 && j + b - ksize / 2 >= 0)
						sum += (tempArray[(i + a - ksize / 2)][(j + b - ksize / 2)]) * mask[a * ksize + b];
				}
			}
			sum *= coef;
			image[i][j] = sum;
		}
	}
	delete2DArray(tempArray, height);
}

void ImageProcessing::GaussianBlur(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double sigma) {
	/* Create mask array */
	double* mask = new double[static_cast<int> (ksize * ksize)];
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

void ImageProcessing::GaussianBlur(double** image, int height, int width, int ksize, double sigma)
{
	/* Create mask array */
	double* mask = new double[static_cast<int> (ksize * ksize)];
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
	double** tempArray = new2DArray(height, width);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			tempArray[i][j] = image[i][j];
		}
	}
	convolution(tempArray, image, height, width, mask, ksize, 1.0 / coef);
}

void ImageProcessing::LowpassFilter(const cv::Mat& in_image, cv::Mat& out_image, int ksize)
{
	double* mask = new double[static_cast<int>(ksize * ksize)];
	for (int i = 0; i < ksize * ksize; ++i)
		mask[i] = 1;
	convolution(in_image, out_image, mask, ksize, 1.0 / (ksize * ksize));
}

/*
	O(N) = N^2 * ksize^2 * log(ksize)
*/
void ImageProcessing::MedianFilter(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double quantile)
{
	for (int i = ksize / 2; i < in_image.rows - ksize / 2; ++i) {
		for (int j = ksize / 2; j < in_image.cols - ksize / 2; ++j) {
			double* medianArray = new double[static_cast<long long>(ksize) * static_cast<long long>(ksize)];
			for (int a = 0, k = 0; a < ksize; ++a) {
				for (int b = 0; b < ksize; ++b, ++k) {
					medianArray[k] = static_cast<double>(in_image.data[(i + a - ksize / 2) * in_image.cols + (j + b - ksize / 2)]);
				}
			}
			std::sort(medianArray, medianArray + ksize * ksize);
			out_image.data[i * in_image.cols + j] = medianArray[static_cast<int> (ksize * ksize * quantile)];
		}
	}
}

void ImageProcessing::SaltAndPepperNoise(const cv::Mat& in_image, cv::Mat& out_image, int noise_percent)
{
	for (int i = 0; i < in_image.rows; ++i) {
		for (int j = 0; j < in_image.cols; ++j) {
			int noise_check = rand() % noise_percent + 1;
			
			if (noise_check == noise_percent) {
				int noise_value = rand() % 256;
				out_image.data[i * out_image.cols + j] = noise_value;
			}
			else {
				out_image.data[i * out_image.cols + j] = in_image.data[i * out_image.cols + j];
			}
		}
	}
}

