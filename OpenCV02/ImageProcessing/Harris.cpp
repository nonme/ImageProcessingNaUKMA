#include "ImageProcessing.h"

std::vector<cv::Point2f> ImageProcessing::HarrisCornerDetector(const cv::Mat& image, cv::Mat& out_image, const double k, int filterRange, int threshold, bool applyGauss)
{
	int height = image.rows;
	int width = image.cols;

	Derivative derivative(height, width);
	SobelOperator(image, derivative.g, derivative.gx, derivative.gy);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			derivative.g[i][j] = derivative.gx[i][j] * derivative.gy[i][j];
			derivative.gx[i][j] *= derivative.gx[i][j];
			derivative.gy[i][j] *= derivative.gy[i][j];
		}
	}
	applyGaussianToDerivative(derivative, 5);
	cv::imshow("Sobel", ImageProcessing::arrayToMat(derivative.g, height, width));
	cv::imshow("SobelX", ImageProcessing::arrayToMat(derivative.gx, height, width));
	cv::imshow("SobelY", ImageProcessing::arrayToMat(derivative.gy, height, width));

	double** responses = new2DArray(height, width);
	computeHarrisResponses(k, derivative, responses);
	normalize(responses, height, width, 0, 255);
	/*for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			std::cout << responses[i][j] << " ";
		}
	}*/
	image.copyTo(out_image);
	findLocalMaxima(responses, out_image, threshold, 5); // 140 for checkergoard
	return std::vector<cv::Point2f>();
}

void ImageProcessing::computeHarrisResponses(double k, const Derivative& d, double** responses)
{
	int height = d.height;
	int width = d.width;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			double a11, a12;
			double a21, a22;

			a11 = d.gx[i][j];
			a12 = d.g[i][j];
			a21 = d.g[i][j];
			a22 = d.gy[i][j];

			double det = a11 * a22 - a12 * a21;
			double trace = a11 + a22;

			responses[i][j] = abs(det - k * trace * trace);
			//std::cout << responses[i][j] << " ";
			//std::cout << responses[i][j] << " ";
		}
	}
}

void ImageProcessing::applyGaussianToDerivative(Derivative& d, int filterRange)
{
	if (filterRange == 0)
		return;
	GaussianBlur(d.gx, d.height, d.width, filterRange);
	GaussianBlur(d.gy, d.height, d.width, filterRange);
	GaussianBlur(d.g, d.height, d.width, filterRange);
}

void ImageProcessing::findLocalMaxima(double** responses, cv::Mat& image, double percentage, int radius)
{
	int height = image.rows;
	int width = image.cols;
	std::vector<PointData> points(height * width);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			points[i * width + j].value = responses[i][j];
			points[i * width + j].pos = cv::Point(j, i);
		}
	}

	sort(points.begin(), points.end(), PointData::byHarisResponses);
	
	int i = 0;
	while(points[i++].value >= percentage && i < points.size()) {
		for (int a = -radius / 2; a <= radius / 2; ++a) {
			for (int b = -radius / 2; b <= radius / 2; ++b) {
				cv::Point& pos = points[i].pos;
				if (pos.y + a < 0 || pos.y + a >= height || pos.x + b < 0 || pos.x + b >= width)
					continue;
				image.data[(pos.y + a) * width + pos.x + b] = 150;
			}
		}
	}
}
