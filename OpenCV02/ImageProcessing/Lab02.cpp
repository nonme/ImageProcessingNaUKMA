#include "ImageProcessing.h"

void ImageProcessing::GetHistogram(const cv::Mat& in_image, int* histogram)
{
	int height = in_image.rows;
	int width = in_image.cols;
	int histogram_width = 256;
	for (int i = 0; i < histogram_width; ++i) {
		histogram[i] = 0;
	}
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int value = in_image.data[i * width + j];
			++histogram[value];
		}
	}
}

void ImageProcessing::HistogramEqualization(const cv::Mat& in_image, cv::Mat& out_image, cv::Mat& in_histogram, cv::Mat& out_histogram) {
	const int histogram_size = 256;
	int histogram[histogram_size];
	int histogram_o[histogram_size];
	int histogram_c[histogram_size];

	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound(static_cast<double>(hist_w / histogram_size));
	int max_in_hist = 0;
	int max_out_hist = 0;
	for (int i = 0; i < histogram_size; ++i) {
		histogram[i] = 0;
		histogram_c[i] = 0;
		histogram_o[i] = 0;
	}

	for (int i = 0; i < in_image.cols * in_image.rows; ++i) {
		int value = in_image.data[i];
		histogram[value]++;
		max_in_hist = std::max(max_in_hist, histogram[value]);
	}
	histogram_c[0] = histogram[0];
	for (int i = 1; i < histogram_size; ++i) {
		histogram_c[i] = histogram_c[i - 1] + histogram[i];
	}

	double alpha = 255.0 / static_cast<double>(in_image.rows * in_image.cols);
	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		int value = in_image.data[i];
		out_image.data[i] = histogram_c[value] * alpha;

		int new_value = out_image.data[i];
		histogram_o[new_value]++;

		max_out_hist = std::max(max_out_hist, histogram_o[new_value]);
	}
	for (int i = 0; i < histogram_size; ++i) {
		double in_value = static_cast<double>(histogram[i]) / static_cast<double>(max_in_hist);
		double out_value = static_cast<double>(histogram_o[i]) / static_cast<double>(max_out_hist);
		cv::line(in_histogram, cv::Point(bin_w * (i), hist_h),
			cv::Point(bin_w * (i), hist_h - cvRound(hist_h * in_value)),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		cv::line(out_histogram, cv::Point(bin_w * (i), hist_h),
			cv::Point(bin_w * (i), hist_h - cvRound(hist_h * out_value)),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}
}

void ImageProcessing::LogTransformation(const cv::Mat& in_image, cv::Mat& out_image, double c)
{
	double max_f(-1.0);
	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		if (max_f == -1 || in_image.data[i] > max_f)
			max_f = in_image.data[i];
	}
	if (c == -1)
		c = 255.0 / (log2(max_f + 1));
	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		out_image.data[i] = c * log2(in_image.data[i] + 1);
	}
}

void ImageProcessing::GammaCorrection(const cv::Mat& in_image, cv::Mat& out_image, double c, double y) {
	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		out_image.data[i] = 255.0 * c * pow(in_image.data[i] / 255.0, 1.0 / y);
	}
}

void ImageProcessing::PiecewiseLinearTransformation(const cv::Mat& in_image, cv::Mat& out_image, double n)
{
	double fbound = 255 / 3;
	double sbound = 3 * 255 / 4.5;

	for (int i = 0; i < in_image.rows * in_image.cols; ++i) {
		if (in_image.data[i] < fbound) {
			out_image.data[i] = in_image.data[i] / 2.0;
		}
		else if (in_image.data[i] < sbound) {
			out_image.data[i] = in_image.data[i];
		}
		else {
			out_image.data[i] = in_image.data[i] * 2.0;
		}
	}
}