#include "ImageProcessing.h"
#include <unordered_map>

void ImageProcessing::RegionGrowing(const cv::Mat& in_image, cv::Mat& out_image, int seed, int threshold, int connectivity)
{
	if (threshold == -1)
		threshold = OptimalThreshold(in_image);

	int height = in_image.rows;
	int width = in_image.cols;

	std::queue<Point> points;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			out_image.data[i * width + j] = 0;
			if (in_image.data[i * width + j] == seed) {
				points.push(Point(j, i));
				out_image.data[i * width + j] = 255;
			}
		}
	}
	while (!points.empty()) {
		Point top = points.front();
		points.pop();

		int step = (connectivity == FOUR_CONNECTED ? 2 : 1);
		for (int i = -1; i <= 1; i+=step) {
			for (int j = -1; j <= 1; j+=step) {
				if (top.x + j < 0 || top.x + j >= width || top.y + i < 0 || top.y + i >= height)
					continue;
				int index = (top.y + i) * width + top.x + j;
				if (out_image.data[index] != 255 && in_image.data[index] >= threshold) {
					out_image.data[index] = 255;
					points.push(Point(top.x + j, top.y + i));
				}
			}
		}
	}
}
void ImageProcessing::RegionMarking(const cv::Mat& in_image, cv::Mat& out_image, int threshold, int connectivity)
{
	if (threshold == -1)
		threshold = 10;

	int height = in_image.rows;
	int width = in_image.cols;

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j)
			out_image.data[i * width + j] = 0;
	}
	std::queue<Point> points;
	int color = 30;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (out_image.data[i * width + j] == 0 && in_image.data[i * width + j] != 0) {
				points.push(Point(j, i));
				out_image.data[i * width + j] = color;
				while (!points.empty()) {
					Point current = points.front();
					points.pop();

					for (int a = -1; a <= 1; ++a) {
						for (int b = -1; b <= 1; ++b) {
							if (current.y + a < 0 || current.y + a >= in_image.rows || current.x + b < 0 || current.x + b >= in_image.cols)
								continue;
							int index = (current.y + a) * in_image.cols + current.x + b;
							if (out_image.data[index] == 0 && abs(in_image.data[index] - in_image.data[i * width + j]) <= threshold) {
								out_image.data[index] = color;
								points.push(Point(current.x + b, current.y + a));
							}
						}
					}
				}
				color++;
			}
		}
	}
}

void ImageProcessing::markRegion(const cv::Mat& in_image, cv::Mat& out_image, int i, int j, int start, int threshold, int color, int connectivity)
{
	out_image.data[i * in_image.cols + j] = color;

	int step = (connectivity == FOUR_CONNECTED ? 2 : 1);
	for (int a = -1; a <= 1; a += step) {
		for (int b = -1; b <= 1; b += step) {
			if (i + a < 0 || i + a >= in_image.rows || j + b < 0 || j + b >= in_image.cols)
				continue;
			int index = (i + a) * in_image.cols + j + b;
			if (out_image.data[index] == 0 && abs(in_image.data[index]-start) <= threshold) {
				markRegion(in_image, out_image, i + a, j + b, start, threshold, color);
			}
		}
	}
}
