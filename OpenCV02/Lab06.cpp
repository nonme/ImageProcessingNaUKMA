#include "ImageProcessing.h"

void ImageProcessing::erosion(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w, int type)
{
	int height = in_image.rows;
	int width = in_image.cols;
	switch (type) {
	case BINARY:
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				out_image.data[i * width + j] = 0;
				if (in_image.data[i * width + j] == 255) {
					bool supress = false;
					for (int a = 0; a < h; ++a) {
						for (int b = 0; b < w; ++b) {
							int index = (i + a - h / 2) * width + j + b - w / 2;
							if (i+a - h/2 < 0 || i + a - h/2 >= height || j + b - w/2 < 0 || j + b - w/2 >= width)
								continue;
							if(in_image.data[index] != 255 && struct_element[a][b] == 1)
								supress = true;
						}
					}
					if (!supress)
						out_image.data[i * width + j] = 255;
				}
			}
		}
		break;
	case GRAYSCALE:
		//TODO
		break;
	case COLOR:
		//TODO
		break;
	default:
		return;
	}
}

void ImageProcessing::dilation(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w, int type)
{
	int height = in_image.rows;
	int width = in_image.cols;
	switch (type) {
	case BINARY:
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				out_image.data[i * width + j] = 0;
				if (in_image.data[i * width + j] == 255) {
					for (int a = 0; a < h; ++a) {
						for (int b = 0; b < w; ++b) {
							int index = (i + a - h / 2) * width + j + b - w / 2;
							if (i + a - h / 2 < 0 || i + a - h / 2 >= height || j + b - w / 2 < 0 || j + b - w / 2 >= width)
								continue;
							out_image.data[index] = 255;
						}
					}						
				}
			}
		}
		break;
	case GRAYSCALE:
		//TODO
		break;
	case COLOR:
		//TODO
		break;
	default:
		return;
	}
}

void ImageProcessing::findBorder(const cv::Mat& in_image, cv::Mat& out_image) {
	int height = in_image.rows;
	int width = in_image.cols;

	int sheight = 5;
	int swidth = 5;
	double** struct_element = new2DArray(sheight, swidth, 1);
	erosion(in_image, out_image, struct_element, sheight, swidth, BINARY);
	cv::imshow("Erosion", out_image);

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			out_image.data[i * width + j] = shrink(in_image.data[i * width + j] - out_image.data[i * width + j], 0, 255);
		}
	}
}
