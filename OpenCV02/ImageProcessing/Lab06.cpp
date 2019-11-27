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
	
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar max_value = 0;
			for (int a = 0; a < h; ++a) {
				for (int b = 0; b < w; ++b) {
					int index = (i + a - h / 2) * width + j + b - w / 2;
					if (i + a - h / 2 < 0 || i + a - h / 2 >= height || j + b - w / 2 < 0 || j + b - w / 2 >= width)
						continue;
					if (struct_element[a][b] == 1)
						max_value = std::max(max_value, in_image.data[index]);
				}
			}
			out_image.data[i * width + j] = max_value;
		}
	}
	return;
}
void ImageProcessing::opening(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w)
{
	cv::Mat temp_image(in_image.size(), in_image.type());
	erosion(in_image, temp_image, struct_element, h, w);
	dilation(temp_image, out_image, struct_element, h, w);
}

void ImageProcessing::closing(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w)
{
	dilation(out_image, out_image, struct_element, h, w);
	erosion(in_image, out_image, struct_element, h, w);
}

void ImageProcessing::distance_transform(const cv::Mat& in_image, cv::Mat& out_image)
{
	cv::distanceTransform(in_image, out_image, cv::DIST_L2, 5);
	cv::normalize(out_image, out_image, 0, 255, cv::NORM_MINMAX, CV_8U);
}

double** ImageProcessing::getStructElement(StructElement type, int height, int width)
{
	double** element = new2DArray(height, (width == 0 ? height : width));
	switch (type) {
	case StructElement::SQUARE:
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < height; ++j) {
				element[i][j] = 1;
			}
		}
		return element;
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

