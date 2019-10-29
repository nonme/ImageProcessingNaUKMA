#include "ImageProcessing.h"

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