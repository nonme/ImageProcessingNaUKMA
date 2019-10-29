#pragma once
#include <opencv2/opencv.hpp>
class ImageProcessing {
private:
	static constexpr double PI = 3.14159265359;
	static constexpr double E = 2.7182818284590452353602874713527;

	/*
		If val is lesser than left, returns left, if val is bigger than right returns right, else returns val
	*/
	static double shrink(double val, double left, double right);
	static void recursiveHysteresis(cv::Mat& image, int low_threshold, int high_threshold, int i = 0, int j = 0, int length = 0);
	static void arrayToMat(double** arr, cv::Mat& image, int rows, int cols);
	static double normalize(double value, double local_value, double max_value);

	static void delete2DArray(double** arr, int rows);
	static double** new2DArray(int height, int width, int fillValue = -1);
public:
	/*
		Decrease the size of image, size / N, where N is the natural number
	*/
	static void decrease(const cv::Mat in_image, cv::Mat& out_image, int n = 2);
	/*
		Increase the size of image, size * N where N is the natural number 
	*/
	static void increase(const cv::Mat in_image, cv::Mat& out_image, int n = 2);
	/*
		Additional task, do not use. Must be private ?
	*/
	static void invertImage(cv::Mat& image);
	/*
		Function to improve the image contrast by equalizing it's histogram.
	*/
	static void HistogramEqualization(const cv::Mat& in_image, cv::Mat& out_image,
		cv::Mat& in_histogram, cv::Mat& out_histogram);
	/*
		Function to change the brightness of image, makes image brighter
	*/
	static void LogTransformation(const cv::Mat& in_image, cv::Mat& out_image, double c = -1);
	/*
		
	*/
	static void GammaCorrection(const cv::Mat& in_image, cv::Mat& out_image, double c = 1, double y = 0.6);
	/*
		
	*/
	static void PiecewiseLinearTransformation(const cv::Mat& in_image, cv::Mat& out_image, double n = 2);
	/*
	
	*/
	static void LaplaceOperator(const cv::Mat& in_image, cv::Mat& out_image);
	static void LaplaceOperator(const cv::Mat& in_image, double** result);
	static void DiagonalLaplaceOperator(const cv::Mat& in_image, cv::Mat& out_image); //TODO
	/*
		
	*/
	static void ZeroCrossOperator(const cv::Mat& in_image, cv::Mat& out_image, int threshold = 15);
	/*
	
	*/
	static void RobertsOperator(const cv::Mat& in_image, cv::Mat& out_image);
	/*

	*/
	static void PrewittOperator(const cv::Mat& in_image, cv::Mat& out_image);
	/*
		Function that add's pixels in in_image and pixels in applied_image and saves in out_image.
	*/
	static void apply(const cv::Mat& in_image, cv::Mat& applied_image, cv::Mat& out_image, double coef = 1);
	/*
		Convolution that returns cv::Mat as the output. Suitable for Gaussian filter and other filters
		that have coefficient and produce only positive values. Otherwise, may cause overflow.
	*/
	static void convolution(const cv::Mat& in_image, cv::Mat& out_image, double mask[], int ksize, double coef);
	/*
		Convolution that returns double values of pixel. Useful for Sobel, Laplace and other filters that output
		negative values.
	*/
	static void convolution(const cv::Mat& in_image, double** out_image, double mask[], int ksize, double coef);
	/*
		Convolution for separability
	*/
	static void convolution(const cv::Mat& in_image, cv::Mat& out_image, double x_mask[], double y_mask[], int ksize, double coef); //TODO
	/*
	
	*/
	static void GaussianBlur(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double sigma);
	/*
		Низькочастотні, або ж усереднюючі, або ж сглажувальні фільтри використовуються
		для зменшення різьких переходів рівня яркості на зображенні. Таким чином, вони
		використовуються для заглушення "несуттєвих деталей на зображенні", тобто такі
		сукупності пікселей, які малі порівняно з розмірами маски фільтра.
	*/
	static void LowpassFilter(const cv::Mat& in_image, cv::Mat& out_image, int ksize);
	/*
		Median filter is a non-linear filter used to reduce noise.
	*/
	static void MedianFilter(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double quantile = 0.5);
	/*
		Salt and pepper filter add noise to the image
	*/
	static void SaltAndPepperNoise(const cv::Mat& in_image, cv::Mat& out_image, int noise_percent);
	/*
	
	*/
	static double** SobelOperator(const cv::Mat& in_image, cv::Mat& out_image);
	static double** ScharrOperator(const cv::Mat& in_image, cv::Mat& out_image);
	/*
	
	*/
	static void CannyEdgeDetection(const cv::Mat& in_image, cv::Mat& out_image, int l_threshold = -1, int h_threshold = -1);
	/*
		Otsu's method to find threshold of the image
	*/
	static double OtsuThreshold(const cv::Mat& in_image);
	
	static void erosion(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w, int type = BINARY);
	static void dilation(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w, int type = BINARY);
	static void findBorder(const cv::Mat& in_image, cv::Mat& out_image);

	static void opening(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w);
	static void closing(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w);

	enum {
		BINARY, GRAYSCALE, COLOR
	};
}; 