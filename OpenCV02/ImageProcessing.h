﻿#pragma once
#include <opencv2/opencv.hpp>
class ImageProcessing {
private:
	static constexpr double PI = 3.14159265359;
	static constexpr double E = 2.7182818284590452353602874713527;
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
	static void convolution(const cv::Mat& in_image, cv::Mat& out_image, double mask[], int ksize, double coef);
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
};