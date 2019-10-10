#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageProcessing.h"

int main(int argc, char* argv[]) {
	srand(time(0));
	std::string image_path = "C:/Users/dmitr/source/repos/OpenCV02/images/bike.jpg";
	cv::Mat image;
	if (argc > 1)
		image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	else
		image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	if (image.empty())
		std::cout << "Failed to load the image.";
	else {
		int input = 0;
		std::cout << "Choose your operation: ";
		std::cout << "0 - Histogram equalization," << std::endl;
		std::cout << "1 - Log transformation," << std::endl;
		std::cout << "2 - Gamma correction, " << std::endl;
		std::cout << "3 - Piecewise linear transformation, " << std::endl;
		std::cout << "4 - Gauss blur, " << std::endl;
		std::cout << "5 - Median filter, " << std::endl;
		std::cout << "6 - Salt and Pepper noise, " << std::endl;
		std::cout << "7 - Normal -> Salt and Pepper -> Median and Gauss, " << std::endl;
		std::cin >> input;
		switch (input) {
		case 0: {
			cv::Mat hist_image = cv::Mat::zeros(image.size(), image.type());
			cv::Mat inHistImage(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));
			cv::Mat outHistImage(400, 512, CV_8UC3, cv::Scalar(0, 0, 0));

			ImageProcessing::HistogramEqualization(image, hist_image, inHistImage, outHistImage);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Equalized image", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Equalized image", hist_image);
			cv::imshow("In_histogram", inHistImage);
			cv::imshow("Out_histogram", outHistImage);
			cv::waitKey(0);

			break;
		}
		case 1: {
			cv::Mat out_image(cv::Size(image.size().width, image.size().height), image.type());
			ImageProcessing::LogTransformation(image, out_image);
			cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Log Transformation", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original", image);
			cv::imshow("Log Transformation", out_image);
			cv::waitKey(0);

			break;
		}
		case 2: {
			cv::Mat out_image(cv::Size(image.size().width, image.size().height), image.type());
			ImageProcessing::GammaCorrection(image, out_image);
			cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Exp Transformation", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original", image);
			cv::imshow("Exp Transformation", out_image);
			cv::waitKey(0);

			break;
		}
		case 3: {
			cv::Mat out_image(cv::Size(image.size().width, image.size().height), image.type());
			ImageProcessing::PiecewiseLinearTransformation(image, out_image);
			cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Piecewise Linear Transformation", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original", image);
			cv::imshow("Piecewise Linear Transformation", out_image);
			cv::waitKey(0);

			break;
		}
		case 4: {
			cv::Mat out_image(image.size(), image.type());
			ImageProcessing::GaussianBlur(image, out_image, 5, 1);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Gaussian Blur", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Gaussian Blur", out_image);
			cv::waitKey(0);
			break;
		}
		case 5: {
			cv::Mat out_image(image.size(), image.type());
			ImageProcessing::MedianFilter(image, out_image, 3);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Median Filter", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Median Filter", out_image);
			cv::waitKey(0);
			break;
		}
		case 6: {
			cv::Mat out_image(image.size(), image.type());
			ImageProcessing::SaltAndPepperNoise(image, out_image, 3);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Salt and Pepper", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Salt and Pepper", out_image);
			cv::waitKey(0);
			break;
		}
		case 7: {
			cv::Mat out_image(image.size(), image.type());
			cv::Mat out_image2(image.size(), image.type());
			cv::Mat out_image3(image.size(), image.type());
			ImageProcessing::SaltAndPepperNoise(image, out_image, 10);
			ImageProcessing::MedianFilter(out_image, out_image2, 3);
			ImageProcessing::GaussianBlur(out_image, out_image3, 3, 1);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Salt and Pepper", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Median Filter", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Gauss Filter", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Salt and Pepper", out_image);
			cv::imshow("Median Filter", out_image2);
			cv::imshow("Gauss Filter", out_image3);
			cv::waitKey(0);
			break;
		}
		case 8: {
			cv::Mat blurred(image.size(), image.type());
			ImageProcessing::GaussianBlur(image, blurred, 5, 1);
			cv::Mat out_image(image.size(), image.type());
			ImageProcessing::SobelOperator(blurred, out_image);
			cv::namedWindow("Original image", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Sobel", cv::WINDOW_AUTOSIZE);

			cv::imshow("Original image", image);
			cv::imshow("Sobel", out_image);
			cv::waitKey(0);
			break;
		}
		}
		
	}
}