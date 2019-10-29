#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageProcessing.h"
void onlyCircles(cv::Mat image, cv::Mat out_image);

int main(int argc, char* argv[]) {
	srand(time(0));
	std::string image_path = "images/im_kola.tif";
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
		std::cout << "8 - Sobel opertaor" << std::endl;
		std::cout << "9 - Canny Edge Detector" << std::endl;
		std::cout << "10 - Border (contour.tif)" << std::endl;
		std::cout << "11 - Laplace operator" << std::endl;
		std::cout << "12 - Prewitt, Scharr and Roberts operators" << std::endl;
		std::cout << "13 - Zero cross operator" << std::endl;
		std::cout << "14 - Circles" << std::endl;
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
			case 9: {
				cv::Mat out_image(image.size(), image.type());
				ImageProcessing::CannyEdgeDetection(image, out_image);
				cv::imshow("Original", image);
				cv::imshow("Canny", out_image);
				
				cv::Mat openCV(image.size(), image.type());
				ImageProcessing::GaussianBlur(image, openCV, 5, 1.4);
				cv::Canny(openCV, openCV, 20, 60, 3);
				cv::imshow("OpenCV", openCV);
				cv::waitKey(0);
				break;
			}
			case 10: {
				cv::Mat out_image(image.size(), image.type());
				ImageProcessing::findBorder(image, out_image);
				cv::imshow("Original", image);
				cv::imshow("Find border", out_image);
				cv::waitKey(0);
				break;
			}
			case 11: {
				cv::Mat out_image(image.size(), image.type());
				cv::Mat restored_backgorund(image.size(), image.type());
				ImageProcessing::LaplaceOperator(image, out_image);
				ImageProcessing::apply(image, out_image, restored_backgorund, 1.7);
				cv::imshow("Original", image);
				cv::imshow("Laplasian", out_image);
				cv::imshow("Restored", restored_backgorund);
				cv::waitKey(0);
				break;
			}
			case 12: {
				cv::Mat pr_image(image.size(), image.type());
				cv::Mat rob_image(image.size(), image.type());
				cv::Mat sch_image(image.size(), image.type());
				cv::Mat sobel_image(image.size(), image.type());

				ImageProcessing::SobelOperator(image, sobel_image);
				ImageProcessing::ScharrOperator(image, sch_image);
				ImageProcessing::RobertsOperator(image, rob_image);
				ImageProcessing::PrewittOperator(image, pr_image);

				cv::imshow("Original", image);
				cv::imshow("Sobel", sobel_image);
				cv::imshow("Roberts", rob_image);
				cv::imshow("Scharr", sch_image);
				cv::imshow("Prewitt", pr_image);

				cv::waitKey(0);
			}
			case 13: {
				cv::Mat out_image(image.size(), image.type());
				ImageProcessing::GaussianBlur(image, image, 3, 1.4);
				ImageProcessing::ZeroCrossOperator(image, out_image);

				cv::imshow("Original (blurred)", image);
				cv::imshow("Zero crossed", out_image);
				
				cv::waitKey(0);
			}
			case 14: {
				cv::Mat out_image(image.size(), image.type());
				onlyCircles(image, out_image);

				cv::imshow("Original", image);
				cv::imshow("Output", out_image);

				cv::waitKey(0);
			}
		}	
	}
}

void onlyCircles(cv::Mat image, cv::Mat out_image) {
	double** s1 = new double* [10];
	for (int i = 0; i < 10; ++i)
		s1[i] = new double[10] { 1, 1, 1 };
	//ImageProcessing::erosion(image, out_image, s1, 10, 10);
	//cv::imshow("1", out_image);
	double** s2 = new double* [10];
	for (int i = 0; i < 10; ++i)
		s2[i] = new double[1]{ 1 };
	ImageProcessing::erosion(image, out_image, s2, 10, 1);
	cv::imshow("2", out_image);
	cv::Mat second(image.size(), image.type());
	double** s3 = new double* [1];
	s3[0] = new double[6] { 1, 1, 1, 1, 1, 1};
	ImageProcessing::erosion(out_image, second, s3, 1, 6);
	cv::imshow("3", second);

	double** s4 = new double* [3];
	for (int i = 0; i < 3; ++i)
		s4[i] = new double[3];
	s4[0][0] = 0; s4[0][1] = 1; s4[0][2] = 0;
	s4[1][0] = 0; s4[1][1] = 1; s4[1][2] = 0;
	s4[2][0] = 0; s4[2][1] = 1; s4[2][2] = 0;
	ImageProcessing::dilation(second, out_image, s3, 1, 6);
	ImageProcessing::dilation(out_image, second, s3, 1, 6);
	ImageProcessing::dilation(second, out_image, s3, 1, 6);
	ImageProcessing::dilation(out_image, second, s2, 10, 1);
	ImageProcessing::dilation(second, out_image, s2, 10, 1);
	/*double** s1 = new double* [4];
	for (int i = 0; i < 4; ++i)
		s1[i] = new double[4]{ 1, 1, 1, 1 };
	ImageProcessing::erosion(image, out_image, s1, 4, 4);

	double** s2 = new double* [3];
	for (int i = 0; i < 3; ++i)
		s2[i] = new double[3];
	s2[0][0] = 0; s2[0][1] = 1; s2[0][2] = 1;
	s2[1][0] = 0; s2[1][1] = 1; s2[1][2] = 0;
	s2[2][0] = 1; s2[2][1] = 1; s2[2][2] = 0;
	ImageProcessing::erosion(image, out_image, s2, 3, 3);
	ImageProcessing::erosion(image, out_image, s2, 3, 3);*/
}