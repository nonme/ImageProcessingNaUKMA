#pragma once
#include <opencv2/opencv.hpp>
class ImageProcessing {
private:
	static constexpr double PI = 3.14159265359;
	static constexpr double E = 2.7182818284590452353602874713527;

	enum Direction {
		NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST
	};
	enum class StructElement {
		SQUARE, RECTANGLE
	};
	struct Pixel {
		int i;
		int j;
		unsigned char value;
		Pixel(int i, int j, unsigned char value) : i(i), j(j), value(value) {};
	};
	struct PointData {
		cv::Point pos;
		double value;
		static bool byHarisResponses(const PointData& a, const PointData& b) {
			return a.value > b.value;
		}
	};
	struct Derivative {
		double** gx;
		double** gy;
		double** g;
		int height;
		int width;
		Derivative(int height, int width) {
			this->height = height;
			this->width = width;
			gx = new2DArray(height, width);
			gy = new2DArray(height, width);
			g = new2DArray(height, width);
		}
		~Derivative() {
			delete2DArray(gx, height);
			delete2DArray(gy, height);
			delete2DArray(g, height);
		}
	};
	/*
		If val is lesser than left, returns left, if val is bigger than right returns right, else returns val
	*/
	static double shrink(double val, double left, double right);
	static void recursiveHysteresis(cv::Mat& image, int low_threshold, int high_threshold, int i = 0, int j = 0, int length = 0);
	static void arrayToMat(double** arr, cv::Mat& image, int rows, int cols);
	static cv::Mat arrayToMat(double** arr, int rows, int cols);
	static double normalize(double value, double local_value, double max_value);
	static void normalize(double** arr, int height, int width, double alpha, double beta);
	static double maxInMat(const cv::Mat& image);
	static cv::Mat subtract(const cv::Mat& a, const cv::Mat& b);

	static void delete2DArray(double** arr, int rows);
	static double** new2DArray(int height, int width, int fillValue = -1);

	static Pixel GetPixel(int i, int j, const cv::Mat& image, Direction head, Direction direction);
	static unsigned char GetPixel(int i, int j, const cv::Mat& image);
	static Direction turnRight(Direction direction);
	static Direction turnLeft(Direction direction);
	class Line {
	private:
		double A;
		double B;
		double C;
	public:
		Line(double a, double b, double c) : A(a), B(b), C(c) {};
		Line(double x1, double y1, double x2, double y2) {
			A = y1 - y2;
			B = x2 - x1;
			C = x1 * y2 - x2 * y1;
		}
		double distance(double x, double y) {
			return std::abs(A * x + B * y + C) / sqrt(A * A + B * B);
		}
	};

	static double sumArray(double* arr, int size);
	static double sumArray(int* arr, int size);
	static double meanBrightness(int* histogram, int size = 256);
public:
	enum {
		BINARY, GRAYSCALE, COLOR
	};
	enum {
		FOUR_CONNECTED, EIGHT_CONNECTED
	};
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
	static void GetHistogram(const cv::Mat& in_image, int* histogram);
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

	*/
	static void convolution(double ** image, double** out_image, int height, int width, double mask[], int ksize, double coef);
	static void convolution(double ** image, int height, int width, double mask[], int ksize, double coef);
	/*
	
	*/
	static void GaussianBlur(const cv::Mat& in_image, cv::Mat& out_image, int ksize, double sigma);
	/*
		Низькочастотні, або ж усереднюючі, або ж сглажувальні фільтри використовуються
		для зменшення різьких переходів рівня яркості на зображенні. Таким чином, вони
		використовуються для заглушення "несуттєвих деталей на зображенні", тобто такі
		сукупності пікселей, які малі порівняно з розмірами маски фільтра.
	*/
	static void GaussianBlur(double** image, int height, int width, int ksize, double sigma = 1);

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
	static void SobelOperator(const cv::Mat& in_image, double** g, double** gx, double** gy);

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
	static void opening(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w);
	static void closing(const cv::Mat& in_image, cv::Mat& out_image, double** struct_element, int h, int w);
	static void distance_transform(const cv::Mat& in_image, cv::Mat& out_image);
	static double** getStructElement(StructElement type, int height, int width = 0);
	static void findBorder(const cv::Mat& in_image, cv::Mat& out_image);

	static void toBinary(const cv::Mat& in_image, cv::Mat& out_image, int threshold);
	static void toBinary(cv::Mat& image, int threshold);
	static void SemiThresholding(const cv::Mat& in_image, cv::Mat& out_image, int high_threshold = 120, int low_threshold = 30);
	static int TriangleAlgorithm(const cv::Mat& in_image);
	static int OptimalThreshold(const cv::Mat& in_image);

	static void RegionGrowing(const cv::Mat& in_image, cv::Mat& out_image, int seed, int threshold = -1, int connectivity = EIGHT_CONNECTED);
	static void RegionMarking(const cv::Mat& in_image, cv::Mat& out_image, int threshold = -1, int connectivity = EIGHT_CONNECTED);
	static void markRegion(const cv::Mat& in_image, cv::Mat& out_image, int i, int j, int start, int threshold, int color, int connectivity = EIGHT_CONNECTED);

	static void SquareTracing(const cv::Mat& in_image, cv::Mat& out_image);
	static void MooreNeighborTracing(const cv::Mat& in_image, cv::Mat& out_image);
	static std::vector<std::vector<cv::Point> > TheoPavlidisAlgorithm(const cv::Mat& in_image);

	static void drawContours(cv::Mat& image, std::vector<std::vector<cv::Point> > contours);
	static void plotCurvature(const std::vector<double>& plot, cv::Mat& out_image);

	static std::vector<double> calculateCurvature(std::vector<cv::Point> contour, int step = 3);

	static std::vector<cv::Point2f> HarrisCornerDetector(const cv::Mat& image, cv::Mat& out_image,
		double k, int filterRange, int threshold, bool applyGauss = true);
	static void computeHarrisResponses(double k, const Derivative& in_image, double** responses);
	static void applyGaussianToDerivative(Derivative& d, int filterRange);
	static void findLocalMaxima(double** responses, cv::Mat& image, double percentage, int radius);

	static cv::Mat findMarkers(const cv::Mat& image, bool showSteps = false);
	static void watershed(const cv::Mat& image, cv::Mat& out_image, cv::Mat& markers);

	private:
		static std::vector<cv::Point> processContour(const cv::Mat&, int, int, Direction, std::vector<std::vector<bool> >&);
	public:
	struct Point {
	public:
		int x;
		int y;
		Point(int x, int y) : x(x), y(y) {};

		bool operator==(const Point& other) const {
			return (this->x == other.x) && (this->y == other.y);
		}
	};
}; 