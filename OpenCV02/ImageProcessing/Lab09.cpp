#include "ImageProcessing.h"

std::vector<std::vector<cv::Point> > ImageProcessing::TheoPavlidisAlgorithm(const cv::Mat& in_image)
{
	int height = in_image.rows;
	int width = in_image.cols;

	std::vector<std::vector<bool> > isVisited(height, std::vector<bool>(width, false));
	std::vector<std::vector<cv::Point> > contour;
	int color = 100;

	for (int i = height - 1; i >= 0; --i) {
		for (int j = 0; j < width; ++j) {
			if ( GetPixel(i, j, in_image) && !isVisited[i][j] &&
				!GetPixel(i, j, in_image, Direction::NORTH, Direction::WEST).value &&
				!GetPixel(i, j, in_image, Direction::NORTH, Direction::SOUTH_WEST).value &&
				!GetPixel(i, j, in_image, Direction::NORTH, Direction::SOUTH_EAST).value) {
				contour.push_back(processContour(in_image, i, j, Direction::NORTH, isVisited));
			}
			else if (
				GetPixel(i, j, in_image) && !isVisited[i][j] &&
				!GetPixel(i, j, in_image, Direction::EAST, Direction::WEST).value &&
				!GetPixel(i, j, in_image, Direction::EAST, Direction::SOUTH_WEST).value &&
				!GetPixel(i, j, in_image, Direction::EAST, Direction::SOUTH_EAST).value) {
				contour.push_back(processContour(in_image, i, j, Direction::EAST, isVisited));
			}
			else if (
				GetPixel(i, j, in_image) && !isVisited[i][j] &&
				!GetPixel(i, j, in_image, Direction::SOUTH, Direction::WEST).value &&
				!GetPixel(i, j, in_image, Direction::SOUTH, Direction::SOUTH_WEST).value &&
				!GetPixel(i, j, in_image, Direction::SOUTH, Direction::SOUTH_EAST).value) {
				contour.push_back(processContour(in_image, i, j, Direction::SOUTH, isVisited));
			}
			else if (
				GetPixel(i, j, in_image) && !isVisited[i][j] &&
				!GetPixel(i, j, in_image, Direction::WEST, Direction::WEST).value &&
				!GetPixel(i, j, in_image, Direction::WEST, Direction::SOUTH_WEST).value &&
				!GetPixel(i, j, in_image, Direction::WEST, Direction::SOUTH_EAST).value) {
				contour.push_back(processContour(in_image, i, j, Direction::WEST, isVisited));
			}
		}
	}
	return contour;
}

std::vector<cv::Point> ImageProcessing::processContour(const cv::Mat& in_image, int i, int j, Direction direction, std::vector<std::vector<bool> >& isVisited)
{
	std::vector<cv::Point> contour;
	cv::Point start(j, i);
	isVisited[i][j] = true;
	bool isFinished = false;
	int runs = 0;
	while (!isFinished) {
		if (runs++ > in_image.rows * in_image.cols / 10)
			break;
		Pixel northWest = GetPixel(i, j, in_image, direction, Direction::NORTH_WEST);
		Pixel north = GetPixel(i, j, in_image, direction, Direction::NORTH);
		Pixel northEast = GetPixel(i, j, in_image, direction, Direction::NORTH_EAST);
		contour.push_back(cv::Point(j, i));
		//std::cout << i << " " << j << " ";
		if (northWest.value) {
			if (northWest.i != start.y || northWest.j != start.x) {
				//std::cout << "NW";
				i = northWest.i;
				j = northWest.j;
				direction = turnLeft(direction);
			}
			else {
				isFinished = true;
				//std::cout << "Start point encountered";
			}
		}
		else if (north.value) {
			if (north.i != start.y || northWest.j != start.x) {
				//std::cout << "N";
				i = north.i;
				j = north.j;
			}
			else {
				isFinished = true;
				//std::cout << "Start point encountered";
			}
		}
		else if (northEast.value) {
			if (northEast.i != start.y || northEast.j != start.x) {
				//std::cout << "NE";
				i = northEast.i;
				j = northEast.j;
			}
			else {
				isFinished = true;
				//std::cout << "Start point encountered";
			}
		}
		else {
			int turns = 0;
			while (!northEast.value && !north.value && !northWest.value && turns < 3) {
				//std::cout << "Turn right" << std::endl;
				direction = turnRight(direction);
				northWest = GetPixel(i, j, in_image, direction, Direction::NORTH_WEST);
				north = GetPixel(i, j, in_image, direction, Direction::NORTH);
				northEast = GetPixel(i, j, in_image, direction, Direction::NORTH_EAST);
				turns++;
			}
			if (turns >= 3)
				isFinished = true;
		}
		//std::cout << std::endl;
		isVisited[i][j] = true;
	}
	return contour;
}
