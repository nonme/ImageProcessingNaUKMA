#include "ImageProcessing.h"

cv::Mat ImageProcessing::findMarkers(const cv::Mat& image, bool showSteps)
{
	// Step #1: threshold image to convert it to binary and use opening to remove the noise
	cv::Mat opened(image.size(), image.type());
	int threshold = OptimalThreshold(image);
	toBinary(image, opened, threshold);
	invertImage(opened);

	if (showSteps)
		cv::imshow("1 - Binary", opened);
	
	double** kernel = getStructElement(StructElement::SQUARE, 3);

	opening(opened, opened, kernel, 3, 3);

	if (showSteps)
		cv::imshow("2 - Opened", opened);

	// Step #2: Now we need to find sure-background ans sure-foreground regions
	cv::Mat dilated(image.size(), image.type());
	cv::Mat temp(image.size(), image.type());
	dilation(opened, dilated, kernel, 3, 3); // This is sure background
	dilation(dilated, temp, kernel, 3, 3);
	dilation(temp, dilated, kernel, 3, 3);

	if (showSteps)
		cv::imshow("3 - Dilated", dilated);

	cv::Mat dt(image.size(), image.type()); // This is going to be sure foreground
	distance_transform(opened, dt);

	if (showSteps)
		cv::imshow("4 - Dt", dt);

	int dt_threshold = 0.7 * maxInMat(dt);
	toBinary(dt, dt_threshold);

	if (showSteps)
		cv::imshow("5 - Dt thresholded", dt);

	// Step #3: Subtract dt image (sure-foreground) from dilated image (sure-background) to get unsure (unknown) positions  
	cv::Mat unknown = subtract(opened, dt);

	// Step #4: Markup all sure-foregrounds and sure-background to different colors and unknown regions to 0
	cv::Mat markers(image.size(), image.type());
	RegionMarking(dt, markers);
	int marker = maxInMat(markers) + 1;
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			if (unknown.data[i * image.cols + j] == 255)
				markers.data[i * image.cols + j] = 0;
			else if (opened.data[i * image.cols + j] == 0)
				markers.data[i * image.cols + j] = marker;
		}
	}
	
	if(showSteps)
		cv::imshow("6 - Markers", markers);

	delete2DArray(kernel, 3);

	return markers;
}

void ImageProcessing::watershed(const cv::Mat& image, cv::Mat& out_image, cv::Mat& markers)
{
	int height = image.rows;
	int width = image.cols;

	int background = maxInMat(markers);
	bool isFinished = false;
	cv::Mat newMarkers(markers.size(), markers.type());
	markers.copyTo(newMarkers);
	cv::imshow("Markers before", newMarkers);
	while (!isFinished) {
		isFinished = true;
		for (int i = 0; i < height; ++i) {
			for (int j = 0; j < width; ++j) {
				uchar min_value = 255;
				uchar max_value = 0;
				for (int a = -1; a <= 1; ++a) {
					for (int b = -1; b <= 1; ++b) {
						if (i + a < 0 || i + a >= height || j + b < 0 || j + b >= width)
							continue;
						if (markers.data[(i + a) * width + j + b] != 0) {
							min_value = std::min(min_value, markers.data[(i + a) * width + j + b]);
						}
						max_value = std::max(max_value, markers.data[(i + a) * width + j + b]);
					}
				}
				if (min_value != max_value && min_value != 255 && max_value != 0) {
					out_image.data[i * width + j] = 255;
				}
				else if (newMarkers.data[i * width + j] == 0 && max_value != background) {
					newMarkers.data[i * width + j] = max_value;
					isFinished = false;
				}
			}
		}
		newMarkers.copyTo(markers);
	}
	cv::imshow("Markers", markers);
}
