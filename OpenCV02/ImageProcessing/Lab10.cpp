#include "ImageProcessing.h"
#include <cmath>

std::vector<double> ImageProcessing::calculateCurvature(std::vector<cv::Point> contour, int step)
{	
	std::vector<double> plot(contour.size());

	if (contour.size() < step)
		return plot;

	//Is contour closed
	cv::Point frontToBack = contour.front() - contour.back();
	bool isClosed = ((int)std::max(std::abs(frontToBack.x), std::abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < contour.size(); i++) {
		const cv::Point2f& pos = contour[i];

		int maxStep = step;
		if (!isClosed) {
			maxStep = std::min(std::min(step, i), (int)contour.size() - 1 - i);
			if (maxStep == 0)
			{
				plot[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}
		int iminus = i - maxStep;
		int iplus = i + maxStep;
		pminus = contour[iminus < 0 ? iminus + contour.size() : iminus];
		pplus = contour[iplus >= contour.size() ? iplus - contour.size() : iplus];

		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double curvature2D;
		double divisor = f1stDerivative.x * f1stDerivative.x + f1stDerivative.y * f1stDerivative.y;
		if (std::abs(divisor) > 10e-8) {
			curvature2D = std::abs(f2ndDerivative.y * f1stDerivative.x - f2ndDerivative.x * f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else {
			curvature2D = std::numeric_limits<double>::infinity();
		}
		plot[i] = curvature2D;
	}
	return plot;
}