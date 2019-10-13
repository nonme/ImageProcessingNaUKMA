#include "ImageProcessing.h"

double ImageProcessing::shrink(double val, double left, double right) {
	if (val < left)
		return left;
	if (val > right)
		return right;
	return val;
}