/*
 * matUtils.cpp
 *
 *  Created on: 18/08/2014
 *      Author: jefcha
 */

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cassert>

#include "matUtils.h"


float l1MatDiff(const float* const mMat1, const float* const mMat2, int rowNum, int colNum)
{
	float diff = 0;
	for (int c = 0; c < colNum; ++c) {
		for (int r = 0; r < rowNum; ++r) {
//			std::cout << mMat1[r+c*rowNum] << std::endl;
//			std::cout << mMat2[r+c*rowNum] << std::endl;
//			std::cout << mMat1[r + c*rowNum] - mMat2[r + c*rowNum] << std::endl;
//			std::cout << "fabs = " << fabs(mMat1[r + c*rowNum] - mMat2[r + c*rowNum]) << std::endl;
			diff += fabs(mMat1[r + c*rowNum] - mMat2[r + c*rowNum]);
		}
	}

	return diff;
} // end of l1MatDiff






void matChange(const float* const mMat1, const float* const mMat2, int rowNum, int colNum, bool* mbCompEntry, float stopThreshold)
{
	// symmetric, so don't need to update
	for (int c = 0; c < colNum; ++c) {
		for (int r = c+1; r < rowNum; ++r) {
			// if the entry is something we are still computing and the difference is now at or less than stopThreshold, we should not compute it
			if (mbCompEntry[r + c * rowNum] && (fabs(mMat1[r + c*rowNum] - mMat2[r + c*rowNum]) <= stopThreshold)) {
				mbCompEntry[r + c * rowNum] = false;
			}
		}
	}

} // end of matChange()




