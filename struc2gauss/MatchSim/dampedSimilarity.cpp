/*
 * dampedSimilarity.cpp
 *
 *  Created on: 18/08/2014
 *      Author: jefcha
 */


#include <vector>
#include <cassert>
#include <cmath>
#include "dampedSimilarity.h"









/* **************************************************** */

IterSimilarity::IterSimilarity(int maxIter)
	: m_maxIter(maxIter), m_convEpsilon(0.1), m_bUseConvEpsilon(false), m_iterRan(0)
{
}

IterSimilarity::IterSimilarity(int maxIter, float convEpsilon)
	: m_maxIter(maxIter), m_convEpsilon(convEpsilon), m_bUseConvEpsilon(true), m_iterRan(0)
{
}






IterSimilarity::~IterSimilarity()
{
}


void IterSimilarity::setMaxIter(int maxIter) {
	m_maxIter = maxIter;
}


int IterSimilarity::getIterRan() const
{
	return m_iterRan;
}


#ifdef _COLLECT_SIM_DELTA_

const std::vector<float>& IterSimilarity::getSimDelta() const
{
	return m_vSimDelta;
}

#endif


#ifdef _COLLECT_EARLYSTOP_STATS_

void IterSimilarity::matIteration(const float* const mMat1, const float* const mMat2, int rowNum, int colNum, float stopThreshold, int currIter)
{
	// symmetric, so don't need to update
	for (int r = 0; r < colNum; ++r) {
		for (int c = r+1; c < rowNum; ++c) {
//			std::cout << mMat1[r + c*rowNum] << " - " << mMat2[r + c*rowNum] << " = " << fabs(mMat1[r + c*rowNum] - mMat2[r + c*rowNum]) << std::endl;
//			std::cout << stopThreshold << std::endl;

			// if the entry is something we are still computing and the difference is now at or less than stopThreshold, we should not compute it
			if (fabs(mMat1[r + c*rowNum] - mMat2[r + c*rowNum]) > stopThreshold) {
//				std::cout << "update with " << currIter << std::endl;
				m_iterConverged[r + c * rowNum] = currIter;
			}
		}
	}
} // end of matIteration()


const std::vector<int>& IterSimilarity::getIterConverged() const
{
	return m_iterConverged;
} // end of getIterConverged()

#endif

/* **************************************************** */

DampedSimilarity::DampedSimilarity(float dampedFactor, int maxIter)
	: IterSimilarity(maxIter), m_dampingFactor(dampedFactor)
{
	assert(m_dampingFactor >= 0 && m_dampingFactor <= 1);
}


DampedSimilarity::DampedSimilarity(float dampedFactor, int maxIter, float convEpsilon)
	: IterSimilarity(maxIter, convEpsilon), m_dampingFactor(dampedFactor)
{
	assert(m_dampingFactor >= 0 && m_dampingFactor <= 1);
}


DampedSimilarity::~DampedSimilarity()
{
}

void DampedSimilarity::setDampingFactor(float dampingFactor) {
	m_dampingFactor = dampingFactor;
}




