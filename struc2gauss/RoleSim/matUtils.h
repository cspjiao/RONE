/*
 * matUtils.h
 *
 *  Created on: 18/08/2014
 *      Author: jefcha
 */

#include <vector>
//#include "pairUtils.h"

#ifndef MATUTILS_H_
#define MATUTILS_H_


/** L1 distance between two matrices
 */
extern
float l1MatDiff(const float* const mMat1, const float* const mMat2, int rowNum, int colNum);





/**
 * Computer which values have changed less than threshold and set these to false.
 */
extern
void matChange(const float* const mMat1, const float* const mMat2, int rowNum, int colNum, bool* mbCompEntry, float stopThreshold);



#endif /* MATUTILS_H_ */
