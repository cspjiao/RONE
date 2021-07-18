/*
 * dampedSimilarity.h
 *
 *  Created on: 18/08/2014
 *      Author: jefcha
 */

#include <list>

#ifndef DAMPEDSIMILARITY_H_
#define DAMPEDSIMILARITY_H_

//class Similarity
//{
//protected:
//
//public:
//
//
//	virtual double* computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum) = 0;
//
//};



/**
 * Pure abstract class for iteration based similarity measures.
 */
class IterSimilarity
{
protected:

	int m_maxIter;

	float m_convEpsilon;

	bool m_bUseConvEpsilon;

	/** Number of actual iterations ran for. */
	int m_iterRan;



#ifdef _COLLECT_SIM_DELTA_
	std::vector<float> m_vSimDelta;
#endif

#ifdef _COLLECT_EARLYSTOP_STATS_
	std::vector<int> m_iterConverged;
#endif

public:

	IterSimilarity(int maxIter);

	IterSimilarity(int maxIter, float convEpsilon);


	virtual ~IterSimilarity();

	void setMaxIter(int maxIter);

	int getIterRan() const;

	virtual float* computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum) = 0;


#ifdef _COLLECT_SIM_DELTA_
	virtual const std::vector<float>& getSimDelta() const;
#endif

#ifdef _COLLECT_EARLYSTOP_STATS_

	void matIteration(const float* const mMat1, const float* const mMat2, int rowNum, int colNum, float stopThreshold, int currIter);

	const std::vector<int>& getIterConverged() const;
#endif
};



/**
 * Pure abstract class for damped similarity measures.
 */
class DampedSimilarity : public IterSimilarity
{
protected:

	float m_dampingFactor;


public:

	DampedSimilarity(float dampingFactor, int maxIter);

	DampedSimilarity(float dampingFactor, int maxInter, float convEpsilon);

	virtual ~DampedSimilarity();

	void setDampingFactor(float dampingFactor);



};




#endif /* DAMPEDSIMILARITY_H_ */
