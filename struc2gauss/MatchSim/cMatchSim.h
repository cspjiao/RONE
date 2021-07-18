/*
 * cMatchSim.h
 *
 *  Created on: 19/08/2014
 *      Author: jefcha
 */

#ifndef CMATCHSIM_H_
#define CMATCHSIM_H_


#include <list>
#include <string>
#include <stdexcept>

#include "dampedSimilarity.h"




class MatchSim : public IterSimilarity
{
protected:


public:

	MatchSim(int maxIter);

	MatchSim(int maxIter, float convEpsilon);

	virtual ~MatchSim();


	void setMaxIter(int maxIter);


	virtual float* computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum);


}; // end of class MatchSim




#endif /* CMATCHSIM_H_ */
