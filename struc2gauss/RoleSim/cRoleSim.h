/*
 * Rolesim.h
 *
 *  Created on: 19/08/2014
 *      Author: jefcha
 */

#ifndef ROLESIM_H_
#define ROLESIM_H_

#include <list>
#include <string>
#include <stdexcept>

#include "dampedSimilarity.h"
#include "RoleSimInit.h"








class RoleSim : public DampedSimilarity
{
protected:



	/** Initialisation algorithm. */
	InitAlgorRoleSim* m_pfInitAlgor;



private:

#ifdef _COLLECT_MATCHING_CHANGES_
	typedef std::vector<std::pair<std::vector<int>, std::vector<int> > > C_MatchingMatrix;
	C_MatchingMatrix m_prevInMatchingPairMatrix;
	C_MatchingMatrix m_prevOutMatchingPairMatrix;
	std::vector<int> m_matchingChanges;
#endif


public:

	RoleSim(float dampingFactor, int maxIter, const std::string& sInitAlgor) throw(std::invalid_argument);

	RoleSim(float dampingFactor, int maxIter, float convEpsilon, const std::string& sInitAlgor) throw(std::invalid_argument);

	virtual ~RoleSim();




	virtual float* computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum);


}; // end of class AutoSim




#endif /* ROLESIM_H_ */
