/*
 * RoleSimInit.h
 *
 *  Created on: 19/08/2014
 *      Author: jefcha
 */

#include <vector>
#include <set>

#ifndef ROLESIMINIT_H_
#define ROLESIMINIT_H_



/**
 * Abstract class for initialisation.
 */
class InitAlgorRoleSim
{
protected:
	float m_dampingFactor;


public:
	InitAlgorRoleSim(float dampingFactor);

	virtual ~InitAlgorRoleSim();

	virtual void initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim) = 0;
}; // end of abstract class InitAlgor()


/* ******************************************************************** */

/**
 * Degree ratio similarity value intialisation.
 */
class DegRatioInitAlgorRoleSim : public InitAlgorRoleSim
{
public:
	DegRatioInitAlgorRoleSim(float dampingFactor);

	virtual ~DegRatioInitAlgorRoleSim();

	void initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim);
};



/**
 * Binary similarity value intialisation.
 */
class BinaryInitAlgorRoleSim : public InitAlgorRoleSim
{
public:
	BinaryInitAlgorRoleSim(float dampingFactor);

	virtual ~BinaryInitAlgorRoleSim();

	void initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim);
};


/**
 * Degree Binary similarity value intialisation.
 */
class DegBinaryInitAlgorRoleSim : public InitAlgorRoleSim
{
public:
	DegBinaryInitAlgorRoleSim(float dampingFactor);

	virtual ~DegBinaryInitAlgorRoleSim();

	void initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim);
};





#endif /* ROLESIMINIT_H_ */
