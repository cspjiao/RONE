/*
 * RolesimInit.cpp
 *
 *  Created on: 19/08/2014
 *      Author: jefcha
 */

#include <algorithm>
#include <vector>
#include <set>
#include <cstdlib>

#include "RoleSimInit.h"

InitAlgorRoleSim::InitAlgorRoleSim(float dampingFactor)
	: m_dampingFactor(dampingFactor)
{
}

InitAlgorRoleSim::~InitAlgorRoleSim()
{
}

/* ***************************************************************** */

DegRatioInitAlgorRoleSim::DegRatioInitAlgorRoleSim(float dampingFactor)
	: InitAlgorRoleSim(dampingFactor)
{
}

DegRatioInitAlgorRoleSim::~DegRatioInitAlgorRoleSim()
{}


void DegRatioInitAlgorRoleSim::initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim)
{
	using namespace std;

	int vertNum = vvNeigh.size();

	// non-diagonal
    for (int i = 0; i < vertNum; ++i) {
        for (int j = i+1; j < vertNum; ++j) {
        	int sizei = vvNeigh[i].size();
        	int sizej = vvNeigh[j].size();
            int maxDeg = max(sizei, sizej);
            float minDeg = min(sizei, sizej);

            if (maxDeg == 0) {
//            	mSim[i + j*vertNum] = 1 - m_dampingFactor;
            	mSim[i + j*vertNum] = 0;
            }
            else {
            	float OUT = minDeg / maxDeg;
                mSim[i + j*vertNum] = m_dampingFactor * OUT + 1 - m_dampingFactor;
            }

            mSim[j + i*vertNum] = mSim[i + j*vertNum];
        }
    } // end of outer for

    // diagonal
    for (int i = 0; i < vertNum; ++i) {
    	mSim[i + i*vertNum] = 1;
    } // end of outer for

} // end of initialise()

/* ********************************************************** */

BinaryInitAlgorRoleSim::BinaryInitAlgorRoleSim(float dampingFactor)
	: InitAlgorRoleSim(dampingFactor)
{
}


BinaryInitAlgorRoleSim::~BinaryInitAlgorRoleSim()
{}

void BinaryInitAlgorRoleSim::initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim)
{
	using namespace std;

	int vertNum = vvNeigh.size();

	// intialise all non-diagonal elements to 0
    for (int i = 0; i < vertNum; ++i) {
    	for (int j = i + 1; j < vertNum; ++j) {
    		mSim[i + j * vertNum] = 0;
    		mSim[j + i * vertNum] = 0;
    	}
    }

    // initialise all diagonal elements to 1
    for (int i = 0; i < vertNum; ++i) {
    	mSim[i + i * vertNum] = 1;
    } // end of outer for

} // end of initialise()


/* ********************************************************** */

DegBinaryInitAlgorRoleSim::DegBinaryInitAlgorRoleSim(float dampingFactor)
	: InitAlgorRoleSim(dampingFactor)
{
}


DegBinaryInitAlgorRoleSim::~DegBinaryInitAlgorRoleSim()
{}

void DegBinaryInitAlgorRoleSim::initialise(const std::vector< std::set<int> >& vvNeigh, float* mSim)
{
	using namespace std;

	int vertNum = vvNeigh.size();

	// non-diagonal elements
    for (int i = 0; i < vertNum; ++i) {
        for (int j = i+1; j < vertNum; ++j) {
        	int inDegi = vvNeigh[i].size();
        	int inDegj = vvNeigh[j].size();

        	if (inDegi == inDegj) {
        		mSim[i + j*vertNum] = 1;
        		mSim[j + i*vertNum] = 1;
        	}
        	else {
        		mSim[i + j*vertNum] = 0;
        		mSim[j + i*vertNum] = 0;
        	}
        }
    } // end of outer for

    // diagonal elements (always 1)
    // initialise all diagonal elements to 1
    for (int i = 0; i < vertNum; ++i) {
    	mSim[i + i * vertNum] = 1;
    } // end of outer for


} // end of initialise()


