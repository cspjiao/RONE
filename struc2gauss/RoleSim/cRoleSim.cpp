/*
 * RoleSim.cpp
 *
 *  Created on: 19/08/2014
 *      Author: jefcha
 */



#include <iostream>
#include <vector>
#include <list>
#include <set>
#include <stdexcept>
#include <cstdlib>
#include <cassert>

#include "cRoleSim.h"
#include "RoleSimInit.h"
#include "BMatching.h"
#include "matUtils.h"




RoleSim::RoleSim(float dampingFactor, int maxIter,  const std::string& sInitAlgor) throw(std::invalid_argument)
	: DampedSimilarity(dampingFactor, maxIter), m_pfInitAlgor(NULL)
{
	if (sInitAlgor == "binaryInit") {
		m_pfInitAlgor = new BinaryInitAlgorRoleSim(dampingFactor);
	}
	else if (sInitAlgor == "degBinaryInit") {
		m_pfInitAlgor = new DegBinaryInitAlgorRoleSim(dampingFactor);
	}
	else if (sInitAlgor == "degRatioInit") {
		m_pfInitAlgor = new DegRatioInitAlgorRoleSim(dampingFactor);
	}
	else {
		throw std::invalid_argument("RoleSim: invalid initialisation algorithm");
	}
} // end of AutoSim()



RoleSim::RoleSim(float dampingFactor, int maxIter, float convEpsilon, const std::string& sInitAlgor) throw(std::invalid_argument)
	: DampedSimilarity(dampingFactor, maxIter, convEpsilon), m_pfInitAlgor(NULL)
{
	if (sInitAlgor == "binaryInit") {
		m_pfInitAlgor = new BinaryInitAlgorRoleSim(dampingFactor);
	}
	else if (sInitAlgor == "degBinaryInit") {
		m_pfInitAlgor = new DegBinaryInitAlgorRoleSim(dampingFactor);
	}
	else if (sInitAlgor == "degRatioInit") {
		m_pfInitAlgor = new DegRatioInitAlgorRoleSim(dampingFactor);
	}
	else {
		throw std::invalid_argument("RoleSim: invalid initialisation algorithm");
	}
} // end of AutoSim()





RoleSim::~RoleSim()
{
	delete m_pfInitAlgor;
} // end of ~AutoSim()










float* RoleSim::computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum)
{
	using namespace std;

    // similarity matrix (column-major)
    float* mPrevSim = new float[vertNum*vertNum];
    float* mCurrSim = new float[vertNum*vertNum];
    float **pmPrevSim = &mPrevSim;
    float **pmCurrSim = &mCurrSim;


    // construct neighbour list
    vector< set<int> > vvNeigh(vertNum);

    // set the neighbourhoods and degrees
    std::list<int>::const_iterator sit = vSrc.begin(), tit = vTar.begin();
    for ( ; sit != vSrc.end(); ++sit, ++tit) {
    	cout << *sit << ", " << *tit << "," << vertNum << endl;
    	assert(*sit < vertNum && *tit < vertNum);
    	vvNeigh[*tit].insert(*sit);
        vvNeigh[*sit].insert(*tit);
    } // end of for




    // initialise similarity matrix
    m_pfInitAlgor->initialise(vvNeigh, *pmPrevSim);

//    for (int i = 0; i < vertNum; ++i) {
//        for (int j = 0; j < vertNum; ++j) {
//        	cout << mPrevSim[i + j*vertNum] << ",";
//        }
//        cout << endl;
//    }



    // temporary structure for mIn and mOut
    vector<float> mNeigh(vertNum * vertNum);

    // perform loop iterations
    for (int t = 1; t <= m_maxIter; ++t) {
    	cout << "iteration " << t << endl;



        float* mTempPrevSim = *pmPrevSim;
        float *mTempCurrSim = *pmCurrSim;
        // loop through pairs
        for (int i = 0; i < vertNum; ++i) {
            for (int j = i+1; j < vertNum; ++j) {

//            	cout << "pair (" << i << ", " << j << ")" << endl;

                // matching
                float matchCost = 0;
                int degI = vvNeigh[i].size(), degJ = vvNeigh[j].size();
                if (degI > 0 && degJ > 0) {
//                    float *mIn = new float[vInDeg[i] * vInDeg[j]];

                    // initialise the cost matrix
                	int x = 0, y = 0;
                    for (set<int>::const_iterator xit = vvNeigh[i].begin(); xit != vvNeigh[i].end();
                    		++xit, ++x)
                    {
                    	y = 0;
                        for (set<int>::const_iterator yit = vvNeigh[j].begin(); yit != vvNeigh[j].end();
                        		++yit, ++y)
                        {
                            mNeigh[x + y*degI] = mTempPrevSim[*xit + *yit * vertNum];
                        }
                    }


                    vector<int> m1;
                    vector<int> m2;
                    matchCost = matching(degI, degJ, mNeigh, m1, m2);

                    assert(degI > 0 && degJ > 0);
                    matchCost = matchCost / max(degI, degJ);
//                    delete[] mIn;
                }

                if (max(degI, degJ) == 0) {
                    mTempCurrSim[i + j*vertNum] = 0;
                }
                else {
                    mTempCurrSim[i + j*vertNum] = matchCost * m_dampingFactor + (1-m_dampingFactor);
                }

                // assign the other symmetric similarity
                mTempCurrSim[j + i*vertNum] = mTempCurrSim[i + j*vertNum];

            } // end of inner for
        } // end of outer for


        // loop through diagonal pairs (always 1)
	    for (int i = 0; i < vertNum; ++i) {
	    	mTempCurrSim[i + i * vertNum] = 1;
	    }





#ifdef _COLLECT_SIM_DELTA_
    	// do comparison
        if (t > 1) {
        	assert(*pmPrevSim != NULL);
        	assert(*pmCurrSim != NULL);

        	m_vSimDelta.push_back(l1MatDiff(*pmPrevSim, *pmCurrSim, vertNum, vertNum));
        }
#endif
        m_iterRan = t;

        // check if we have convergence epsilon
        if (m_bUseConvEpsilon) {
//        	cout << "l1MatDiff(*pmPrevSim, *pmCurrSim, vertNum, vertNum) = " << l1MatDiff(*pmPrevSim, *pmCurrSim, vertNum, vertNum) << endl;
        	if (l1MatDiff(*pmPrevSim, *pmCurrSim, vertNum, vertNum) / (vertNum * vertNum) < m_convEpsilon) {
        		break;
        	}
        }

        // swap the sim matrices so we do not need to allocate more matrices then need to be
        float **pmTempSim = pmPrevSim;
        pmPrevSim = pmCurrSim;
        pmCurrSim = pmTempSim;
    } // end of loop through iterations

    // destroy dynamically allocated memory
    delete[] *pmPrevSim;
    pmPrevSim = NULL;


    return *pmCurrSim;
} // end of AutoSim::computeSim()



















