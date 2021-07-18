/*
 * cMatchSim.cpp
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

#include "cMatchSim.h"
#include "BMatching.h"
#include "matUtils.h"






MatchSim::MatchSim(int maxIter)
	: IterSimilarity(maxIter)
{
} // end of MatchSim()

MatchSim::MatchSim(int maxIter, float convEpsilon)
	: IterSimilarity(maxIter, convEpsilon)
{
} // end of MatchSim()



MatchSim::~MatchSim()
{
} // end of ~MatchSim()



void MatchSim::setMaxIter(int maxIter) {
	m_maxIter = maxIter;
}






float* MatchSim::computeSim(const std::list<int>& vSrc, const std::list<int>& vTar, int edgeNum, int vertNum)
{
	using namespace std;

    // similarity matrix (column-major)
    float* mPrevSim = new float[vertNum*vertNum];
    float* mCurrSim = new float[vertNum*vertNum];
    float **pmPrevSim = &mPrevSim;
    float **pmCurrSim = &mCurrSim;


    // construct neighbour list
    vector< vector<int> > vvInNeigh(vertNum);

    // set the neighbourhoods and degrees
    std::list<int>::const_iterator sit = vSrc.begin(), tit = vTar.begin();
    for ( ; sit != vSrc.end(); ++sit, ++tit) {
    	assert(*sit < vertNum && *tit < vertNum);
    	vvInNeigh[*tit].push_back(*sit);
    } // end of for

    // initialise the values of matchSim
    // non-diagonals to 0
    for (int c = 0; c < vertNum; ++c) {
    	for (int r = c+1; r < vertNum; ++r) {
    		mPrevSim[r + c*vertNum] = 0;
    		mPrevSim[c + r*vertNum] = 0;
    	}
    }
    // diagonals to 1
    for (int i = 0; i < vertNum; ++i) {
    	mPrevSim[i + i * vertNum]  = 1;
    }

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

                // In matching
                float matchCost = 0;
                int degI = vvInNeigh[i].size(), degJ = vvInNeigh[j].size();
                if (degI > 0 && degJ > 0) {
//                    float *mIn = new float[vInDeg[i] * vInDeg[j]];

                    // initialise the cost matrix
                	vector<int>::const_iterator xit = vvInNeigh[i].begin(), yit = vvInNeigh[j].begin();
                    for ( ; xit != vvInNeigh[i].end(); ++xit) {
                        for ( ; yit != vvInNeigh[j].end(); ++yit) {
                            mNeigh[*xit + *yit*degI] = mTempPrevSim[*xit + *yit * vertNum];
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
                    mTempCurrSim[i + j*vertNum] = matchCost;
                }

                // assign the other symmetric similarity
                mTempCurrSim[j + i*vertNum] = mTempCurrSim[i + j*vertNum];

            } // end of inner for
        } // end of outer for


        // loop through diagonal pairs (always equal to 1)
	    for (int i = 0; i < vertNum; ++i) {
	    	mTempCurrSim[i + i * vertNum] = 1;
	    }

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



