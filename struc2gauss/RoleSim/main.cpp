#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_set>

#include "cRoleSim.h"
#include "BMatching.h"

/* ******************************************************** */

//void usage(char* sProgname);
using namespace std;

/** Read in a file into the provided data structures. */
// void readGraphFromFile(const list<char*>& lFileNames, int& vertNum, list<int>& vSrc, list<int>& vTar);
void readGraph(const char* sGraphFilename, int& vertNum, list<int>& vSrc, list<int>& vTar);

// Command line options
//extern char* optarg;
//extern int optind, optopt;
//extern int optind;

// default values for input parameters of executable
int g_iterInfo = 10; // (need to convert to int if use as maxIter)
float g_convEpsilon = 0.1;
bool g_bUseConvEpsilon = false;

float g_dampingFactor = 0.9;
std::string g_initAlgorName = "degRatioInit";
float g_ioBalance = 0.5;
bool g_bUseInputBalance = false;
float g_icebergThres = 0.8;
float g_icebergApproxFactor = 0.5;
float g_earlySimStopThres = 0.01;
bool g_bVertSubtractOne = false;
int g_randGraphVertNum = 0;


/* ************************************************************* */


int main(int argc, char *argv[])
{
    //using namespace std;
    
//    for (int a = 0; a < argc; ++a) {
//    	cout << argv[a] << endl;
//    }

	// read in parameters
    int vertNum = atoi(argv[1]);

    char* sMeasure = "rolesim";
    
    //char* sSimOutFilename = "similarity.txt";
    //char* sGraphFilename = "graph.txt";

    char* sGraphFilename = argv[2];
    char* sSimOutFilename = argv[3];

    cout << sGraphFilename << " " << sSimOutFilename << endl;
    //list<char*> lGraphFiles;
    //std::copy(&argv[nextOptIndex+2], &argv[argc], std::inserter(lGraphFiles, lGraphFiles.begin()));

    // graph data structure
    //int vertNum = 0;
    
    list<int> vSrc;
    list<int> vTar;
    
    // read in the files
    readGraph(sGraphFilename, vertNum, vSrc, vTar);

    // number of vertices
//	int vertNum = vUniqueVerts.size();

    
    IterSimilarity* pfSim = NULL;
    
    if (g_bUseConvEpsilon) {
    	pfSim = new RoleSim(g_dampingFactor, g_iterInfo, g_convEpsilon, g_initAlgorName);
    }
    else {
    	pfSim = new RoleSim(g_dampingFactor, g_iterInfo, g_initAlgorName);
    }
    

    // construct distances
//    double* mSim = roleSim2(vSrc, vTar, vSrc.size(), vertNum, iterNum, dampingFactor);
    //assert(pfSim != NULL);

    float* mSim = pfSim->computeSim(vSrc, vTar, vSrc.size(), vertNum);


    // output
    ofstream fOut(sSimOutFilename);

	for (int i = 0; i < vertNum; ++i) {
		for (int j = 0; j < vertNum-1; ++j) {
			fOut << mSim[i + j*vertNum] << ",";
		}
		fOut << mSim[i + (vertNum-1) * vertNum] << endl;
	}

    fOut.close();

    delete pfSim;

    // release memory from roleSim2()
    delete[] mSim;
} // end of main()


void readGraph(const char* sGraphFilename, int& vertNum, list<int>& vSrc, list<int>& vTar)
{
	//vertNum = 77;
    ifstream fIn(sGraphFilename);

    char line[1024]={0};  
    string s = "";
    string t = "";

    while(fIn.getline(line, sizeof(line)))  
    {  
        stringstream word(line);  
        word >> s;
        word >> t;

        int src = atoi(s.c_str());
        int tar = atoi(t.c_str());

        vSrc.push_back(src);
        vTar.push_back(tar);
    }

    fIn.close();
}
