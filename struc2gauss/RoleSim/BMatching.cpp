#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>


typedef std::vector<int> INT_VECTOR;
typedef std::vector<float> FLOAT_VECTOR;
typedef std::vector<double> DOUBLE_VECTOR;

/**
 * Maximum matching algorithm for continuous costs.
 */
double matching(int n, int m, const std::vector<float>& vA, std::vector<int>& m1, std::vector<int>& m2)
{
    using namespace std;
    
	vector <double>nzv;
	vector <int>nzi;
	vector <int>nzj;
    
//	cout << "vA" << endl;
    // convert to sparse vector form
	for (int j=0;j<m; ++j)
	{
		for (int i=0;i<n; ++i)
		{
            // column major
            double tempVal = vA[i + j*n];
//            cout << vA[i + j*n] << ", ";
			//if ((*((double*)A+m*i+j))!=0)
            if (tempVal > 0)
			{
// 				nzv.push_back(*((double*)A+m*i+j));
                nzv.push_back(tempVal);
				nzi.push_back(i);
				nzj.push_back(j);
			}
		}
//		cout << endl;
	}

	int nedges=nzi.size();
//	cout << nedges << endl;
    
//     int  *rp=new int[n+1];
// 	for (int i=0; i<n+1;i++)
// 	{
// 		rp[i]=1;
// 	
// 	}        
    INT_VECTOR rp = INT_VECTOR(n+1, 1);

//     int  *ci=new int[nedges+n];
//     double  *ai=new double[nedges+n];
// 	for (int i=0; i < nedges+n; i++)
// 	{
// 		ci[i]=0;
//         ai[i] = 0;
// 	}
    
    INT_VECTOR ci= INT_VECTOR(nedges+n, 0);
    DOUBLE_VECTOR ai = DOUBLE_VECTOR(nedges+n, 0);
// 	for (int i=0; i < nedges+n; i++)
// 	{
// 		ci[i]=0;
//         ai[i] = 0;
// 	}    
    
    

	rp[0]=0;
	//matlab rp[1]
	// count the number of elements in each row
    for(int i=0; i<nedges;i++)
	{
		rp[nzi[i]+1]=rp[nzi[i]+1]+1;
	
    }

    // compute the cumulative sum of each row
    int cumsum=rp[0];
	for (int i=1; i<n+1; ++i)
	{
		rp[i]=cumsum+rp[i];
		cumsum=rp[i];
		//cout<<rp[i]<<"  "<<cumsum<<endl;
	}
    
    for(int i=0; i<nedges; ++i)
	{
		ai[rp[nzi[i]]] = nzv[i];
		ci[rp[nzi[i]]] = nzj[i];
		rp[nzi[i]] = rp[nzi[i]]+1;
    }	
    // in matlab ai[rp[nzi[i]]-1]=nzv[i]

    for(int i=0;i<n; ++i)
    {
        ai[rp[i]]=0;
        ci[rp[i]]=m+i;
        rp[i]=rp[i]+1;
    }

    for(int i=n-1;i>=0; --i)
        // in matlab from n to 1.  in c++ from n-1 to 0
    {
        rp[i+1]=rp[i];
    }
    rp[0]=0;
    //in matlab rp[1]
    for (int i=0; i<n+1; ++i)
        // the size of rp is n+1
    {
    rp[i]=rp[i]+1;
    }

    // check for duplicates
//    vector<bool> colind(m+n, false);
//    for (int i = 0; i < n; ++i) {
//    	for (int rpi = rp[i]; rpi <= rp[i+1]-1; ++rpi) {
//    		if (colind[ci[rpi]]) {
//    			cerr << "duplicate Edge detected (" << i << "," << ci[rpi] << ")" << endl;
//    			exit(1);
//    		}
//    	}
//    }


//     double  *alpha=new double[n];
// 	for (int i=0; i<n;i++)
// 	{
// 		alpha[i]=0;
// 	}
    
    DOUBLE_VECTOR alpha = DOUBLE_VECTOR(n, 0);    

//     int *tmod=new int[n+m];
//     for(int i=0;i<n+m;i++)
//     {
//         tmod[i]=0;
//     }
    
    INT_VECTOR tmod = INT_VECTOR(n+m, 0);      
    
//     int *t=new int[n+m];
//     for(int i=0;i<n+m;i++)
//     {
//         t[i]=-1;
//     }
    
    INT_VECTOR t = INT_VECTOR(n+m, -1);       
    
//     double *beta= new double[n+m];
//     for(int i=0;i<n+m;i++)
// 	{
//         beta[i]=0;
// 	}
    
    DOUBLE_VECTOR beta = DOUBLE_VECTOR(n+m, 0);   
    
//     int *match1= new int[n];
//     for(int i=0;i<n;i++)
// 	{
// 		match1[i]=-1;
// 	}
    
    INT_VECTOR match1 = INT_VECTOR(n, -1);       
    
// 	int *match2=new int[n+m];
// 	for(int i=0;i<n+m;i++)
// 	{
// 		match2[i]=-1;
// 	}
    
    INT_VECTOR match2 = INT_VECTOR(n+m, -1);       
    
    int ntmod=-1;
    
//     int *queue=new int[n];
//     for(int i=0;i<n;i++)
//     {
//         queue[i]=0;
//     }  
    
    INT_VECTOR queue = INT_VECTOR(n+1, 0);
    
    for(int i=0;i<n;i++)
    {
        for (int rpi=(rp[i]-1);rpi<(rp[i+1]-1);rpi++)
        {
            if(ai[rpi]>alpha[i])
            {
                alpha[i]=ai[rpi];
            }
        }
    }

    // main matchin loop
    int ii=0;
    //in matlab this is i
    while (ii < n)
//    for (int ii = 0; ii < n; ++ii)
    {
    	assert(ntmod < n+m);
        for(int j=0;j<=ntmod;j++)
        {
            t[tmod[j]]=-1;	
        }

        ntmod=-1;
        int head=0;
        int tail=0;
        queue[head]=ii;
        
//        cout << "here ii = " << ii << endl;
        while(head<=tail && match1[ii] == -1)
        {
            int k=queue[head];
            for (int rpi = rp[k]-1; rpi < rp[k+1]-1; rpi++) {
                int jj = ci[rpi];
                    //in matlab this is j
                if(ai[rpi] < alpha[k] + beta[jj] - 1e-8) {
                        continue;
                }

                if (t[jj] == -1)
                {
                        tail = tail+1;
//                        cout << "tail = " << tail << ", jj = " << jj << ", n = " << n << ", m = " << m << endl;;
                        queue[tail] = match2[jj];
                        t[jj] = k;
                        ntmod=ntmod+1;
                        tmod[ntmod] = jj;
                        if (match2[jj] < 0)
                        {
                            while (jj>-1)
                            {
                                match2[jj] = t[jj];
                                k=t[jj];
                                int temp=match1[k];
                                match1[k]=jj;
                                jj=temp;
                            }
                            break;
                        }
                }
            }
            head=head+1;
        } // end of while
	
        if (match1[ii]<0)
        {
            #define inf 2147483647;
            double theta=inf;
            for(int j=0;j<=head-1;j++)
            {
                int t1=queue[j];
                for (int rpi=rp[t1]-1; rpi < rp[t1+1]-1; rpi++)
                {
                    int t2= ci[rpi];
                    if (t[t2] == -1 && alpha[t1] + beta[t2] - ai[rpi] < theta)
                    {
                        theta = alpha[t1] + beta[t2] - ai[rpi];
                    }
                }
            }
            for (int j=0;j<=head-1;j++)
            {
                alpha[queue[j]] = alpha[queue[j]] - theta;
            }
            for (int j=0;j<=ntmod;j++)
            {
                beta[tmod[j]] = beta[tmod[j]] + theta;
            }
            continue;
        } // end of if
        ii=ii+1;
    } // end of outer while

	double val=0;

    for(int i=0;i<n;i++)
    {
        for(int rpi=rp[i]-1;rpi<rp[i+1]-1;rpi++)
        {
        //	cout<<rpi<<endl;
            if(ci[rpi]==match1[i])
            {
                val=val+ai[rpi];
            }
        }
    }

    
    // only copy to m1 and m2 if they are not NULL
//    if (m1 != NULL && m2 != NULL) {
        int noute=0;
        for(int i=0;i<n;i++)
        {
            if(match1[i] < m)
            {
                noute=noute+1;
            }
        }

    //     int *m1= new int[noute];
        //     int *m2=new int[noute];
        m1.resize(noute, 0);
        m2.resize(noute, 0);
//        for(int i=0;i<noute;i++)
//        {
//            m1[i]=0;
//            m2[i]=0;
//        }

    //     for(int i=0;i<noute;i++)
    //     {
    //         m2[i]=0;
    //     }

        // we use the smaller of m and n to set noute
        noute=0;
        for(int i=0;i<n;i++)
        {
            if (match1[i] < m)
            {
                m1[noute]=i;
    //             m2[noute]=match1[i]-1;
                m2[noute]=match1[i];
                noute=noute+1;
            }
        }
//    }

       
//     cout << "m1" << endl;
//     for (int i = 0; i < noute; ++i) {
//         cout << m1[i] << endl;
//     }
//     
// 
//     cout << "m2" << endl;
//     for (int i = 0; i < noute; ++i) {
//         cout << m2[i] << endl;
//     }    
    
    // free dynamically allocated memory
//     cout << "freeing memory queue" << endl;
//     delete[] queue;
//     cout << "freeing memory match2" << endl;
//     delete[] match2;
//     cout << "freeing memory match1" << endl;
//     delete[] match1;
//     cout << "freeing memory beta" << endl;
//     delete[] beta;
//     cout << "freeing memory t" << endl;
//     delete[] t;
//     cout << "freeing memory tmod" << endl;
//     delete[] tmod;
//     cout << "freeing memory alpha" << endl;
//     delete[] alpha;
//     cout << "freeing memory ai" << endl;
//     delete[] ai;
//     cout << "freeing memory ci" << endl;
//     delete[] ci;
//     cout << "freeing memory rp" << endl;
//     delete[] rp;

    return val;
} // end of function



/** 
 * Convert matlab matrix to a vector.
 * Inefficient.
 */
// void arrayToMatrix(const double* vMat, double **mCost, int rows, int cols) 
// {
//     int i,j;
//     
//     
//     for(j=0;j<cols;j++) {
//         for(i=0;i<rows;i++)
//             mCost[i][j] = vMat[i + j * rows];
//     }
// 
// } // end of array_to_matrix()







// 	int main()
// 	{
// 		int row=100;
// 		int col=100;
//         
//         double* m = new double[row*col];
//         for (int r = 0; r < row; ++r) {
//             for (int c = 0; c < col; ++c) {
//                 m[r + c * row] = static_cast<double>(rand() % 1000) / 1000;
//             }
//         }
//         
//         
// // 		double m[]={0.2,0.3,0.4,0.1,0.6,0.5,0.7,0.9,0.2,0.3,0.4,0.7,0.3,0.4,0.6,0.1,0.3,0.4,0.4,0.5};
// 		
//         int* m1 = new int[row];
//         int* m2 = new int[col];
//         
//         cout << matching(row,col,m, m1, m2) << endl;;
//         
//         delete[] m1;
//         delete[] m2;
//         delete[] m;
// 		
// 	}

