/*

	helper.h: 

		This library implements functions to aid in processing the exhuberant barrels experiments. 

*/


// Imports: 
#include <iostream> 
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <cstdlib> 
#include <vector>
#include <random>
#include <algorithm>
using namespace std; 
// std::default_random_engine kMapGenerator; 

#define _USE_MATH_DEFINES

namespace h{


	/*

		helper library namespace: 

	*/

	double *summarizeWBlock(int nNeurons, double **w, bool fSelf){
		/* summarizeWBlock function: 

			This function extracts the mean and standard deviation of a matrix of connections. This function is intended
			to analyze one of the 4 block matrices that make up a connectivity matrix at large. The variable
			nNeurons indicates the number of neurons within this block. The variable fSelf indicates whether these
			are connections within a same barrel or from a barrel to the other (which matters because in the same
			case the diagonal should not be taken into account). 

			Inputs: 
				>> nNeurons: Number of neurons within the provided block of connections. 
				>> w: Connections from which we wish to obtain the mean and standard variation. 
				>> fSelf: Flag indicating whether the block corresponds to connections within a barrel, or from a barrel
				   to the other. 

			Returns: 
				<< wSummary: Array of dimension 2 with the mean and standard deviation of the connections. 

		*/

		// Init variables: 
		double *wSummary; 
		wSummary = new double[2]; 
		int Z; 
		Z = pow(nNeurons, 2); 
		if (fSelf) Z -= nNeurons; 

		wSummary[0] = 0.; 
		for (int i=0; i<nNeurons; i++){
			if (not(fSelf)) wSummary[0] += w[i][i]; 
			for (int j=i+1; j<nNeurons; j++){
				wSummary[0] += w[i][j] + w[j][i]; 
			}
		}
		wSummary[0] /= Z; 

		wSummary[1] = 0.; 
		for (int i=0; i<nNeurons; i++){
			if (not(fSelf)) wSummary[1] += pow(w[i][i] - wSummary[0], 2); 
			for (int j=i+1; j<nNeurons; j++){
				wSummary[1] += pow(w[i][j] - wSummary[0], 2) + pow(w[j][i] - wSummary[0], 2); 
			}
		}
		wSummary[1] /= Z; 
		wSummary[1] = sqrt(wSummary[1]); 

		return wSummary; 
	}

	double *summarizeW(int nNeurons, double **w){
		/* summarizeW function: 

			This function summarizes (i.e. returns means and standard deviations for) a whole matrix of connections.
			This matrix consists of 4 blocks that contain connections within a same barrel and from a barrel to the
			other. Statistics are computed and returned for each block separatedly. 

			Inputs: 
				>> nNeurons: Number of neurons in the system at large. 
				>> w: Connections from which we wish to obtain the mean and standard variation. 

			Returns: 
				<< wSummary: Array containing summaries for each block of connections. 
					< wSummary[0]: mean L-->L connections. 
					< wSummary[1]: mean L-->R connections. 
					< wSummary[2]: mean R-->L connections. 
					< wSummary[3]: mean R-->R connections. 
					< wSummary[4]: std L-->L connections. 
					< wSummary[5]: std L-->R connections. 
					< wSummary[6]: std R-->L connections. 
					< wSummary[7]: std R-->R connections. 

		*/

		int nBlock=nNeurons/2; 
		double *wSummary, *dummySummary, **wBlockLL, **wBlockLR, **wBlockRL, **wBlockRR; 

		// Initializing block matrices: 
		wBlockLL = new double*[nBlock]; 
		wBlockLR = new double*[nBlock]; 
		wBlockRL = new double*[nBlock]; 
		wBlockRR = new double*[nBlock]; 
		for (int i=0; i<nBlock; i++){
			wBlockLL[i] = new double[nBlock]; 
			wBlockLR[i] = new double[nBlock]; 
			wBlockRL[i] = new double[nBlock]; 
			wBlockRR[i] = new double[nBlock]; 
		}

		// Reading block matrices off of connectivity matrix: 
		for (int i=0; i<nBlock; i++){
			wBlockLL[i][i] = w[i][i]; 
			wBlockLR[i][i] = w[i][i+nBlock]; 
			wBlockRL[i][i] = w[i+nBlock][i]; 
			wBlockRR[i][i] = w[i+nBlock][i+nBlock]; 
			for (int j=i+1; j<nBlock; j++){
				// LL: 
				wBlockLL[i][j] = w[i][j]; 
				wBlockLL[j][i] = w[j][i]; 
				// LR: 
				wBlockLR[i][j] = w[i][j+nBlock]; 
				wBlockLR[j][i] = w[j][i+nBlock]; 
				// RL: 
				wBlockRL[i][j] = w[i+nBlock][j]; 
				wBlockRL[j][i] = w[j+nBlock][i]; 
				// RR: 
				wBlockRR[i][j] = w[i+nBlock][j+nBlock]; 
				wBlockRR[j][i] = w[j+nBlock][i+nBlock]; 				
			}
		}

		// Computing summaries: 
		wSummary = new double[8]; 
		dummySummary = summarizeWBlock(nBlock, wBlockLL, true); 
		wSummary[0] = dummySummary[0]; 
		wSummary[4] = dummySummary[1]; 
		dummySummary = summarizeWBlock(nBlock, wBlockLR, false); 
		wSummary[1] = dummySummary[0]; 
		wSummary[5] = dummySummary[1]; 
		dummySummary = summarizeWBlock(nBlock, wBlockRL, false); 
		wSummary[2] = dummySummary[0]; 
		wSummary[6] = dummySummary[1]; 
		dummySummary = summarizeWBlock(nBlock, wBlockRR, true); 
		wSummary[3] = dummySummary[0]; 
		wSummary[7] = dummySummary[1]; 

		return wSummary; 
	}

}
