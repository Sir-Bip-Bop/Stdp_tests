/*

	script_varySigmaExternalDelay.cpp: 

		This experiments varies the variance of external delays. 

*/


// Imports: 
#include <iostream> 
#include <fstream>
#include <string>
#include <math.h>
#include <cstdlib> 
#include <vector>
#include <sstream>
#include "rwNeuron.h"
#include "helper.h"
using namespace std; 

#include <chrono> 
using namespace std::chrono; 

#include <stdlib.h>     /* srand, rand */



int main(int argc, char* argv[]){
	/* main routine: */ 

	// Seed for random numbers: 
	srand(23498764);

	// Defining variables and loading paths: 	
	double rwStep = 0.01, refractoryT = 200, amplitudeSTDP=10000., sigmaSTDP=100.; 

	// Variables for barrels: 
	int nNeuronsPerBarrel=10, nNeurons; 
	nNeurons = 2*nNeuronsPerBarrel; 
	double shortDelay=1, longDelay=100, sigmaExternalDelay; 
	rwN::RWNetwork network; 

	// Variables for experiment: 
	int t, tMax=10000, tSignal=1000, nRepeats=100; 
	double **wMatrix, *wSummary, sigmaExternalDelayMin=10, sigmaExternalDelayMax=200; 


	// I/O variables: 
	ostringstream fOutName;
	ofstream fOutSummary; 


	// Loop over longDelays (i.e. distances between barrels): 
	for (sigmaExternalDelay=sigmaExternalDelayMin; sigmaExternalDelay<=sigmaExternalDelayMax; sigmaExternalDelay += 10){

		cout << "Sigma external delay: " << sigmaExternalDelay << endl; 

		// Opening summary file: 
		fOutName.str(""); 
		fOutName << "./Experiment_ExternalDelay/summary_" << sigmaExternalDelay << ".csv"; 
		fOutSummary.open(fOutName.str().c_str()); 

		for (int iRepeat=0; iRepeat<nRepeats; iRepeat++){
			cout << "\tRepeat: " << iRepeat << endl; 

			// Initializing new network: 
			network.initL4Barrels(nNeuronsPerBarrel, rwStep, refractoryT, amplitudeSTDP, sigmaSTDP, shortDelay, longDelay, sigmaExternalDelay); 

			// Running network dynamics: 
			for (t=0; t<tMax; t++){
				network.update(t); 
				if (not(t%tSignal)){
					network.deliverExternalSpike(t); 
				}
			}

			// Computing experiment summaries: 
			wMatrix = network.getWMatrix(); 
			wSummary = h::summarizeW(nNeurons, wMatrix); 
			for (int i=0; i<8; i++){
				fOutSummary << wSummary[i]; 
				if (i<7) fOutSummary << ", "; 
				else fOutSummary << endl; 
			}
		}

		fOutSummary.close(); 
	}

	return 0;
}
