/*

	script_varySignalToRightBarrel.cpp: 

		This script runs a simple experiment in which left- and right-barrel neurons receive inputs of different
		intensity. We vary this intensity to check at what level we lose intercallosal connections. 

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
	srand(324582707);

	// Defining variables and loading paths: 	
	double rwStep = 0.01, refractoryT = 200, amplitudeSTDP=10000., sigmaSTDP=100., delays=10, *efferentDelays; 

	// Variables for barrels: 
	int nNeuronsPerBarrel=10, nNeurons; 
	nNeurons = 2*nNeuronsPerBarrel; 
	double shortDelay=1, longDelay=100, sigmaExternalDelay=10; 
	rwN::RWNetwork network; 

	// Variables for experiment: 
	int t, tMax=10000, tSignal=1000, nRepeats=100; 
	double leftExternalBias=1., rightExternalBias, **wMatrix, *wSummary; 
	vector<int> leftBarrelNeurons, rightBarrelNeurons; 
	leftBarrelNeurons.clear(); 
	rightBarrelNeurons.clear(); 
	for (int i=0; i<nNeuronsPerBarrel; i++){
		leftBarrelNeurons.push_back(i); 
		rightBarrelNeurons.push_back(i + nNeuronsPerBarrel); 
	}


	// I/O variables: 
	ostringstream fOutName;
	ofstream fOutSummary; 



	// Loop over right externaal bias: 
	double extBiasStep=0.01; 
	for (rightExternalBias=0.; rightExternalBias<(1. + extBiasStep); rightExternalBias+=extBiasStep){
		cout << "External bias: " << rightExternalBias << endl; 

		// Opening summary file: 
		fOutName.str(""); 
		fOutName << "./Experiment_AsymmetricSignal/summary_" << rightExternalBias << ".csv"; 
		//cout << "\tOutput file: " << fOutName.str().c_str() << endl;
		fOutSummary.open(fOutName.str().c_str()); 

		for (int iRepeat=0; iRepeat<nRepeats; iRepeat++){
			cout << "\tRepeat: " << iRepeat << endl; 

			// Initializing new network: 
			network.initL4Barrels(nNeuronsPerBarrel, rwStep, refractoryT, amplitudeSTDP, sigmaSTDP, shortDelay, longDelay, sigmaExternalDelay); 

			// Running network dynamics: 
			for (t=0; t<tMax; t++){
				network.update(t); 
				if (not(t%tSignal)){
					network.deliverExternalSpike(t, leftExternalBias, leftBarrelNeurons); 
					network.deliverExternalSpike(t, rightExternalBias, rightBarrelNeurons); 
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
