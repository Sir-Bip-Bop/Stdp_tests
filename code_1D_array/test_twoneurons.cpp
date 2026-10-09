// two_neuron_sim.cpp
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

int main(int argc, char* argv[]){
    // Seed RNG
    srand(1053416);

    // Network parameters
    double rwStep = 0.01;
    double refractoryT = 200;
    double amplitudeSTDP = 10000.;
    double sigmaSTDP = 100.;
    double delay = 1.0;

    // Two neurons only
    int nNeurons = 2;

    // Simulation parameters
    int t, tMax = 10000;
    int tSignal = 1000;
    int nRepeats = 1; // single run by default

    // Output file
    ofstream fOut("weights_summary.csv");
    fOut << "iNeuron,jNeuron,weight\n";

    for (int rep = 0; rep < nRepeats; ++rep) {
        // Construct a fresh network of 2 neurons (fully connected except self-connections)
        rwN::RWNetwork network(
            nNeurons,
            rwStep,
            refractoryT,
            amplitudeSTDP,
            sigmaSTDP,
            delay
        );
    for (int i = 0; i < nNeurons; ++i) {
    // deliver immediate external spike of bias 1.0
    network.getNeuron(i).receiveExternalSpike(0.0, 1.0); // or at time t when needed
    }

        // Run simulation
        for (t = 0; t < tMax; ++t) {
            network.update(t);

            // Inject external spike every tSignal steps (keeps network active)
            if (t % tSignal == 0) {
for (int i = 0; i < nNeurons; ++i) {
    network.getNeuron(i).receiveExternalSpike(t, 1.0);
}
            }
        }

        // Get final weight matrix and write to CSV
        double** W = network.getWMatrix();
        for (int i = 0; i < nNeurons; ++i) {
            for (int j = 0; j < nNeurons; ++j) {
                fOut << i << "," << j << "," << W[i][j] << "\n";
            }
        }

        // If helper::summarizeW is preferred, you can also do:
        // double* wSummary = h::summarizeW(nNeurons, W);
        // write wSummary values as needed.
    }

    fOut.close();
    cout << "Summary written to weights_summary.csv\n";
    return 0;
}
