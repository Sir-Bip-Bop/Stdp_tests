// test_2d_grid.cpp
//
// Same analysis pipeline as test_1d_network.cpp (multi-source Gaussian-pulse external
// stimulation, periodic weight-matrix snapshots, initial/final weight logging), but on a
// 2D grid of neurons instead of a 1D chain. See rwN::RWNetwork::init2DGrid() in rwNeuron.h
// for the topology: neuron index n = row*nCols + col, with delays given by Euclidean
// distance / velocity between grid points.
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

// See test_1d_network.cpp for full documentation of this function -- unchanged here.
void deliverGaussianPulse(rwN::RWNetwork &network, int targetNeuron, double centerTime,
                           double sigma, double windowSigmas, int peakSpikes,
                           double spikeBias, ofstream &stimLog){

    int halfWidth = (int) ceil(windowSigmas * sigma);

    for (int dt = -halfWidth; dt <= halfWidth; ++dt){
        double t = centerTime + dt;
        if (t < 0) continue;

        double gaussVal = exp(-(double(dt)*dt) / (2.0*sigma*sigma));
        int nSpikesAtT = (int) llround(peakSpikes * gaussVal);

        for (int k = 0; k < nSpikesAtT; ++k){
            network.getNeuron(targetNeuron).receiveExternalSpike(t, spikeBias);
            stimLog << (int)t << "," << targetNeuron << "," << (int)centerTime << "," << dt << "\n";
        }
    }

    return;
}

vector<int> parseIntList(const string &s){
    vector<int> result;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')){
        if (item.size()) result.push_back(atoi(item.c_str()));
    }
    return result;
}


int main(int argc, char* argv[]){
    // -------------------------
    // Grid / simulation parameters
    // -------------------------
    const int nRows = 4;
    const int nCols = 5;
    const int nNeurons = nRows * nCols;   // 20 neurons, same total as the 1D array
    const double rwStep = 0.01;
    const double refractoryT = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP = 100.;
    const double spacing = 1.0;
    const double velocity = 1.0;

    int tMax = (argc > 2) ? atoi(argv[2]) : 8000;
    const int weightSnapshotEvery = 50;

    // -------------------------
    // Gaussian external stimulation pulse parameters
    // -------------------------
    // Default targets are the 4 grid corners (row-major indices): (0,0)=0, (0,4)=4,
    // (3,0)=15, (3,4)=19 -- spread out spatially so their stimulation waves interact
    // across the grid instead of just along a line.
    vector<int> targetNeurons;
    if (argc > 1) targetNeurons = parseIntList(argv[1]);
    if (targetNeurons.empty()) targetNeurons = {0, 4, 15, 19};

    const int    pulsePeriod       = 50;
    const double pulseSigma        = 5.0;
    const double pulseWindowSigmas = 3.0;
    const int    pulsePeakSpikes   = 6;
    const double externalSpikeBias = 0.12;
    int pulseStaggerSteps = (argc > 3) ? atoi(argv[3]) : 15;

    for (int tn : targetNeurons){
        if (tn < 0 || tn >= nNeurons){
            cerr << "targetNeuron " << tn << " out of range [0, " << nNeurons-1 << "]\n";
            return 1;
        }
    }

    srand(1053416);

    // -------------------------
    // Create network
    // -------------------------
    rwN::RWNetwork network;
    network.init2DGrid(
        nRows, nCols,
        rwStep,
        refractoryT,
        amplitudeSTDP,
        sigmaSTDP,
        spacing,
        velocity
    );

    // -------------------------
    // Log neuron grid positions explicitly, so downstream Python plotting doesn't have to
    // re-derive the row-major indexing convention -- it just reads (row, col, x, y) per neuron.
    // -------------------------
    ofstream posOut("output_2D/positions.csv");
    posOut << "neuron,row,col,x,y\n";
    for (int n = 0; n < nNeurons; ++n){
        int row = n / nCols;
        int col = n % nCols;
        posOut << n << "," << row << "," << col << "," << (col*spacing) << "," << (row*spacing) << "\n";
    }
    posOut.close();

    ofstream spikeLog("output_2D/spikes.csv");
    spikeLog << "time,neuron\n";

    ofstream stimLog("output_2D/stimulation.csv");
    stimLog << "time,neuron,pulseCenter,offset\n";

    ofstream weightEvoLog("output_2D/weights_evolution.csv");
    weightEvoLog << "time,iNeuron,jNeuron,weight\n";

    // Explicit initial (pre-STDP) weight matrix snapshot
    {
        double** Winit = network.getWMatrix();
        ofstream wInitOut("output_2D/weights_initial.csv");
        wInitOut << "iNeuron,jNeuron,weight\n";
        for (int i = 0; i < nNeurons; ++i){
            for (int j = 0; j < nNeurons; ++j){
                wInitOut << i << "," << j << "," << Winit[i][j] << "\n";
            }
        }
        wInitOut.close();
        // NOTE: not manually freed -- see the double-free note further down.
    }

    // Schedule all Gaussian stimulation pulses up front (one independent, optionally
    // staggered pulse train per target neuron).
    for (size_t ti = 0; ti < targetNeurons.size(); ++ti){
        int targetNeuron = targetNeurons[ti];
        int phase = (int)ti * pulseStaggerSteps;
        for (int centerTime = phase; centerTime < tMax; centerTime += pulsePeriod){
            deliverGaussianPulse(network, targetNeuron, centerTime, pulseSigma, pulseWindowSigmas,
                                  pulsePeakSpikes, externalSpikeBias, stimLog);
        }
    }

    // -------------------------
    // Simulation loop
    // -------------------------
    for (int t = 0; t < tMax; ++t) {
        network.update(t);

        for (int i = 0; i < nNeurons; ++i) {
            if (network.getNeuron(i).getHasSpiked()) {
                spikeLog << t << "," << i << "\n";
            }
        }

        // NOTE: getWMatrix() stores its allocation in an internal member pointer that the
        // RWNetwork destructor also frees at the end; we deliberately don't free intermediate
        // snapshots here to avoid a double-free (see test_1d_network.cpp for the full note).
        if (t % weightSnapshotEvery == 0) {
            double** Wsnap = network.getWMatrix();
            for (int i = 0; i < nNeurons; ++i) {
                for (int j = 0; j < nNeurons; ++j) {
                    if (i == j) continue;
                    weightEvoLog << t << "," << i << "," << j << "," << Wsnap[i][j] << "\n";
                }
            }
        }

        if (t % 1000 == 0) {
            cout << "t=" << t << " / " << tMax << "\n";
        }
    }
    spikeLog.close();
    stimLog.close();
    weightEvoLog.close();

    // -------------------------
    // Final weight matrix
    // -------------------------
    double** W = network.getWMatrix();
    ofstream wOut("output_2D/weights_summary.csv");
    wOut << "iNeuron,jNeuron,weight\n";
    for (int i = 0; i < nNeurons; ++i) {
        for (int j = 0; j < nNeurons; ++j) {
            wOut << i << "," << j << "," << W[i][j] << "\n";
        }
    }
    wOut.close();
    // NOTE: W is not manually freed -- freed once by RWNetwork's destructor at scope exit.

    cout << "Simulation finished. Grid: " << nRows << "x" << nCols << ". Target neurons: ";
    for (int tn : targetNeurons) cout << tn << " ";
    cout << "\nOutputs in output_2D/: positions.csv, spikes.csv, stimulation.csv, "
         << "weights_initial.csv, weights_evolution.csv, weights_summary.csv\n";
    return 0;
}