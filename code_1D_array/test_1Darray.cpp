// test_1d_array.cpp
// 1D_array of neurons, connected to each other separated by a constant spacing (modifiable). 
// Neuron 0 receives a single external spike, and it propagates throughout the networ
// The simplest case of all these experiments

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

int main(int argc, char* argv[]){
    // -------------------------
    // Simulation parameters
    // -------------------------
    const int nNeurons = 100;        // array size
    const double rwStep = 0.01;
    const double refractoryT = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP = 100.;
    const double spacing = 1.0;       // distance between adjacent neurons (spatial units)
    const double velocity = 1.0;      // conduction velocity (spatial units / time unit)
    const int tMax = 2000;            // simulation timesteps
    const int externalEvery = 50;     // inject external spikes every 'externalEvery' timesteps
    const double externalBias = 0.5;  // bias added by external spike
    const int weightSnapshotEvery = 10; //Timestep interval to snaphsot the full weight matrix
    
    srand(1053416); // Seed RNG

    // -------------------------
    // Create network
    // -------------------------
    rwN::RWNetwork network;
    network.init1DArray(
        nNeurons,
        rwStep,
        refractoryT,
        amplitudeSTDP,
        sigmaSTDP,
        spacing,
        velocity
    );

    // Open spike log file
    ofstream spikeLog("output_1D/spikes.csv");
    spikeLog << "time,neuron\n";

    // Log of external stimulation events (which neuron, at which time)
    ofstream stimLog("output_1D/stimulation.csv");
    stimLog << "time,neuron\n";

    // Log of weight matrix snapshots over time, long format
    ofstream weightEvoLog("output_1D/weights_evolution.csv");
    weightEvoLog << "time,iNeuron,jNeuron,weight\n";

    // -------------------------
    // Simulation loop
    // -------------------------
    for (int t = 0; t < tMax; ++t) {
        // Inject external spikes to selected neurons at regular intervals
        if (t % externalEvery == 0) {
            for (int i = 0; i < nNeurons; ++i) {
                // deliver an external spike arriving at time t with bias externalBias
                // use receiveExternalSpike to avoid relying on externalW/externalDelays
                if (i == 0){
                    network.getNeuron(i).receiveExternalSpike(t, externalBias);
                    stimLog << t << "," << i << "\n";
                }
            }
        }

        // Update network (neurons update their v and schedule outgoing spikes)
        network.update(t);

        // Log spikes that occurred at this timestep
        for (int i = 0; i < nNeurons; ++i) {
            if (network.getNeuron(i).getHasSpiked()) {
                cout << "t=" << t << " neuron=" << i << " SPIKED\n";
                spikeLog << t << "," << i << "\n";
            }
        }

        // Periodically snapshot the weight matrix
        if (t % weightSnapshotEvery == 0) {
            double** Wsnap = network.getWMatrix();
            for (int i = 0; i < nNeurons; ++i) {
                for (int j = 0; j < nNeurons; ++j) {
                    if (i == j) continue; // no self-connections
                    weightEvoLog << t << "," << i << "," << j << "," << Wsnap[i][j] << "\n";
                }
            }
        }
    }
    spikeLog.close();
    stimLog.close();
    weightEvoLog.close();

    // -------------------------
    // Get final weight matrix and write CSV
    // -------------------------
    double** W = network.getWMatrix(); // allocates nNeurons x nNeurons matrix
    ofstream wOut("output_1D/weights_summary.csv");
    wOut << "iNeuron,jNeuron,weight\n";
    for (int i = 0; i < nNeurons; ++i) {
        for (int j = 0; j < nNeurons; ++j) {
            wOut << i << "," << j << "," << W[i][j] << "\n";
        }
    }
    wOut.close();

    cout << "Simulation finished. Spikes -> spikes.csv, stimulation -> stimulation.csv, "
         << "weight evolution -> weights_evolution.csv, final weights -> weights_summary.csv\n";
    return 0;
}