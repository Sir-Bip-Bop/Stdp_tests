// test_1d_network.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

// -------------------------
// deliverGaussianPulse:
//
//   Delivers a burst of external "micro-spikes" to targetNeuron, shaped like a Gaussian in
//   time around centerTime: few micro-spikes in the early tail, a peak of micro-spikes right
//   at centerTime, then fewer again in the late tail. This replaces a single discrete external
//   spike with a small spike volley, so that -- relative to a postsynaptic neuron's own spike
//   time -- part of the pulse can land "early" (large negative deltaT) and part can land "late"
//   (small negative / positive deltaT), letting STDP act differently on different parts of the
//   same pulse.
//
//   All micro-spikes are scheduled up front via receiveExternalSpike(), which just inserts them
//   into the neuron's internal spike queue; they are actually applied later during network.update(t)
//   once the simulation clock reaches each spike's time. So this can safely be called before the
//   main simulation loop starts.
//
//   Inputs:
//     >> network:       The network to stimulate.
//     >> targetNeuron:  Index of the neuron that receives the pulse.
//     >> centerTime:    Timestep at which the pulse is centered (its peak).
//     >> sigma:         Standard deviation (in timesteps) of the Gaussian pulse shape.
//     >> windowSigmas:  Truncate the pulse at +/- windowSigmas * sigma from the center.
//     >> peakSpikes:    Number of micro-spikes delivered exactly at centerTime (the peak of
//                        the Gaussian); the count at other offsets is peakSpikes * gaussian(dt).
//     >> spikeBias:     Bias contributed by EACH individual micro-spike.
//     >> stimLog:       Stream to log every scheduled micro-spike to, for later analysis/plotting.
//                        Columns: time,neuron,pulseCenter,offsetFromCenter
// -------------------------
void deliverGaussianPulse(rwN::RWNetwork &network, int targetNeuron, double centerTime,
                           double sigma, double windowSigmas, int peakSpikes,
                           double spikeBias, ofstream &stimLog){

    int halfWidth = (int) ceil(windowSigmas * sigma);

    for (int dt = -halfWidth; dt <= halfWidth; ++dt){
        double t = centerTime + dt;
        if (t < 0) continue; // no negative simulation times

        double gaussVal = exp(-(double(dt)*dt) / (2.0*sigma*sigma));
        int nSpikesAtT = (int) llround(peakSpikes * gaussVal);

        for (int k = 0; k < nSpikesAtT; ++k){
            network.getNeuron(targetNeuron).receiveExternalSpike(t, spikeBias);
            stimLog << (int)t << "," << targetNeuron << "," << (int)centerTime << "," << dt << "\n";
        }
    }

    return;
}


int main(int argc, char* argv[]){
    // -------------------------
    // Simulation parameters
    // -------------------------
    const int nNeurons = 200;        // network size
    const double rwStep = 0.01;
    const double refractoryT = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP = 100.;
    const double spacing = 1.0;       // distance between adjacent neurons (spatial units)
    const double velocity = 1.0;      // conduction velocity (spatial units / time unit)
    const int tMax = 2000;            // simulation timesteps

    // NEW: how often (in timesteps) to snapshot the full weight matrix for the
    // "evolution of weights" panel. Smaller -> smoother curves but bigger file / more getWMatrix() calls.
    const int weightSnapshotEvery = 20;

    // -------------------------
    // Gaussian external stimulation pulse parameters
    // -------------------------
    // targetNeuron can be overridden from the command line: ./test_1d_network <targetNeuron>
    int targetNeuron = (argc > 1) ? atoi(argv[1]) : 0;
    const int    pulsePeriod       = 50;    // time between pulse centers (timesteps)
    const double pulseSigma        = 5.0;   // temporal width (std dev, timesteps) of each pulse
    const double pulseWindowSigmas = 3.0;   // truncate each pulse at +/- this many sigmas
    const int    pulsePeakSpikes   = 6;     // micro-spikes delivered exactly at the pulse center
    const double externalSpikeBias = 0.12;  // bias contributed by EACH micro-spike

    if (targetNeuron < 0 || targetNeuron >= nNeurons){
        cerr << "targetNeuron " << targetNeuron << " out of range [0, " << nNeurons-1 << "]\n";
        return 1;
    }

    // Seed RNG
    srand(1053416);

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
    network.setSynapticPacketShape(/*packetSigma=*/1.5, /*packetWindowSigmas=*/3.0);

    // Open spike log file
    ofstream spikeLog("output_1D/spikes.csv");
    spikeLog << "time,neuron\n";

    // Log of every external micro-spike delivered (which neuron, at which time, which pulse,
    // and its offset from that pulse's center -- negative offset = arrived "early", positive = "late")
    ofstream stimLog("output_1D/stimulation.csv");
    stimLog << "time,neuron,pulseCenter,offset\n";

    // Log of weight matrix snapshots over time, long format
    ofstream weightEvoLog("output_1D/weights_evolution.csv");
    weightEvoLog << "time,iNeuron,jNeuron,weight\n";

    // -------------------------
    // Schedule all Gaussian stimulation pulses up front. receiveExternalSpike() just queues
    // spikes internally; they are only actually applied once the simulation clock (in the main
    // loop below) reaches each spike's time, so pre-scheduling the whole run is safe.
    // -------------------------
    for (int centerTime = 0; centerTime < tMax; centerTime += pulsePeriod){
        deliverGaussianPulse(network, targetNeuron, centerTime, pulseSigma, pulseWindowSigmas,
                              pulsePeakSpikes, externalSpikeBias, stimLog);
    }

    // -------------------------
    // Simulation loop
    // -------------------------
    for (int t = 0; t < tMax; ++t) {

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
        // NOTE: getWMatrix() stores its allocation in an internal member pointer that the
        // RWNetwork destructor also frees at the end. Manually deleting each snapshot here
        // causes a double-free once the destructor runs on the last allocation. Since each
        // snapshot is tiny (nNeurons x nNeurons doubles) and the program is short-lived, we
        // deliberately don't free intermediate snapshots -- only the final call's matrix gets
        // freed (once) by the destructor.
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

    // NOTE: W is not manually freed here -- it is freed once by RWNetwork's destructor
    // when `network` goes out of scope (see note above about getWMatrix()'s internal pointer).

    cout << "Simulation finished (target neuron " << targetNeuron << "). "
         << "Spikes -> spikes.csv, stimulation -> stimulation.csv, "
         << "weight evolution -> weights_evolution.csv, final weights -> weights_summary.csv\n";
    return 0;
}