// test_1d_network.cpp
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

// Parses a comma-separated list of ints, e.g. "0,5,10,15" -> {0,5,10,15}
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
    // Simulation parameters
    // -------------------------
    const int nNeurons = 20;        // network size
    const double rwStep = 0.01;
    const double refractoryT = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP = 100.;
    const double spacing = 1.0;       // distance between adjacent neurons (spatial units)
    const double velocity = 1.0;      // conduction velocity (spatial units / time unit)

    // NEW: much longer run by default, so STDP has many more cycles to sculpt the
    // all-to-all connectivity down into (or up into) a structured graph.
    // Override via: ./test_1d_network <targetNeurons> <tMax> <staggerSteps>
    int tMax = (argc > 2) ? atoi(argv[2]) : 8000;

    // How often (in timesteps) to snapshot the full weight matrix for the time-evolution
    // panel. Widened a bit relative to before since tMax is now much larger -- keeps the
    // weights_evolution.csv file a reasonable size while still resolving the slow structural
    // drift we care about here.
    const int weightSnapshotEvery = 50;

    // -------------------------
    // Gaussian external stimulation pulse parameters
    // -------------------------
    // targetNeurons: comma-separated list of neuron indices that receive external stimulation,
    // e.g. "0,5,10,15" spreads 4 independent stimulation sources across the array (instead of
    // just neuron 0), so STDP has more than one "seed" to build structure around.
    vector<int> targetNeurons;
    if (argc > 1) targetNeurons = parseIntList(argv[1]);
    if (targetNeurons.empty()) targetNeurons = {0, 5, 10, 15};

    const int    pulsePeriod       = 50;    // time between pulse centers (timesteps), per target
    const double pulseSigma        = 5.0;   // temporal width (std dev, timesteps) of each pulse
    const double pulseWindowSigmas = 3.0;   // truncate each pulse at +/- this many sigmas
    const int    pulsePeakSpikes   = 6;     // micro-spikes delivered exactly at the pulse center
    const double externalSpikeBias = 0.12;  // bias contributed by EACH micro-spike

    // Stagger: successive target neurons (in the order given) have their pulses shifted in time
    // by this many timesteps, so stimulation sources are not all synchronous -- this gives STDP
    // richer relative-timing relationships to sculpt (rather than everything moving in lockstep).
    // Set to 0 for fully synchronous stimulation of all targets.
    int pulseStaggerSteps = (argc > 3) ? atoi(argv[3]) : 15;

    for (int tn : targetNeurons){
        if (tn < 0 || tn >= nNeurons){
            cerr << "targetNeuron " << tn << " out of range [0, " << nNeurons-1 << "]\n";
            return 1;
        }
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
    ofstream spikeLog("output_graph/spikes.csv");
    spikeLog << "time,neuron\n";

    // Log of every external micro-spike delivered (which neuron, at which time, which pulse,
    // and its offset from that pulse's center -- negative offset = arrived "early", positive = "late")
    ofstream stimLog("output_graph/stimulation.csv");
    stimLog << "time,neuron,pulseCenter,offset\n";

    // Log of weight matrix snapshots over time, long format
    ofstream weightEvoLog("output_graph/weights_evolution.csv");
    weightEvoLog << "time,iNeuron,jNeuron,weight\n";

    // -------------------------
    // Log the initial (pre-STDP) weight matrix separately and explicitly, so downstream
    // graph-comparison analysis has an unambiguous "before" snapshot to compare the final
    // result against (rather than relying on interpreting the first weights_evolution.csv row).
    // -------------------------
    {
        double** Winit = network.getWMatrix();
        ofstream wInitOut("output_graph/weights_initial.csv");
        wInitOut << "iNeuron,jNeuron,weight\n";
        for (int i = 0; i < nNeurons; ++i){
            for (int j = 0; j < nNeurons; ++j){
                wInitOut << i << "," << j << "," << Winit[i][j] << "\n";
            }
        }
        wInitOut.close();
        // NOTE: not manually freed -- see the note further down about getWMatrix()'s internal
        // pointer and why we deliberately let the destructor own cleanup instead.
    }

    // -------------------------
    // Schedule all Gaussian stimulation pulses up front, one independent pulse train per target
    // neuron (each optionally phase-shifted by pulseStaggerSteps relative to the previous one).
    // receiveExternalSpike() just queues spikes internally; they are only actually applied once
    // the simulation clock (in the main loop below) reaches each spike's time, so pre-scheduling
    // the whole run is safe.
    // -------------------------
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

        // Update network (neurons update their v and schedule outgoing spikes)
        network.update(t);

        // Log spikes that occurred at this timestep
        for (int i = 0; i < nNeurons; ++i) {
            if (network.getNeuron(i).getHasSpiked()) {
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

        if (t % 1000 == 0) {
            cout << "t=" << t << " / " << tMax << "\n";
        }
    }
    spikeLog.close();
    stimLog.close();
    weightEvoLog.close();

    // -------------------------
    // Get final weight matrix and write CSV
    // -------------------------
    double** W = network.getWMatrix(); // allocates nNeurons x nNeurons matrix
    ofstream wOut("output_graph/weights_summary.csv");
    wOut << "iNeuron,jNeuron,weight\n";
    for (int i = 0; i < nNeurons; ++i) {
        for (int j = 0; j < nNeurons; ++j) {
            wOut << i << "," << j << "," << W[i][j] << "\n";
        }
    }
    wOut.close();

    // NOTE: W is not manually freed here -- it is freed once by RWNetwork's destructor
    // when `network` goes out of scope (see note above about getWMatrix()'s internal pointer).

    cout << "Simulation finished. Target neurons: ";
    for (int tn : targetNeurons) cout << tn << " ";
    cout << "\nSpikes -> spikes.csv, stimulation -> stimulation.csv, "
         << "initial weights -> weights_initial.csv, "
         << "weight evolution -> weights_evolution.csv, final weights -> weights_summary.csv\n";
    return 0;
}