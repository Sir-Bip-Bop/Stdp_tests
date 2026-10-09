// 1D_array_Figure1.cpp

// Reproduces data for Miguel's TFM figure 1
//
// Runs the 1-D array simulation twice:
//   Run A – default random weights produced by init1DArray
//   Run B – user-supplied weight matrix (CSV file as first argument), or all-zero weights if no file is given
//
// Output files in output_figure1/:
//   voltage_A.csv  spikes_A.csv  weights_A.csv
//   voltage_B.csv  spikes_B.csv  weights_B.csv
//
// Weight CSV formats accepted:
//   (a) triplet rows:  i,j,weight  (header line optional)
//   (b) plain N×N matrix (N rows, N comma-separated values each, no header)
//
// Usage:
//   ./Fig1                    # B = all-zero (disconnected)
//   ./Fig1 weights_input_B.csv   # B from file

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

//Matrix utilities

double **allocMatrix(int N, double val = 0.0) {
    double **M = new double*[N];
    for (int i = 0; i < N; ++i) {
        M[i] = new double[N];
        for (int j = 0; j < N; ++j) M[i][j] = val;
    }
    return M;
}

void freeMatrix(double **M, int N) {
    for (int i = 0; i < N; ++i) delete[] M[i];
    delete[] M;
}

// Copy the network's current weight state into a freshly allocated NxN matrix.
// We do not call getWMatrix() because it stores the pointer inside the network
// and the destructor would then double-free it.
double **extractWeights(rwN::RWNetwork &net) {
    int N = net.getNNeurons();
    double **W = allocMatrix(N, 0.0);
    vector<int>    *eff = net.getEfferentNeurons();
    vector<double> *w   = net.getW();
    for (int i = 0; i < N; ++i)
        for (int slot = 0; slot < (int)eff[i].size(); ++slot)
            W[i][ eff[i][slot] ] = w[i][slot];
    return W;
}

// Overwrite the network's internal weights from a full NxN matrix.
void setWeights(rwN::RWNetwork &net, double **W) {
    int N = net.getNNeurons();
    vector<int>    *eff = net.getEfferentNeurons();
    vector<double> *w   = net.getW();
    for (int i = 0; i < N; ++i)
        for (int slot = 0; slot < (int)eff[i].size(); ++slot)
            w[i][slot] = W[i][ eff[i][slot] ];
}

// Load NxN weights from a CSV file (triplet or plain-matrix format).
double **loadWeightCSV(const char *path, int N) {
    double **W = allocMatrix(N, 0.0);
    ifstream f(path);
    if (!f.is_open()) { cerr << "ERROR: cannot open '" << path << "'\n"; exit(1); }

    string line;
    bool firstLine = true, isTriplet = false;
    int row = 0;

    while (getline(f, line)) {
        if (line.empty()) continue;
        if (firstLine) {
            firstLine = false;
            // skip a text header
            if (!line.empty() && !isdigit(line[0]) && line[0] != '-') continue;
            int commas = 0;
            for (char c : line) if (c == ',') ++commas;
            isTriplet = (commas == 2);
        }
        istringstream ss(line);
        if (isTriplet) {
            int i, j; double val; char comma;
            if (ss >> i >> comma >> j >> comma >> val)
                if (i >= 0 && i < N && j >= 0 && j < N)
                    W[i][j] = val;
        } else {
            if (row >= N) break;
            for (int col = 0; col < N; ++col) {
                double val; char comma;
                if (!(ss >> val)) break;
                W[row][col] = val;
                if (col < N-1) ss >> comma;
            }
            ++row;
        }
    }
    return W;
}

// Builds one mean external-input interval per neuron, spread evenly across
// [minInterval, maxInterval] (neuron 0 -> minInterval, neuron N-1 -> maxInterval).
vector<double> buildExternalIntervals(int nNeurons, double minInterval, double maxInterval) {
    vector<double> interval(nNeurons);
    for (int i = 0; i < nNeurons; ++i) {
        double frac = (nNeurons > 1) ? double(i) / double(nNeurons - 1) : 0.0;
        interval[i] = minInterval + frac * (maxInterval - minInterval);
    }
    return interval;
}

// Single simulation run 

void runSim(const char *tag,
            int nNeurons, double rwStep, double refractoryT,
            double amplitudeSTDP, double sigmaSTDP,
            double spacing, double velocity,
            int tMax, double externalIntervalMin, double externalIntervalMax,
            double externalJitter, double externalBias,
            double **W_init)   // nullptr → keep the random weights from init1DArray
{
    srand(1053416);   // identical seed → only the weight matrix differs between runs
    rwN::RWNetwork network;
    network.init1DArray(nNeurons, rwStep, refractoryT,
                        amplitudeSTDP, sigmaSTDP, spacing, velocity);

    if (W_init != nullptr)
        setWeights(network, W_init);

    // Each neuron gets its own external-input rhythm: a mean inter-spike interval
    // (spread linearly across [externalIntervalMin, externalIntervalMax] over the
    // neuron index) plus uniform jitter of +/- externalJitter on every spike.
    // A separate RNG is used so this doesn't disturb the rand() sequence that
    // init1DArray/updateV rely on for reproducible weights and random-walk steps.
    vector<double> externalInterval = buildExternalIntervals(nNeurons, externalIntervalMin, externalIntervalMax);
    std::mt19937 externalRng(12345);
    std::uniform_real_distribution<double> jitterDist(-externalJitter, externalJitter);
    vector<double> nextExternalSpike(nNeurons);
    for (int i = 0; i < nNeurons; ++i)
        nextExternalSpike[i] = externalInterval[i] + jitterDist(externalRng);

    // Open output files
    ofstream voltLog (string("output_figure1/voltage_") + tag + ".csv");
    ofstream spikeLog(string("output_figure1/spikes_")  + tag + ".csv");

    voltLog << "time";
    for (int i = 0; i < nNeurons; ++i) voltLog << ",v" << i;
    voltLog << "\n";
    spikeLog << "time,neuron\n";

    // Simulation loop
    for (int t = 0; t < tMax; ++t) {
        for (int i = 0; i < nNeurons; ++i) {
            if (t >= nextExternalSpike[i]) {
                network.getNeuron(i).receiveExternalSpike(t, externalBias);
                nextExternalSpike[i] += externalInterval[i] + jitterDist(externalRng);
            }
        }

        network.update(t);

        voltLog << t;
        for (int i = 0; i < nNeurons; ++i)
            voltLog << "," << network.getNeuron(i).getV();
        voltLog << "\n";

        for (int i = 0; i < nNeurons; ++i)
            if (network.getNeuron(i).getHasSpiked())
                spikeLog << t << "," << i << "\n";
    }
    voltLog.close();
    spikeLog.close();

    // Save final weights using our own extractor (avoids double-free from getWMatrix)
    double **Wfinal = extractWeights(network);
    ofstream wOut(string("output_figure1/weights_") + tag + ".csv");
    wOut << "iNeuron,jNeuron,weight\n";
    for (int i = 0; i < nNeurons; ++i)
        for (int j = 0; j < nNeurons; ++j)
            wOut << i << "," << j << "," << Wfinal[i][j] << "\n";
    wOut.close();
    freeMatrix(Wfinal, nNeurons);

    cout << "Run '" << tag << "' complete.\n";
}

// main

int main(int argc, char *argv[]) {

    const int    nNeurons      = 20;
    const double rwStep        = 0.01;
    const double refractoryT   = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP     = 100.;
    const double spacing       = 1.0;
    const double velocity      = 1.0;
    const int    tMax          = 2000;
    // Each neuron gets its own external drive: neuron 0 fires on average every
    // externalIntervalMin ms, neuron nNeurons-1 every externalIntervalMax ms
    // (linearly spread in between), each ± externalJitter ms.
    const double externalIntervalMin = 25.0;
    const double externalIntervalMax = 80.0;
    const double externalJitter      = 1.0;
    const double externalBias  = 0.5;

    // Run A – random init weights (nullptr keeps what init1DArray set)
    runSim("A",
           nNeurons, rwStep, refractoryT, amplitudeSTDP, sigmaSTDP,
           spacing, velocity, tMax, externalIntervalMin, externalIntervalMax,
           externalJitter, externalBias,
           nullptr);

    // Run B – user file or all-zero
    double **W_B = nullptr;
    if (argc >= 2) {
        W_B = loadWeightCSV(argv[1], nNeurons);
        cout << "Weight matrix B loaded from '" << argv[1] << "'\n";
    } else {
        W_B = allocMatrix(nNeurons, 0.0);
        cout << "No weight file given – using all-zero weight matrix for run B.\n";
    }

    runSim("B",
           nNeurons, rwStep, refractoryT, amplitudeSTDP, sigmaSTDP,
           spacing, velocity, tMax, externalIntervalMin, externalIntervalMax,
           externalJitter, externalBias,
           W_B);

    freeMatrix(W_B, nNeurons);
    return 0;
}