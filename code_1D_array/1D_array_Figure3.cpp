// fig3_sim.cpp
//
// Reproduces the data needed for Figure 3 of the paper: the distribution of
// inter-spike intervals Delta-t^s between a chosen presynaptic neuron A and
// postsynaptic neuron B in the 1D array, with a FIXED synaptic weight w_AB
// (STDP is disabled for this pair so the weight never drifts during the run,
// exactly as in Sec. III A of the paper).
//
// Delta t^s_{i,k;j,k'} = t^s_i,k - t^s_j,k' - t^d_ij      (paper's Eq. 4)
//   i = postsynaptic (B), j = presynaptic (A)
//
// We restrict to pairs within a window of 5*sigma of each other, as the
// paper does (Sec. III A, end of paragraph after Eq. 7).
//
// Output: output_test/deltaT_<tag>.csv  -- one Delta-t value per row
//         output_test/meta_<tag>.txt    -- w, sigma, amplitude, delay used
//
// Usage:
//   ./fig3_sim <neuronA> <neuronB> <fixedWeight> <Tend> <tag>
//
// Example:
//   ./fig3_sim 0 1 0.25 10000000 w025

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

int main(int argc, char *argv[]) {

    if (argc < 6) {
        cerr << "Usage: " << argv[0]
             << " <neuronA_pre> <neuronB_post> <fixedWeight> <Tend> <tag>\n";
        return 1;
    }

    int    neuronA   = atoi(argv[1]);   // presynaptic
    int    neuronB   = atoi(argv[2]);   // postsynaptic
    double wFixed    = atof(argv[3]);   // fixed synaptic weight A->B
    long   Tend      = atol(argv[4]);   // simulation length
    string tag       = argv[5];

    // ── Network parameters (same as the rest of the project) ───────────────
    const int    nNeurons      = 20;
    const double rwStep        = 0.01;
    const double refractoryT   = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP     = 100.;
    const double spacing       = 1.0;
    const double velocity      = 1.0;

    srand(1053416);
    rwN::RWNetwork network;
    network.init1DArray(nNeurons, rwStep, refractoryT,
                        amplitudeSTDP, sigmaSTDP, spacing, velocity);

    // ── Isolate the A->B synapse: zero out every OTHER outgoing weight from
    //    every neuron so nothing else interferes with B's dynamics, and fix
    //    w[A->B] = wFixed. We rebuild weights directly through w/efferent
    //    accessors (no other neuron should drive B). ─────────────────────────
    vector<int>    *eff = network.getEfferentNeurons();
    vector<double> *w   = network.getW();
    vector<double> *delays = network.getDelays();

    // Find delay(A->B) and the local slot of B within A's efferent list:
    double delayAB = -1;
    int slotB = -1;
    for (int slot = 0; slot < (int)eff[neuronA].size(); ++slot) {
        if (eff[neuronA][slot] == neuronB) { slotB = slot; delayAB = delays[neuronA][slot]; }
    }
    if (slotB < 0) { cerr << "ERROR: neuronB not found in neuronA's efferent list.\n"; return 1; }

    // Zero every weight in the network, then set only A->B:
    for (int i = 0; i < nNeurons; ++i)
        for (int slot = 0; slot < (int)eff[i].size(); ++slot)
            w[i][slot] = 0.0;
    w[neuronA][slotB] = wFixed;

    // ── Run the simulation, freezing w[A->B] after every step (no STDP drift) ──
    vector<double> spikesA, spikesB;

    for (long t = 0; t < Tend; ++t) {
        network.update(t);

        // Re-freeze the weight (STDP inside update() may have changed it)
        w[neuronA][slotB] = wFixed;

        if (network.getNeuron(neuronA).getHasSpiked())
            spikesA.push_back((double)t);
        if (network.getNeuron(neuronB).getHasSpiked())
            spikesB.push_back((double)t);
    }

    cout << "Simulation finished. nSpikesA=" << spikesA.size()
         << " nSpikesB=" << spikesB.size() << "\n";

    // ── Build Delta-t histogram data: for every B-spike, compare to every
    //    A-spike within a window of 5*sigma (paper's restriction, Sec III A) ──
    double window = 5.0 * sigmaSTDP;
    ofstream dtOut("output_figure3/deltaT_" + tag + ".csv");
    dtOut << "deltaT\n";

    // Both spikesA and spikesB are sorted by construction (we push_back in
    // time order). For each B-spike, only A-spikes with
    //   tB - window - delayAB <= tA <= tB + window - delayAB
    // matter. We use a sliding pointer (lo) since tB is increasing, so the
    // valid A-range only moves forward -- O(nSpikesA + nSpikesB) total.
    size_t lo = 0;
    for (size_t k = 0; k < spikesB.size(); ++k) {
        double tB = spikesB[k];
        double tA_min = tB - window - delayAB;
        double tA_max = tB + window - delayAB;

        while (lo < spikesA.size() && spikesA[lo] < tA_min) ++lo;

        size_t kp = lo;
        while (kp < spikesA.size() && spikesA[kp] <= tA_max) {
            double deltaT = tB - spikesA[kp] - delayAB;
            dtOut << deltaT << "\n";
            ++kp;
        }
    }
    dtOut.close();

    // ── Metadata ─────────────────────────────────────────────────────────────
    ofstream metaOut("output_figure3/meta_" + tag + ".txt");
    metaOut << "neuronA=" << neuronA << "\n";
    metaOut << "neuronB=" << neuronB << "\n";
    metaOut << "weight=" << wFixed << "\n";
    metaOut << "delayAB=" << delayAB << "\n";
    metaOut << "sigmaSTDP=" << sigmaSTDP << "\n";
    metaOut << "amplitudeSTDP=" << amplitudeSTDP << "\n";
    metaOut << "Tend=" << Tend << "\n";
    metaOut << "nSpikesA=" << spikesA.size() << "\n";
    metaOut << "nSpikesB=" << spikesB.size() << "\n";
    metaOut.close();

    cout << "Wrote output_figure3/deltaT_" << tag << ".csv and meta_" << tag << ".txt\n";
    return 0;
}