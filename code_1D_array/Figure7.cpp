// fig7_sim.cpp
//
// Reproduces the data needed for Figure 7 of the paper: fixed points omega*
// of the STDP dynamics as a function of distance d between two neurons (A
// presynaptic, B postsynaptic), under a periodic external synchronizing
// signal of period T (paper Sec. III B).
//
// For each distance d and each weight omega in a sweep, we:
//   1. Build a 2-neuron-only system (A, B) with a FIXED weight w[A->B]=omega
//      (STDP disabled / re-pinned every step so the weight never drifts).
//   2. Deliver an external synchronizing spike to BOTH neurons every T steps
//      (I_ext = 1, forcing immediate threshold crossing, per paper Sec III B).
//   3. Also let intrinsic random-walk activity run as usual.
//   4. Record spikes of A and B, compute Delta-t^s pairs within a 5*sigma
//      window (same convention as fig3_sim: Delta t = t_post - t_pre - delay).
//   5. Sum the STDP kernel over all such pairs to get Delta-omega(omega) for
//      this (d, omega) point -- this is the paper's Eq. 7.
//
// We sweep omega over a grid, producing one Delta-omega(omega) curve per
// distance d. Output is written as flat CSV; the polynomial fit + fixed-point
// extraction happens in Python (plot_fig7.py), exactly mirroring how the
// paper says it fits curves to a degree-2 polynomial to locate roots.
//
// Output: output_test/fig7_sweep.csv
//   columns: distance,omega,deltaOmega,delay,nSpikesA,nSpikesB
//
// Usage:
//   ./fig7_sim <T_period> <T_end_per_point> <d_min> <d_max> <d_step> \
//              <omega_min> <omega_max> <omega_step> <velocity>
//
// Example (paper-like, T=10000):
//   ./fig7_sim 10000 200000 0 1000 25 0 1 0.05 1.0

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

// Run one (distance, omega) point. Returns deltaOmega and writes nothing else.
// We build a minimal 2-neuron network manually (not via init1DArray) since we
// only need A and B with a single directed synapse and a controllable delay.
struct PointResult {
    double deltaOmega;
    double delay;
    long   nSpikesA, nSpikesB;
};

PointResult runPoint(double distance, double omega,
                      double velocity, double Tperiod, long Tend,
                      double rwStep, double refractoryT,
                      double amplitudeSTDP, double sigmaSTDP)
{
    srand(1053416);  // identical seed across all (d, omega) points for comparability

    // Manually build a 2-neuron network: neuron 0 = A (pre), neuron 1 = B (post)
    rwN::RWNeuron neuronA(rwStep, refractoryT);
    rwN::RWNeuron neuronB(rwStep, refractoryT);

    double delay = distance / velocity;

    vector<double> spikesA, spikesB;
    vector<int> bothNeurons = {0, 1};  // for the synchronizing signal

    for (long t = 0; t < Tend; ++t) {

        // ── Periodic synchronizing external signal: every Tperiod steps, both
        //    neurons receive I_ext = 1, which is enough to force an immediate
        //    spike on the next update (paper Sec III B: "membrane potential
        //    reaches its threshold value and activates"). ──────────────────
        if (t % (long)Tperiod == 0) {
            neuronA.receiveExternalSpike(t, 1.0);
            neuronB.receiveExternalSpike(t, 1.0);
        }

        // ── Update both neurons' intrinsic + external dynamics ─────────────
        neuronA.updateV(t);
        neuronB.updateV(t);

        // ── Deliver A's spike to B after the fixed delay, with fixed weight ──
        if (neuronA.getHasSpiked()) {
            neuronB.receiveSpike(t + delay, omega);
            spikesA.push_back((double)t);
        }
        if (neuronB.getHasSpiked()) {
            spikesB.push_back((double)t);
        }
    }

    // ── Compute Delta-omega via the same sliding-window approach as fig3_sim ──
    double window = 5.0 * sigmaSTDP;
    double deltaOmega = 0.0;

    size_t lo = 0;
    for (size_t k = 0; k < spikesB.size(); ++k) {
        double tB = spikesB[k];
        double tA_min = tB - window - delay;
        double tA_max = tB + window - delay;
        while (lo < spikesA.size() && spikesA[lo] < tA_min) ++lo;
        size_t kp = lo;
        while (kp < spikesA.size() && spikesA[kp] <= tA_max) {
            double myDeltaT = tB - spikesA[kp] - delay;   // t_post - t_pre - delay
            // Actual weight update (see fig3_sim derivation): deltaW = -deltaW_STDP_raw(myDeltaT)
            double raw = -amplitudeSTDP * myDeltaT * exp(-0.5*pow(myDeltaT/sigmaSTDP,2))
                        / (pow(sigmaSTDP,3) * sqrt(2*M_PI));
            deltaOmega += -raw;
            ++kp;
        }
    }

    PointResult res;
    res.deltaOmega = deltaOmega;
    res.delay = delay;
    res.nSpikesA = spikesA.size();
    res.nSpikesB = spikesB.size();
    return res;
}

int main(int argc, char *argv[]) {
    if (argc < 10) {
        cerr << "Usage: " << argv[0]
             << " <Tperiod> <Tend> <d_min> <d_max> <d_step> "
             << "<omega_min> <omega_max> <omega_step> <velocity>\n";
        return 1;
    }

    double Tperiod   = atof(argv[1]);
    long   Tend      = atol(argv[2]);
    double d_min     = atof(argv[3]);
    double d_max     = atof(argv[4]);
    double d_step    = atof(argv[5]);
    double omega_min = atof(argv[6]);
    double omega_max = atof(argv[7]);
    double omega_step= atof(argv[8]);
    double velocity  = atof(argv[9]);

    const double rwStep        = 0.01;
    const double refractoryT   = 200;
    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP     = 100.;

    ofstream out("output_figure7/fig7_sweep.csv");
    out << "distance,omega,deltaOmega,delay,nSpikesA,nSpikesB\n";

    int nD = (int)((d_max - d_min) / d_step) + 1;
    int nW = (int)((omega_max - omega_min) / omega_step) + 1;
    int total = nD * nW;
    int done = 0;

    for (int di = 0; di < nD; ++di) {
        double d = d_min + di * d_step;
        for (int wi = 0; wi < nW; ++wi) {
            double omega = omega_min + wi * omega_step;
            if (omega > omega_max + 1e-9) continue;

            PointResult r = runPoint(d, omega, velocity, Tperiod, Tend,
                                     rwStep, refractoryT, amplitudeSTDP, sigmaSTDP);

            out << d << "," << omega << "," << r.deltaOmega << ","
                << r.delay << "," << r.nSpikesA << "," << r.nSpikesB << "\n";

            ++done;
            if (done % 20 == 0 || done == total) {
                cerr << "Progress: " << done << "/" << total
                     << "  (d=" << d << ", omega=" << omega << ")\n";
            }
        }
    }

    out.close();
    cout << "Wrote output_figure7/fig7_sweep.csv (" << total << " points)\n";
    return 0;
}