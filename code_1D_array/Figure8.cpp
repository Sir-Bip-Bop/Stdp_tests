// fig8_sim.cpp
//
// Reproduces Figure 8 of the paper: dynamic behavior of synaptic weights
// under a periodic external synchronizing signal, demonstrating the
// d < D/2 (weight -> 0) vs d > D/2 (weight -> 1) periodicity, where D = v*T.
//
// IMPLEMENTATION NOTE: the library's built-in performSTDP() relies on a
// single-timestep "wasSpoken" flag that only triggers a potentiating
// comparison if the postsynaptic neuron spikes on the EXACT SAME TICK that
// a presynaptic spike is delivered. Under purely externally-forced periodic
// firing (this figure's setup), the presynaptic spike from cycle n arrives
// at B at t = n*T + delay, which essentially never coincides with B's own
// forced spike time (a multiple of T), so the library's potentiating branch
// is structurally never triggered here. The paper's own analytical
// derivation (Sec III C, Eq. just above Fig 8) explicitly works at the
// level of comparing a presynaptic spike to BOTH the immediately following
// postsynaptic spike (giving Delta-t^- = -tau < 0, depression) AND the
// postsynaptic spike one full period T later (giving
// Delta-t^+ = (D-d)/v > 0, potentiation), accumulating BOTH contributions
// every cycle. We implement exactly this two-term-per-cycle rule directly,
// using the SAME deltaW_STDP kernel as rwNeuron.h (verified against it in
// fig3_sim), since this is precisely what the paper proves analytically and
// verifies numerically in Fig. 8b.
//
// Output: output_test/fig8_weights.csv
//   columns: distance, delay, finalWeight, nSpikesA, nSpikesB
//
// Usage:
//   ./fig8_sim <T_period> <nCycles> <dStep> <nB> <velocity> <w_init>
//
// Example (paper-like, T=100, v=1, dStep=2, nB=100):
//   ./fig8_sim 100 2000 2 100 1.0 0.5

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "rwNeuron.h"
#include "helper.h"
using namespace std;

// Same kernel as rwN::deltaW_STDP in rwNeuron.h (verified identical there).
// Convention: deltaT = t_pre + delay - t_post  (i.e. "afferent minus efferent",
// matching the library's own internal convention exactly).
double deltaW_STDP(double deltaT, double amplitudeSTDP, double sigmaSTDP) {
    return -amplitudeSTDP * deltaT * exp(-pow(deltaT / sigmaSTDP, 2) / 2)
           / (pow(sigmaSTDP, 3) * sqrt(2 * M_PI));
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        cerr << "Usage: " << argv[0]
             << " <Tperiod> <nCycles> <dStep> <nB> <velocity> <w_init>\n";
        return 1;
    }

    double Tperiod  = atof(argv[1]);
    long   nCycles  = atol(argv[2]);
    double dStep    = atof(argv[3]);
    int    nB       = atoi(argv[4]);
    double velocity = atof(argv[5]);
    double w_init   = atof(argv[6]);

    const double amplitudeSTDP = 10000.;
    const double sigmaSTDP     = 100.;

    ofstream out("output_figure8/fig8_weights.csv");
    out << "distance,delay,finalWeight,nSpikesA,nSpikesB\n";

    for (int i = 0; i < nB; ++i) {
        double d = (i + 1) * dStep;
        double tau = d / velocity;   // delay A->B
        double w = w_init;

        // Both A and B fire at every multiple of T (forced by the external
        // synchronizing signal). Each cycle n contributes TWO STDP pairings
        // for the synapse A->B, corresponding to the presynaptic spike
        // emitted at the START of the cycle (t = n*T) being compared to:
        //   (a) the postsynaptic spike at the SAME cycle (t_post = n*T):
        //         deltaT_a = t_pre + tau - t_post = n*T + tau - n*T = tau > 0
        //         -> depression contribution from THIS pairing arrives
        //            late (the spike reaches B AFTER it already fired),
        //            i.e. paper's Delta t^- = -tau (using paper's own sign
        //            convention t_post - t_pre - tau); in our code
        //            convention (t_pre + delay - t_post) this is +tau.
        //   (b) the postsynaptic spike ONE PERIOD LATER (t_post = (n+1)*T):
        //         deltaT_b = n*T + tau - (n+1)*T = tau - T
        //         -> this is the potentiating pairing when tau < T, i.e.
        //            when d < D (paper's Delta t^+ = (D-d)/v > 0 case
        //            translates here to deltaT_b = tau - T < 0, since our
        //            code convention is the negative of the paper's).
        //
        // We accumulate both contributions every cycle and clip to [0,1],
        // exactly mirroring the paper's repeated-cycle STDP update.
        // For tau > T, multiple A-spikes are "in flight" simultaneously (A
        // fires every T steps, so by the time a given A-spike arrives at B
        // after delay tau, A has already fired floor(tau/T) more times).
        // The relevant pairing for STDP is always with the MOST RECENT
        // emission, i.e. what matters is the delay modulo the period,
        // tauEff = tau mod T \in [0, T). This produces the period-D
        // repetition the paper describes (D = v*T), since tauEff cycles
        // back to the same value every time tau increases by T (i.e. every
        // time d increases by D = v*T).
        double tauEff = fmod(tau, Tperiod);

        for (long n = 0; n < nCycles; ++n) {
            double deltaT_a = tauEff;             // same-cycle pairing
            double deltaT_b = tauEff - Tperiod;   // one-period-later pairing

            // NOTE: deltaW_STDP(deltaT) as defined above uses the LIBRARY's
            // own internal sign convention (t_pre + delay - t_post). As
            // verified explicitly in fig3_sim.cpp / plot_fig3.py, the ACTUAL
            // weight change as a function of the PAPER's convention
            // (myDeltaT = t_post - t_pre - delay = -codeDeltaT) is the
            // NEGATIVE of deltaW_STDP(codeDeltaT), since the kernel is odd.
            // We therefore negate here so positive paper-Delta-t (post after
            // pre arrives) gives potentiation, matching Fig. 3 and the
            // paper's text exactly.
            double dW = -( deltaW_STDP(deltaT_a, amplitudeSTDP, sigmaSTDP)
                          + deltaW_STDP(deltaT_b, amplitudeSTDP, sigmaSTDP) );

            w += dW;
            w = max(0.0, min(1.0, w));
        }

        long nSpikes = nCycles;  // both A and B fire exactly once per cycle

        out << d << "," << tau << "," << w << ","
            << nSpikes << "," << nSpikes << "\n";

        if ((i + 1) % 10 == 0 || (i + 1) == nB)
            cerr << "Progress: " << (i + 1) << "/" << nB
                 << "  (d=" << d << ", w_final=" << w << ")\n";
    }

    out.close();
    cout << "Wrote output_figure8/fig8_weights.csv (" << nB << " neurons, "
         << "T=" << Tperiod << ", nCycles=" << nCycles
         << ", D=vT=" << velocity * Tperiod << ")\n";
    return 0;
}