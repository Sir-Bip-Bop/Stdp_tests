/*
	1D_spontaneous.cpp

	Motivation
	----------
	Can STDP acting through distance-dependent conduction delays shape the connectivity of a 1D
	array on its own, with no external drive at all? Every synapse starts at the same weight omega,
	neurons fire only through their own random walk plus whatever they pass to each other, and the
	run is long enough to see whether the weight matrix keeps changing or settles. The expectation
	being tested: when delays become small compared with sigmaSTDP the STDP changes should vanish,
	so any structure that survives at small delays must come from saturated activity, where one
	spike sets off endless cascades (unrealistic, epilepsy-like).

	Two scales of the model set where the interesting regimes are, and the sweep should straddle
	both (see run_spontaneous_sweep.py):
	  * Branching ratio. v has no leak, so every input is integrated until the neuron fires. One
	    spike delivers omega to each of the N-1 other neurons, so the mean number of spikes it
	    causes is m ~ omega*(N-1). m < 1: sparse, subcritical activity. m > 1: each spontaneous
	    spike recruits a large part of the array. For N = 100 the boundary is omega ~ 1/(N-1) ~ 0.01.
	  * Reverberation. If the longest delay (N-1)*d0 exceeds refractoryT, spikes from the far end
	    of the array arrive after a neuron's refractory period has ended and can fire it again, so
	    a cascade can sustain itself indefinitely. For N = 100, refractoryT = 200: d0 > ~2.

	What it measures
	----------------
	One run at fixed (N, omega, d0), d0 being the nearest-neighbour delay. Writes to --outdir:
	  meta.json         parameters, snapshot times, raster windows, spike totals, wall time and a
	                    "status" field ("running" while the run is going, "done" at the end)
	  weights.bin       float32 snapshots of the full N x N matrix W[pre][post] (diagonal 0), taken
	                    at t = 0, every --snap-every steps and at t = tmax. Shape (n_snap, N, N).
	  spike_counts.bin  uint16 spike counts per neuron in bins of --count-bin steps, the whole run.
	                    Shape (n_bins, N). Population rate and activity regime come from here.
	  spikes.bin        int32 pairs (t, neuron) in time order. With --spike-log windows (default)
	                    only inside --n-rasters evenly spaced windows of --raster-len steps (the
	                    first starts at t = 0, the last ends at t = tmax); with "all" every spike
	                    (can reach hundreds of MB in self-sustained runs); with "none" nothing.
	  progress.csv      one row per snapshot: mean weight, branching proxy mean_w*(N-1), fraction
	                    of weights near 0 and near 1, RMS weight change since the previous
	                    snapshot, spikes since the previous snapshot. Watch convergence while it runs.

	Model and conventions are those of rwNeuron.h, which is used unchanged:
	  * delays: positions x_i = i*d0 (spacing = d0, velocity = 1), delay(i,j) = |i-j|*d0. A spike
	    sent at t with delay tau is integrated at the first step strictly after t + tau, so the
	    effective lag is floor(tau) + 1 steps even when tau -> 0. d0 = 0 is allowed: all delays
	    vanish and index distance loses any physical meaning (null model for distance structure).
	  * STDP: deltaW = -A * dt * exp(-dt^2 / (2 sigma^2)) / (sigma^3 sqrt(2 pi)), with
	    dt = t_arrival - t_post. Arrival before the postsynaptic spike potentiates. Weights clipped
	    to [0, 1]. All pre/post pairs within 5 sigma are scored.
	  * omega is written over the random initial weights after init1DArray. No external input.

	Speed: RWNetwork::update() runs performSTDP() over all N(N-1) edges at every step, even when no
	neuron spiked, which is most steps in the sparse regime. The default loop ("--loop fast") calls
	the same public library methods in the same order but only runs the STDP pass on steps where
	some neuron spiked, and catches up the history cleaning first. Spikes and weights are bit
	identical to "--loop reference" (plain RWNetwork::update); check it on your machine with the
	commands in Usage before trusting long runs.

	rand() is the only RNG (as in the rest of the repo). RAND_MAX is 2^31-1 with glibc and 32767 with
	MinGW/MSVC, so a given seed reproduces only on the same platform.

	Usage
	-----
	  g++ -O3 -std=c++17 -o 1D_spontaneous 1D_spontaneous.cpp

	  ./1D_spontaneous --omega 0.01 --d0 1 --tmax 10000000 --outdir runs/w0.0100_d1.000_s1
	  ./1D_spontaneous --omega 0.25 --d0 5 --N 100 --tmax 2000000 --snap-every 5000 --seed 7 --outdir runs/test
	  ./1D_spontaneous --omega 0.05 --d0 0 --tmax 5000000 --spike-log all --outdir runs/null_d0

	  # exactness check of the fast loop (outputs must be byte identical):
	  ./1D_spontaneous --omega 0.05 --d0 3 --N 40 --tmax 300000 --spike-log all --outdir chk_fast
	  ./1D_spontaneous --omega 0.05 --d0 3 --N 40 --tmax 300000 --spike-log all --outdir chk_ref --loop reference
	  cmp chk_fast/weights.bin chk_ref/weights.bin && cmp chk_fast/spikes.bin chk_ref/spikes.bin

	  ./1D_spontaneous --help
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <filesystem>
#include "rwNeuron.h"
using namespace std;


struct Params {
	int N = 100;
	double omega = -1.;           // required
	double d0 = 1.0;              // nearest-neighbour delay (timesteps)
	long long tmax = 10000000;
	long long snapEvery = 0;      // 0 -> tmax/400
	unsigned seed = 1053416;      // same default seed as the other drivers in the repo
	double rwStep = 0.01;
	double refractoryT = 200.;
	double amplitudeSTDP = 10000.;
	double sigmaSTDP = 100.;
	double packetSigma = 0.;      // 0 -> single discrete spike per synaptic event
	double packetWindow = 3.;
	string spikeLog = "windows";  // windows | all | none
	long long rasterLen = 20000;
	int nRasters = 10;
	long long countBin = 1000;
	string outdir = "";
	string loop = "fast";         // fast | reference
	bool quiet = false;
};


void printHelp(){
	cout <<
	"1D_spontaneous: spontaneous activity of a 1D rwNeuron array with uniform initial weights\n"
	"  --omega W          initial weight of every synapse (required, in [0,1])\n"
	"  --d0 D             nearest-neighbour delay in timesteps, >= 0 (default 1.0)\n"
	"  --N n              number of neurons (default 100)\n"
	"  --tmax T           number of timesteps (default 1e7)\n"
	"  --snap-every S     steps between weight snapshots (default tmax/400)\n"
	"  --seed s           srand seed (default 1053416)\n"
	"  --rw-step x        random-walk step (default 0.01)\n"
	"  --refractory x     refractory period (default 200)\n"
	"  --A-stdp x         STDP amplitude (default 1e4)\n"
	"  --sigma-stdp x     STDP width (default 100)\n"
	"  --packet-sigma x   Gaussian synaptic packet width, 0 = discrete spikes (default 0)\n"
	"  --packet-window x  packet truncation in sigmas (default 3)\n"
	"  --spike-log M      windows | all | none (default windows)\n"
	"  --raster-len L     length of each raw-spike window (default 20000)\n"
	"  --n-rasters K      number of raw-spike windows, evenly spaced (default 10)\n"
	"  --count-bin B      bin width of the per-neuron spike counts (default 1000)\n"
	"  --outdir path      output directory (required)\n"
	"  --loop fast|reference   fast exact loop (default) or plain RWNetwork::update\n"
	"  --quiet            no progress lines on stdout\n";
}


bool parseArgs(int argc, char* argv[], Params &p){
	for (int i = 1; i < argc; ++i){
		string k = argv[i];
		auto next = [&](void) -> string {
			if (i + 1 >= argc){ cerr << "missing value for " << k << "\n"; exit(2); }
			return string(argv[++i]);
		};
		if (k == "--help" or k == "-h"){ printHelp(); exit(0); }
		else if (k == "--omega") p.omega = atof(next().c_str());
		else if (k == "--d0") p.d0 = atof(next().c_str());
		else if (k == "--N") p.N = atoi(next().c_str());
		else if (k == "--tmax") p.tmax = atoll(next().c_str());
		else if (k == "--snap-every") p.snapEvery = atoll(next().c_str());
		else if (k == "--seed") p.seed = (unsigned) strtoul(next().c_str(), nullptr, 10);
		else if (k == "--rw-step") p.rwStep = atof(next().c_str());
		else if (k == "--refractory") p.refractoryT = atof(next().c_str());
		else if (k == "--A-stdp") p.amplitudeSTDP = atof(next().c_str());
		else if (k == "--sigma-stdp") p.sigmaSTDP = atof(next().c_str());
		else if (k == "--packet-sigma") p.packetSigma = atof(next().c_str());
		else if (k == "--packet-window") p.packetWindow = atof(next().c_str());
		else if (k == "--spike-log") p.spikeLog = next();
		else if (k == "--raster-len") p.rasterLen = atoll(next().c_str());
		else if (k == "--n-rasters") p.nRasters = atoi(next().c_str());
		else if (k == "--count-bin") p.countBin = atoll(next().c_str());
		else if (k == "--outdir") p.outdir = next();
		else if (k == "--loop") p.loop = next();
		else if (k == "--quiet") p.quiet = true;
		else { cerr << "unknown argument " << k << " (try --help)\n"; return false; }
	}
	if (p.omega < 0. or p.omega > 1.){ cerr << "--omega is required and must be in [0,1]\n"; return false; }
	if (p.d0 < 0.){ cerr << "--d0 must be >= 0\n"; return false; }
	if (p.N < 2){ cerr << "--N must be >= 2\n"; return false; }
	if (p.tmax < 1 or p.tmax > 2000000000LL){ cerr << "--tmax must be in [1, 2e9] (library uses int time)\n"; return false; }
	if (p.outdir.empty()){ cerr << "--outdir is required\n"; return false; }
	if (p.loop != "fast" and p.loop != "reference"){ cerr << "--loop must be fast or reference\n"; return false; }
	if (p.spikeLog != "windows" and p.spikeLog != "all" and p.spikeLog != "none"){ cerr << "--spike-log must be windows, all or none\n"; return false; }
	if (p.countBin < 1 or p.rasterLen < 1 or p.nRasters < 1){ cerr << "--count-bin, --raster-len, --n-rasters must be >= 1\n"; return false; }
	if (p.snapEvery <= 0) p.snapEvery = max(1LL, p.tmax / 400);
	return true;
}


// Raw-spike windows [start, end), evenly spaced, first at 0, last ending at tmax.
vector<pair<long long, long long>> rasterWindows(const Params &p){
	vector<pair<long long, long long>> win;
	if (p.spikeLog == "none") return win;
	if (p.spikeLog == "all" or p.rasterLen >= p.tmax){ win.push_back({0, p.tmax}); return win; }
	int K = p.nRasters;
	for (int k = 0; k < K; ++k){
		long long s = (K == 1) ? 0 : (long long) llround(double(k) * double(p.tmax - p.rasterLen) / (K - 1));
		long long e = min(p.tmax, s + p.rasterLen);
		if (not win.empty() and s <= win.back().second) win.back().second = e;   // merge overlaps
		else win.push_back({s, e});
	}
	return win;
}


void writeMeta(const Params &p, const string &status, const vector<long long> &snapTimes,
               const vector<pair<long long, long long>> &windows, long long nBins,
               long long nSpikes, double wallSeconds, long long stepsWithSpikes){
	ofstream m(p.outdir + "/meta.json");
	m.precision(10);
	m << "{\n";
	m << "  \"status\": \"" << status << "\",\n";
	m << "  \"N\": " << p.N << ",\n";
	m << "  \"omega\": " << p.omega << ",\n";
	m << "  \"d0\": " << p.d0 << ",\n";
	m << "  \"max_delay\": " << p.d0 * (p.N - 1) << ",\n";
	m << "  \"tmax\": " << p.tmax << ",\n";
	m << "  \"snap_every\": " << p.snapEvery << ",\n";
	m << "  \"seed\": " << p.seed << ",\n";
	m << "  \"rw_step\": " << p.rwStep << ",\n";
	m << "  \"refractory\": " << p.refractoryT << ",\n";
	m << "  \"A_stdp\": " << p.amplitudeSTDP << ",\n";
	m << "  \"sigma_stdp\": " << p.sigmaSTDP << ",\n";
	m << "  \"packet_sigma\": " << p.packetSigma << ",\n";
	m << "  \"packet_window\": " << p.packetWindow << ",\n";
	m << "  \"loop\": \"" << p.loop << "\",\n";
	m << "  \"rand_max\": " << RAND_MAX << ",\n";
	m << "  \"n_spikes\": " << nSpikes << ",\n";
	m << "  \"spikes_per_neuron\": " << double(nSpikes) / p.N << ",\n";
	m << "  \"steps_with_spikes\": " << stepsWithSpikes << ",\n";
	m << "  \"wall_seconds\": " << wallSeconds << ",\n";
	m << "  \"count_bin\": " << p.countBin << ",\n";
	m << "  \"n_count_bins\": " << nBins << ",\n";
	m << "  \"spike_log\": \"" << p.spikeLog << "\",\n";
	m << "  \"raster_windows\": [";
	for (size_t k = 0; k < windows.size(); ++k){
		if (k) m << ", ";
		m << "[" << windows[k].first << ", " << windows[k].second << "]";
	}
	m << "],\n";
	m << "  \"weights_dtype\": \"float32\",\n";
	m << "  \"weights_layout\": \"(n_snap, N, N), W[pre][post]\",\n";
	m << "  \"spike_counts_layout\": \"uint16 (n_count_bins, N)\",\n";
	m << "  \"spikes_layout\": \"int32 pairs (t, neuron)\",\n";
	m << "  \"snap_times\": [";
	for (size_t k = 0; k < snapTimes.size(); ++k){
		if (k) m << ", ";
		m << snapTimes[k];
	}
	m << "]\n}\n";
}


int main(int argc, char* argv[]){
	Params p;
	if (not parseArgs(argc, argv, p)) return 2;
	std::filesystem::create_directories(p.outdir);

	srand(p.seed);

	// -------------------------
	// Network: 1D array, all-to-all, delay(i,j) = |i-j|*d0, every weight = omega
	// -------------------------
	rwN::RWNetwork network;
	network.init1DArray(p.N, p.rwStep, p.refractoryT, p.amplitudeSTDP, p.sigmaSTDP,
	                    /*spacing=*/p.d0, /*velocity=*/1.0);
	if (p.packetSigma > 0.) network.setSynapticPacketShape(p.packetSigma, p.packetWindow);

	const int N = p.N;
	vector<double> *W = network.getW();
	vector<int> *eff = network.getEfferentNeurons();
	int *nEff = network.getNEfferent();
	for (int i = 0; i < N; ++i)
		for (int j = 0; j < nEff[i]; ++j) W[i][j] = p.omega;

	// -------------------------
	// Outputs
	// -------------------------
	FILE *fSpikes = fopen((p.outdir + "/spikes.bin").c_str(), "wb");
	FILE *fCounts = fopen((p.outdir + "/spike_counts.bin").c_str(), "wb");
	FILE *fWeights = fopen((p.outdir + "/weights.bin").c_str(), "wb");
	ofstream prog(p.outdir + "/progress.csv");
	if (not fSpikes or not fCounts or not fWeights or not prog){
		cerr << "cannot open output files in " << p.outdir << "\n";
		return 1;
	}
	prog << "t,mean_w,branching_proxy,frac_below_0.01,frac_above_0.99,rms_change,spikes_since_last\n";

	const vector<pair<long long, long long>> windows = rasterWindows(p);
	size_t iWin = 0;

	vector<long long> snapTimes;
	vector<float> snap((size_t) N * N, 0.f), prevSnap((size_t) N * N, 0.f);
	vector<uint16_t> counts(N, 0);
	long long nSpikes = 0, spikesAtLastSnap = 0, stepsWithSpikes = 0, nBins = 0;
	vector<int32_t> spikeBuf;
	spikeBuf.reserve(1 << 16);

	auto takeSnapshot = [&](long long t){
		double sum = 0., n0 = 0., n1 = 0., dsq = 0.;
		for (int i = 0; i < N; ++i){
			for (int j = 0; j < nEff[i]; ++j){
				double w = W[i][j];
				snap[(size_t) i * N + eff[i][j]] = (float) w;
				sum += w;
				if (w < 0.01) n0 += 1.;
				if (w > 0.99) n1 += 1.;
			}
		}
		const double nEdges = double(N) * (N - 1);
		if (not snapTimes.empty()){
			for (size_t k = 0; k < snap.size(); ++k){ double d = snap[k] - prevSnap[k]; dsq += d * d; }
		}
		fwrite(snap.data(), sizeof(float), snap.size(), fWeights);
		fflush(fWeights);
		snapTimes.push_back(t);
		double meanW = sum / nEdges;
		double rms = snapTimes.size() > 1 ? sqrt(dsq / nEdges) : 0.;
		prog << t << "," << meanW << "," << meanW * (N - 1) << "," << n0 / nEdges << "," << n1 / nEdges
		     << "," << rms << "," << (nSpikes - spikesAtLastSnap) << "\n";
		prog.flush();
		if (not p.quiet){
			cout << "t=" << t << "  mean_w=" << meanW << "  m~" << meanW * (N - 1)
			     << "  spikes_since_last=" << (nSpikes - spikesAtLastSnap) << "\n" << flush;
		}
		spikesAtLastSnap = nSpikes;
		prevSnap.swap(snap);   // snap now holds the older matrix; fully overwritten next call (diagonal stays 0)
	};

	auto flushSpikes = [&](){
		if (spikeBuf.size()){
			fwrite(spikeBuf.data(), sizeof(int32_t), spikeBuf.size(), fSpikes);
			spikeBuf.clear();
		}
	};

	auto flushCounts = [&](){
		fwrite(counts.data(), sizeof(uint16_t), counts.size(), fCounts);
		fill(counts.begin(), counts.end(), (uint16_t) 0);
		++nBins;
	};

	writeMeta(p, "running", snapTimes, windows, 0, 0, 0., 0);

	// -------------------------
	// Simulation loop (no external input of any kind)
	// -------------------------
	rwN::RWNeuron *neurons = network.getNeurons();
	const double window = 5. * p.sigmaSTDP;
	int lastSTDP = -1;
	const bool fast = (p.loop == "fast");
	auto t0 = chrono::steady_clock::now();

	for (long long tl = 0; tl < p.tmax; ++tl){
		const int t = (int) tl;
		if (tl % p.snapEvery == 0) takeSnapshot(tl);
		if (tl > 0 and tl % p.countBin == 0) flushCounts();

		bool anySpike = false;
		if (fast){
			// Same calls and order as RWNetwork::update(t), skipping the STDP pass on silent steps.
			for (int i = 0; i < N; ++i) neurons[i].updateV(t);
			for (int i = 0; i < N; ++i){
				if (neurons[i].getHasSpiked()){
					anySpike = true;
					for (int j = 0; j < nEff[i]; ++j) network.deliverSpikePacket(i, j, eff[i][j], t);
				}
			}
			if (anySpike){
				// Bring the history cleaning to the state RWNetwork::update would have left it in.
				for (int i = 0; i < N; ++i) neurons[i].cleanEfferentSpikeTrain(t - window);
				if (lastSTDP < t - 1){
					for (int i = 0; i < N; ++i)
						for (int j = 0; j < nEff[i]; ++j) network.cleanPacketHistory(i, j, (t - 1) - window);
				}
				network.performSTDP(t);
				lastSTDP = t;
			}
		}
		else{
			network.update(t);
			for (int i = 0; i < N; ++i) if (neurons[i].getHasSpiked()){ anySpike = true; break; }
		}

		if (anySpike){
			++stepsWithSpikes;
			while (iWin < windows.size() and tl >= windows[iWin].second) ++iWin;
			const bool logRaw = iWin < windows.size() and tl >= windows[iWin].first;
			for (int i = 0; i < N; ++i){
				if (neurons[i].getHasSpiked()){
					++nSpikes;
					if (counts[i] < 65535) ++counts[i];
					if (logRaw){
						spikeBuf.push_back((int32_t) t);
						spikeBuf.push_back((int32_t) i);
					}
				}
			}
			if (spikeBuf.size() > (1 << 20)) flushSpikes();
		}
	}
	flushCounts();   // last (possibly partial) bin
	takeSnapshot(p.tmax);
	flushSpikes();
	fclose(fSpikes);
	fclose(fCounts);
	fclose(fWeights);
	prog.close();

	double wall = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
	writeMeta(p, "done", snapTimes, windows, nBins, nSpikes, wall, stepsWithSpikes);
	if (not p.quiet){
		cout << "done: " << nSpikes << " spikes (" << double(nSpikes) / N << " per neuron), "
		     << wall << " s, outputs in " << p.outdir << "\n";
	}
	return 0;
}
