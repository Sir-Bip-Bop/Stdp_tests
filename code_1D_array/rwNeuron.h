/*

	rwNeuron.h: 

		This file implements the simplest random-walk neurons. 

*/

// Imports: 
#include <iostream> //The default input/output library
#include <fstream> //The default file writting/reading library 
#include <string> //Library for string operations
#include <sstream> //Library for the << when using streams
#include <math.h> //Math operations library
#include <cstdlib>  //C Standard General Utilities Library
#include <vector> //Vector functions
#include <random> //Random functions
#include <algorithm> //has many functions that allow you to modify ranges of data from data structures.
using namespace std; //so for the standard library we don't need to add std:: for functions like cout
// std::default_random_engine kMapGenerator; 

#define _USE_MATH_DEFINES //Allows to use the time management thingy

namespace rwN{ //Wraps all the code in the name rwN to avoid name collisions

	/*

		random-walk neuron namespace: 

	*/

	double deltaW_STDP(double deltaT, double amplitudeSTDP, double sigmaSTDP){
		/* deltaW_STDP function: 

			This function implements the STDP step. Specifically, it computes the increase or decrease that must be
			added or taken away from the weight. Somewhere else we need to take care of the weight becoming 0 or 1. 

			Inputs: 
				>> deltaT: Difference between spikes in the efferent and the afferent neurons. 
					- i.e. (tAfferent - tEfferent) such that deltaT<0 implies reinforcement (deltaW>0). 
					Aferent is what comes and efferent is what leaves
				>> amplitudeSTDP: Amplitude of the STDP modulating function. 
				>> sigmaSTDP: Standard deviation of the Gaussian whose derivative implements the STDP step. 

			Returns: 
				<< deltaW: Increase to be added to the corresponding weight (this might be negative). 

		*/

		double deltaW = -amplitudeSTDP*deltaT*exp(-pow(deltaT/sigmaSTDP, 2)/2) / (pow(sigmaSTDP,3)*pow(2*M_PI, 0.5)); 

		return deltaW; 
	}



	class RWNeuron{
		/* RWNeuron class: 

			This class defines objects of the kind random-walk neurons. rwNeurons have an internal variable, v,
			representing a membane voltage, that is a number within [0,1). When v=1, the neuron spikes, v is reset
			to 0, and the neuron stays at 0 for a refractory period. The variable v changes through a random walk
			(which represents a stereotypical external input) and through biases that arrive as spikes from other
			neurons. rwNeurons have weights that tell how much each of its afferent neurons can bias the random
			walk, and a train of incoming spikes. As an output, rwNeurons have an axon that arrives with delays to
			each of its efferent neurons. 

			Variables: 
				>> v \in [0,1): Membrane potential. 
				>> rwStep: Size of random-walk steps -- determines spontaneous firing rate. 
				>> refractoryT: Refractory period. 
				>> spikeTrain: Vector containing incoming spikes. 
				>> spikeTrainBias: Vector containing the bias that incoming spikes will exert on the neuron's state. 
				>> lastEfferentSpike: Last spike from this neuron. 
				>> efferentSpikeTrain: Vector with the times of the last spikes by the neuron. 
				>> hasSpiked: Boolean indicating whether this neuron has spiked. 
				>> wasSpoken: Boolean indicating whether this neuron has received a new spike. 

		*/

			double v, rwStep, refractoryT; 
			double lastEfferentSpike; 
			vector<double> efferentSpikeTrain, spikeTrain, spikeTrainBias; 
			vector<int> spikeTrainIndex; 
			bool hasSpiked, wasSpoken; 
		public: 
			// Init functions: 
			RWNeuron(), RWNeuron(double, double); 

			//Destruction function;
			~RWNeuron();

			// Get functions: 
			double getV(), getRefractoryT(), getRWStep(); 
			double getLastSpike(), getSpikeTrainTime(int), getSpikeTrainBias(int), getEfferentSpike(int); 
			vector<double> &getEfferentSpikeTrain(), &getSpikeTrain(), &getSpikeTrainBias(); 
			bool getHasSpiked(), getWasSpoken(); 

			// Functional functions: 
			void cleanEfferentSpikeTrain(double); 
			void receiveSpike(double, double), receiveExternalSpike(double, double); 
			void updateV(int); 

	}; 



	///////////////////////////////
	// 
	//  RWNeuron functions: 

		///////////////////////////////
		// 
		//  Init Functions: 

		RWNeuron::RWNeuron(){return;}

		RWNeuron::RWNeuron(double rwStep_, double refractoryT_){
			/* Init function for ojects of class RWNeuron: 

				This function initializes random-walk neurons with a refractory period and a random-walk step. 

				Inputs: 
					>> rwStep_: Size of random-walk step. 
					>> refractoryT_: Neuron's refractory period. 

			*/

			rwStep = rwStep_; 
			refractoryT = refractoryT_; //Why  don't we initiate voltage here
			v = 0.0;
			lastEfferentSpike = -1E6; 		// This is set so long ago that it has no influence in first SPDT. 
			efferentSpikeTrain.clear(); 	// Hopefully, this will make efferentSpikeTrain unnecessary. 
			spikeTrain.clear(); 
			spikeTrainBias.clear(); 
			spikeTrainIndex.clear(); 
			hasSpiked = false; 
			wasSpoken = false; 

			return; 
		}


		///////////////////////////////
		// 
		//  Get Functions: 

		double RWNeuron::getV(){
			/* getV function: */

			return v; 			
		}

		double RWNeuron::getRefractoryT(){
			/* getRefractoryT function: */

			return refractoryT; 
		}

		double RWNeuron::getRWStep(){
			/* getRWStep function: */

			return rwStep; 
		}

		double RWNeuron::getLastSpike(){
			/* getLastSpike function: 

				Returns the time of the last spike. 

			*/

			return lastEfferentSpike; 
		}

		double RWNeuron::getSpikeTrainTime(int iSpike){
			/* getSpikeTime function: 

				Returns the time of the iSpike-th spike in the afferent spike train! 

			*/

			return spikeTrain[iSpike]; 
		}

		double RWNeuron::getSpikeTrainBias(int iSpike){
			/* getSpikeBias function: 

				Returns the bias of the iSpike-th in the afferent spike train! 

			*/

			return spikeTrainBias[iSpike]; 
		}

		double RWNeuron::getEfferentSpike(int iSpike){
			/* getEfferentSpike function: */

			return efferentSpikeTrain[iSpike]; 
		}

		vector<double> &RWNeuron::getEfferentSpikeTrain(){
			/* getEfferentSpikeTrain function: */

			return efferentSpikeTrain; 
		}
	
		vector<double> &RWNeuron::getSpikeTrain(){
			/* getSpikeTrain function: */

			return spikeTrain; 
		}

		vector<double> &RWNeuron::getSpikeTrainBias(){
			/* getSpikeBias function: */

			return spikeTrainBias; 
		}

		bool RWNeuron::getHasSpiked(){
			/* getHasSpiked function: */

			return hasSpiked; 
		}

		bool RWNeuron::getWasSpoken(){
			/* getWasSpoken function: */

			return wasSpoken; 
		}


		///////////////////////////////
		// 
		//  Functional Functions: 

		void RWNeuron::cleanEfferentSpikeTrain(double cleaningThreshold){
			/* cleanEfferentSpikeTrain function: 

				This function removes spikes from the efferentSpikeTrain that are older than cleaningThreshold. 

				Inputs: 
					>> cleaningThreshold: Time before which all spikes should be removed from the efferentSpikeTrain. 

			*/

			while(efferentSpikeTrain.size() and efferentSpikeTrain[0] <= cleaningThreshold){
				efferentSpikeTrain.erase(efferentSpikeTrain.begin()); 
			}

			return; 
		}

		void RWNeuron::receiveSpike(double spikeTime, double spikeBias){
			/* receiveSpike function: 

				This function receives an afferent spike and sorts it out within the spike train vector. 

				Inputs: 
					>> spikeTime: Time at which the afferent spike arrives to the neuron. 
					>> spikeBias: Bias that the spike will induce in the internal state. 

			*/

			// If the new spike arrives later than all other spikes, just add to spike train: 
			if (not(spikeTrain.size()) or spikeTime > spikeTrain.back()){
				spikeTrain.push_back(spikeTime); 
				spikeTrainBias.push_back(spikeBias); 
			}
			// Else, we need to inset it in the right position: 
			else{
				for (int i=0; i<spikeTrain.size(); i++){
					if (spikeTime < spikeTrain[i]){
						spikeTrain.insert(spikeTrain.begin()+i, spikeTime); 
						spikeTrainBias.insert(spikeTrainBias.begin()+i, spikeBias); 
						break; 
					}
				}
			}
			wasSpoken = true; 

			return; 
		}

		void RWNeuron::receiveExternalSpike(double spikeTime, double spikeBias){
			/* receiveExternalSpike function: 

				This function delivers an external spike. The only different with respect to internal spikes is that it
				does not prompt the neuron to update its connections due to STDP. 

				Inputs: 
					>> spikeTime: Time at which the afferent spike arrives to the neuron. 
					>> spikeBias: Bias that the spike will induce in the internal state. 

			*/
			// If the new spike arrives later than all other spikes, just add to spike train: 
			if (not(spikeTrain.size()) or spikeTime > spikeTrain.back()){
				spikeTrain.push_back(spikeTime); 
				spikeTrainBias.push_back(spikeBias); 
			}
			// Else, we need to inset it in the right position: 
			else{
				for (int i=0; i<spikeTrain.size(); i++){
					if (spikeTime < spikeTrain[i]){
						spikeTrain.insert(spikeTrain.begin()+i, spikeTime); 
						spikeTrainBias.insert(spikeTrainBias.begin()+i, spikeBias); 
						break; 
					}
				}
			}

			return; 
		}

		void RWNeuron::updateV(int t){
			/* updateV function: 

				This function updates the neuron's internal state by two means: (i) A random walk, which reflects
				spontaneous activity and (ii) a positive bias from each afferent neuron. 

				Inputs: 
					>> t: Current time step. 

			*/

			if(wasSpoken) wasSpoken = false; 

			if (t - lastEfferentSpike > refractoryT){
				v += 2*rwStep*( double(rand())/RAND_MAX - 0.5); 
				v = max(v, 0.); 


				bool fCheckSpike=true; 
				while (fCheckSpike){
					if (spikeTrain.size() and t>spikeTrain[0]){
						v += spikeTrainBias[0]; 
						spikeTrain.erase(spikeTrain.begin()); 
						spikeTrainBias.erase(spikeTrainBias.begin()); 
					}
					else fCheckSpike = false; 
				}

				if (v >= 1.){
					v=0.; 
					lastEfferentSpike = t; 
					efferentSpikeTrain.push_back(t); 
					hasSpiked = true; 
				}
			}
			else{	
				if (hasSpiked) hasSpiked=false; 

				// We need to remove spikes that arrived during the refractory period: 
				bool fCheckSpike=true; 
				while (fCheckSpike){
					if (spikeTrain.size() and t>spikeTrain[0]){
						spikeTrain.erase(spikeTrain.begin()); 
						spikeTrainBias.erase(spikeTrainBias.begin()); 
					}
					else fCheckSpike = false; 
				}
			}

			return; 
		}
		RWNeuron::~RWNeuron(){
    		// vectors will free themselves; nothing allocated with new inside RWNeuron
    		// but clear to be explicit
    		efferentSpikeTrain.clear();
    		spikeTrain.clear();
    		spikeTrainBias.clear();
    		spikeTrainIndex.clear();
    		hasSpiked = false;
   			 wasSpoken = false;
    		// no dynamic memory to delete here
		}



	class RWNetwork{
		/* RWNetwork class: 

			This class defines objects of the kind random-walk network. RWNnetworks contain an array of neurons, and
			store connections between them as well as weights and delays. RWNetworks also take care of updating the
			weights with stdp after each spike. 

			Variables: 
				>> rwStep: Size of random-walk steps -- determines spontaneous firing rate. 
				>> refractoryT: Refractory period. 
				>> sigmaSTDP: Constant defining the width of the Gaussian involved in STDP. 
				>> neurons: Array of neurons in the network. 
				>> nNeurons: Number of neurons in the network. 
				>> nAfferent, nEfferent: Array of integers with the numbers of afferent (resp. efferent) neurons in each neuron. 
				>> afferentNeurons, efferentNeurons: Array of vectors with the indexes of afferents (resp. efferent) neurons. 
				>> w: Array of vectors with each neuron's afferent weigths. 
				>> delays: Array of vectors with each neuron's efferent delays. 

		*/

			// Basic variables: 
			double rwStep, refractoryT, amplitudeSTDP, sigmaSTDP; 
			rwN::RWNeuron *neurons; 
			int nNeurons, *nAfferent, *nEfferent; 
			vector<int> *afferentNeurons, *efferentNeurons; 
			vector<double> *w, *delays; 

			// Indexes and such: 
			int iNeuron_dummy, iNeuron, jNeuron, iSpike; 
			double deltaT, deltaW; 

			// Extended variables (to implement different architechtures): 
			double *externalW,  *externalDelays, **wMatrix; 

			// Synaptic spike-packet shape: instead of delivering a single discrete spike of size
			// w[iNeuron][jNeuron] at t+delay, an efferent spike is delivered as a train of
			// micro-spikes discretized from a Gaussian centered at t+delay. packetSigma is the
			// standard deviation (in timesteps) of that Gaussian; packetWindowSigmas truncates the
			// train at +/- that many sigmas. The micro-spike biases are normalized to sum to
			// w[iNeuron][jNeuron], so the total charge delivered per efferent spike is unchanged --
			// only its temporal profile changes. Setting packetSigma<=0 reverts to the original
			// single discrete spike (kept as the default, for backward compatibility). 
			double packetSigma = 0.0, packetWindowSigmas = 3.0; 

			// Per-edge retained micro-spike history, for STDP purposes: packetTimes[iNeuron][jNeuron]
			// / packetFractions[iNeuron][jNeuron] hold the arrival time and packet-fraction of every
			// micro-spike sent along that specific (iNeuron -> its jNeuron-th efferent) connection,
			// kept until it ages out of the STDP window. This lets STDP score each micro-spike
			// individually against the postsynaptic spike history, instead of collapsing a whole
			// packet into one nominal t+delay timestamp -- so different micro-spikes from the SAME
			// packet can potentiate or depress depending on their own arrival time. Lazily allocated
			// (nullptr until first used) so no init function needs to be touched to set them up. 
			vector<vector<double>> *packetTimes = nullptr, *packetFractions = nullptr; 

		public: 
			// Init functions: 
			RWNetwork(), RWNetwork(int, double, double, double, double, double); 
			void initL4Barrels(int, double, double, double, double, double, double, double); 
			void init1DArray(int, double, double, double, double, double, double);

			// Destruction function;
			~RWNetwork();   
			RWNetwork(const RWNetwork&) = delete;
    		RWNetwork& operator=(const RWNetwork&) = delete;
			// Get functions: 
			double getAmplitudeSTDP(), getSigmaSTDP(); 
			RWNeuron &getNeuron(int), *&getNeurons(); 
			int getNNeurons(), getNAfferent(int), *getNAfferent(), getNEfferent(int), *getNEfferent(); 
			vector<int> getAfferentNeurons(int), *getAfferentNeurons(), getEfferentNeurons(int), *getEfferentNeurons(); 
			vector<double> getW(int), *getW(), getDelays(int), *getDelays(); 
			double &getExternalW(int), *&getExternalW(), &getExternalDelay(int), *&getExternalDelays(); 
			double **&getWMatrix(); 

			// Functional functions: 
			void update(int), performSTDP(int), performSTDP_verbose(int); 
			void deliverExternalSpike(double, double), deliverExternalSpike(double, double, vector<int> &); 
			void setSynapticPacketShape(double, double); 
			void deliverSpikePacket(int, int, int, double); 
			void computePacketShape(int, int, double, vector<double> &, vector<double> &); 
			void cleanPacketHistory(int, int, double); 

	}; 


	///////////////////////////////
	// 
	//  RWNetwork functions: 

		///////////////////////////////
		// 
		//  Init Functions: 

		RWNetwork::RWNetwork(){return;}

		RWNetwork::RWNetwork(int nNeurons_, double rwStep_, double refractoryT_, double amplitudeSTDP_, double sigmaSTDP_, double delay_){
			/* Init function for ojects of class RWNetwork: 

				This function initializes random-walk networks with all neurons connected to each-other except
				themselves. Neurons are initialized with rwStep and refractoryT. Delays are initialized to delay.
				Weights are initialized randomly. 

				Inputs: 
					>> nNeurons_: Number of neurons in the network. 
					>> rwStep_: Size of random-walk steps -- determines spontaneous firing rate. 
					>> refractoryT_: Refractory period. 
					>> delay_: Fixed delay for every neuron. 
					>> amplitudeSTDP_: Amplitude of the STDP modulating function. 
					>> sigmaSTDP_: Constant defining the width of the Gaussian involved in STDP. 

			*/

			nNeurons = nNeurons_; 
			amplitudeSTDP = amplitudeSTDP_; 
			sigmaSTDP = sigmaSTDP_; 
			neurons = new rwN::RWNeuron[nNeurons]; 
			nAfferent = new int[nNeurons]; 
			nEfferent = new int[nNeurons]; 
			afferentNeurons = new vector<int>[nNeurons]; 
			efferentNeurons = new vector<int>[nNeurons]; 
			w = new vector<double>[nNeurons]; 
			delays = new vector<double>[nNeurons]; 
			for (int i=0; i<nNeurons; i++){
				neurons[i] = rwN::RWNeuron(rwStep_, refractoryT_); 
				nAfferent[i] = nNeurons-1; 
				nEfferent[i] = nNeurons-1; 
				for (int j=0; j<nNeurons; j++){
					if (j!=i){
						afferentNeurons[i].push_back(j);
						efferentNeurons[i].push_back(j);
						w[i].push_back(double(rand())/RAND_MAX); 
						delays[i].push_back(delay_); 
					}
				}
			}

			return; 
		}

		void RWNetwork::initL4Barrels(int nNeuronsPerBarrel, double rwStep_, double refractoryT_, double amplitudeSTDP_, 
										double sigmaSTDP_, double shortDelay, double longDelay, double sigmaExternalDelay){
			/* initL4Barrels function: 

				This function initializes a circuit that emulates a simplified disposition of the Layer 4 of two
				mirror-symmetric barrel cortical columns. 

				Each of the two mirror-symmetric L4 circuits consists of nNeuronsPerBarrel distributed uniformly over a
				1D segment such that each neuron is at a fixed distance from its nearest neighbors. This distance is
				such that a spike takes a time shortDelay to reach nearest neightbors. Accordingly, it takes
				2*shortDelay to reach the next nearest neighbors, 3*shortDelay to reach the nearest neighbors after
				them, etc. The mirror circuits are at a fixed distance from each other as well, such that spikes
				from the i-th neuron in the left barrel reach the i-th neuron from the right barrel a time longDelay
				after the spike happened. Short delays are added to reach the next neurons, etc. 

				Finally, all neurons can receive external spikes. These are not subjected to STDP, and their weight is
				usually 1. External spikes reach neurons with a delay given by the externalDelays array. For this
				simplest version of barrels, external delays are produced from a Gaussian distribution with standard
				deviation sigmaExternalDelay, such that the earliest arriving external spike arrives with delay 0. 

				In the simplest model, all neurons are connected to each other and weights are initialized randomly and
				uniformly within [0,1]. 

			*/

			// Auxiliary variables for this init function: 
			std::default_random_engine generator;
			generator.seed(rand()); 
			std::normal_distribution<double> distribution(0., sigmaExternalDelay);
  			double minDelay = 1E6, newDelay, *positions; 

			// Initializing common variables: 
			nNeurons = 2*nNeuronsPerBarrel; 
			amplitudeSTDP = amplitudeSTDP_; 
			sigmaSTDP = sigmaSTDP_; 
			neurons = new rwN::RWNeuron[nNeurons]; 
			nAfferent = new int[nNeurons]; 
			nEfferent = new int[nNeurons]; 
			afferentNeurons = new vector<int>[nNeurons]; 
			efferentNeurons = new vector<int>[nNeurons]; 
			w = new vector<double>[nNeurons]; 
			delays = new vector<double>[nNeurons]; 

			// Initializing extended variables: 
			externalW = new double[nNeurons]; 
			externalDelays = new double[nNeurons]; 

			// Initializing positions, which is used as an aux var to initialize delays: 
  			positions = new double[nNeurons];
  			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
  				positions[iNeuron] = iNeuron; 
  				if (iNeuron>=nNeuronsPerBarrel) positions[iNeuron] += longDelay - nNeuronsPerBarrel; 
  			}

			// Loop over neurons to continue with initializations: 
			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
				
				// Common variables: 
				neurons[iNeuron] = rwN::RWNeuron(rwStep_, refractoryT_); 
				nAfferent[iNeuron] = nNeurons - 1; 
				nEfferent[iNeuron] = nNeurons - 1; 

				// Extended variables: 
				externalW[iNeuron] = 1.; 
				// ACHTUNG!! 
				// externalDelays[iNeuron] = sigmaExternalDelay*double(rand())/RAND_MAX; 
				externalDelays[iNeuron] = distribution(generator); 
				if (externalDelays[iNeuron]<minDelay) minDelay = externalDelays[iNeuron]; 

				// Loop over neighbors to finish initializing: 
				for (jNeuron=0; jNeuron<nNeurons; jNeuron++){
					if (jNeuron!=iNeuron){
						afferentNeurons[iNeuron].push_back(jNeuron); 
						efferentNeurons[iNeuron].push_back(jNeuron); 
						w[iNeuron].push_back(double(rand())/RAND_MAX); 
						delays[iNeuron].push_back(abs(positions[iNeuron] - positions[jNeuron])); 
					}
				}
			}

			// External delays must be corrected such that minimum is 0: 
			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
				externalDelays[iNeuron] -= minDelay; 
			}

			return; 
		}

		void RWNetwork::init1DArray(int nNeurons_, double rwStep_, double refractoryT_, double amplitudeSTDP_, double sigmaSTDP_, double spacing, double velocity){
			/* initL4Barrels function: 

				This function initializes a networkthat is a 1D array, with each neuron has a position on the x-axis. Currently, the positions are 
				equidistant. The propagation delay is defined as the distance between two neurons divided by the velocity.

				Finally, all neurons can receive external spikes. These are not subjected to STDP, and their weight is
				usually 1.

				In the simplest model, all neurons are connected to each other and weights are initialized randomly and
				uniformly within [0,1]. 

			*/
			
			//initiating variables
			nNeurons = nNeurons_;
			amplitudeSTDP = amplitudeSTDP_;
			sigmaSTDP = sigmaSTDP_;
			neurons = new rwN::RWNeuron[nNeurons];
			nAfferent = new int[nNeurons];
			nEfferent = new int[nNeurons];
			afferentNeurons = new vector<int>[nNeurons];
			efferentNeurons = new vector<int>[nNeurons];
			w = new vector<double>[nNeurons];
			delays = new vector<double>[nNeurons];
			externalW = new double[nNeurons];
			externalDelays = new double[nNeurons];

			wMatrix = nullptr;

			//creating the x-axis positions
			double *x = new double[nNeurons];
			for (iNeuron = 0; iNeuron < nNeurons; iNeuron++){
				//modify this loop if we want to create a different spacing
				x[iNeuron] = iNeuron * spacing;
			}

			//initialating the neurons
			for(iNeuron = 0; iNeuron< nNeurons; iNeuron++){
				neurons[iNeuron] = rwN::RWNeuron(rwStep_,refractoryT_);
				//Change these two lines if we want to change the connectivity from all to all to something different
				nAfferent[iNeuron] = nNeurons -1;
				nEfferent[iNeuron] = nNeurons -1;

				//External input: like the barrels (for now, it's a false assumption we'll also change later)
				externalW[iNeuron] = 1.;
				externalDelays[iNeuron] = 0.;

				//Finish connection types (also change if we want to move from all to all connectivity)
				for (jNeuron = 0; jNeuron < nNeurons; jNeuron ++){
					if (jNeuron != iNeuron){
						afferentNeurons[iNeuron].push_back(jNeuron);
						efferentNeurons[iNeuron].push_back(jNeuron);
						w[iNeuron].push_back(double(rand()) / RAND_MAX);

						//First change! Delay is proportional to the physical distance between the two neurons
						double distance = abs(x[iNeuron] - x[jNeuron]);
						delays[iNeuron].push_back(distance / velocity);
					}
				}
			}

			delete[] x;

			return;

		}


		///////////////////////////////
		// 
		//  Get Functions: 

		double RWNetwork::getAmplitudeSTDP(){
			/* getAmplitudeSTDP function: */

			return amplitudeSTDP; 
		}

		double RWNetwork::getSigmaSTDP(){
			/* getSigmaSTDP function: */

			return sigmaSTDP; 
		}

		RWNeuron &RWNetwork::getNeuron(int iNeuron){
			/* getNeuron function: */

			return neurons[iNeuron]; 
		}

		RWNeuron *&RWNetwork::getNeurons(){
			/* getNeuron function: */

			return neurons; 
		}

		int RWNetwork::getNNeurons(){
			/* getNNeurons function: */

			return nNeurons; 
		}

		int RWNetwork::getNAfferent(int iNeuron){
			/* getNAfferent function: */

			return nAfferent[iNeuron]; 
		}

		int *RWNetwork::getNAfferent(){
			/* getNAfferent function: */

			return nAfferent; 
		}

		int RWNetwork::getNEfferent(int iNeuron){
			/* getNEfferent function: */

			return nEfferent[iNeuron]; 
		}

		int *RWNetwork::getNEfferent(){
			/* getNEfferent function: */

			return nEfferent; 
		}

		vector<int> RWNetwork::getAfferentNeurons(int iNeuron){
			/* getAfferentNeurons function: */

			return afferentNeurons[iNeuron]; 
		}

		vector<int> *RWNetwork::getAfferentNeurons(){
			/* getAfferentNeurons function: */

			return afferentNeurons; 
		}

		vector<int> RWNetwork::getEfferentNeurons(int iNeuron){
			/* getEfferentNeurons function: */

			return efferentNeurons[iNeuron]; 
		}

		vector<int> *RWNetwork::getEfferentNeurons(){
			/* getEfferentNeurons function: */

			return efferentNeurons; 
		}

		vector<double> RWNetwork::getW(int iNeuron){
			/* getW function: */

			return w[iNeuron]; 
		}

		vector<double> *RWNetwork::getW(){
			/* getW function: */

			return w; 
		}

		vector<double> RWNetwork::getDelays(int iNeuron){
			/* getDelays function: */

			return delays[iNeuron]; 
		}

		vector<double> *RWNetwork::getDelays(){
			/* getDelays function: */

			return delays; 
		}

		double &RWNetwork::getExternalW(int iNeuron){
			/* getExternalW functino: */

			return externalW[iNeuron]; 
		}

		double *&RWNetwork::getExternalW(){
			/* getExternalW functino: */

			return externalW; 
		}
		
		double &RWNetwork::getExternalDelay(int iNeuron){
			/* getExternalDelay functino: */

			return externalDelays[iNeuron]; 
		}

		double *&RWNetwork::getExternalDelays(){
			/* getExternalDelays functino: */

			return externalDelays; 
		}

		double **&RWNetwork::getWMatrix(){
			/* getWMatrix function: 

				This function returns the weights in the shape of a matrix. 

				Returns: 
					<< wMatrix: Matrix of size nNeurons*nNeurons containing the weights of the network. 

			*/

			// Initializing wMatrix: 
			wMatrix = new double*[nNeurons]; 
			for (int i=0; i<nNeurons; i++){
				wMatrix[i] = new double[nNeurons]; 
			}
			for (int i=0; i<nNeurons; i++){
				wMatrix[i][i] = 0.; 
				for (int j=(i+1); j<nNeurons; j++){
					wMatrix[i][j] = 0.; 
					wMatrix[j][i] = 0.; 
				}
			}

			// Reaading values from actual weights: 
			for (iNeuron = 0; iNeuron<nNeurons; iNeuron++){
				for (int j=0; j<nEfferent[iNeuron]; j++){
					jNeuron = efferentNeurons[iNeuron][j]; 
					wMatrix[iNeuron][jNeuron] = w[iNeuron][j]; 
				}
			}

			return wMatrix; 
		}


		///////////////////////////////
		// 
		//  Functional Functions: 

		void RWNetwork::update(int t){
			/* update function: 

				This function updates the state of every neuron in the network. It also takes care of delivering spikes
				from each spiking neuron to its efferent neurons. 

				Inputs: 
					>> t: Time. 

			*/
			// Loop over all neurons to update internal states and send out spikes: 
			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
				neurons[iNeuron].updateV(t);
			}

			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
				// If the neuron has spiked we need to deliver spikes to all its efferent neurons: 
				if (neurons[iNeuron].getHasSpiked()){
					for (jNeuron=0; jNeuron<nEfferent[iNeuron]; jNeuron++){
						iNeuron_dummy = efferentNeurons[iNeuron][jNeuron]; 
						deliverSpikePacket(iNeuron, jNeuron, iNeuron_dummy, t); 
					}
				}
	
				// In any case, clean very old spikes from efferent trains: 
				neurons[iNeuron].cleanEfferentSpikeTrain(t - 5*sigmaSTDP); 
			}
			performSTDP(t); 
			return; 
		}

		void RWNetwork::performSTDP(int t){
			/* performSTDP function: 

				This function implements the STDP step. 

				As of the Gaussian spike-packet delivery (deliverSpikePacket), a single efferent
				spike is no longer a single event at t+delay -- it's a train of micro-spikes spread
				around that center. So a synaptic event is no longer scored as one deltaT: EACH
				micro-spike is scored individually against the postsynaptic spike history, weighted
				by its own fraction of the packet (fractions sum to 1 per packet). This means
				different micro-spikes from the SAME packet can land on different sides of a given
				post spike -- some potentiating, some depressing -- instead of the whole packet
				being forced to agree on one sign. 

				This uses two retained histories per edge: 
					>> neurons[iNeuron_dummy].getEfferentSpikeTrain(): iNeuron_dummy's own past spike
						times (unchanged from before -- still just real spikes, not micro-spikes). 
					>> packetTimes[iNeuron][jNeuron] / packetFractions[iNeuron][jNeuron]: every
						micro-spike THIS specific edge has sent recently (new bookkeeping, see class
						declaration and computePacketShape()/cleanPacketHistory()). 

				Per edge, in order: 
					1) BRANCH 1 -- if iNeuron_dummy (post) just spiked at t: score it against every
					   OLDER retained micro-spike on this edge (i.e. packets from strictly earlier
					   timesteps -- today's packet, if any, hasn't been appended yet at this point). 
					2) BRANCH 2 -- if iNeuron (pre) just spiked at t, delivering a NEW packet along
					   this edge: score every micro-spike of that new packet against ALL of post's own
					   spike history (which may include a spike from this very timestep, if pre and
					   post fired simultaneously). 
					3) Only THEN append the new packet's micro-spikes to packetTimes/packetFractions,
					   for future lookups by branch 1. 

				Because branch 1 only ever sees packets appended in a STRICTLY earlier call, and
				branch 2 handles today's packet completely (including against a simultaneous post
				spike), every (micro-spike, post-spike) pair is scored exactly once -- no "scored
				twice, subtract one contribution" correction is needed anymore (the original
				single-discrete-spike version of this function needed that because it re-derived
				todays's spike from iNeuron's own spike train, which already included it by the time
				performSTDP ran; this version's branch 1 explicitly excludes today's entry instead). 

				Inputs: 
					>> t: Time. 

			*/

			if (not packetTimes){
				packetTimes = new vector<vector<double>>[nNeurons]; 
				packetFractions = new vector<vector<double>>[nNeurons]; 
			}

			int iSuperDummy;  

			// Loop over all neurons to update the weights: 
//			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
			for (iNeuron=nNeurons-1; iNeuron>=0; iNeuron--){

				// Loop over efferent neurons to iNeuon: 
				for (jNeuron=0; jNeuron<nEfferent[iNeuron]; jNeuron++){

					// Save external index of efferent neuron, thus we have access to local (jNeuron) and global (iNeuron_summy) ID: 
					iNeuron_dummy = efferentNeurons[iNeuron][jNeuron]; 

					if ((int)packetTimes[iNeuron].size() <= jNeuron){
						packetTimes[iNeuron].resize(nEfferent[iNeuron]); 
						packetFractions[iNeuron].resize(nEfferent[iNeuron]); 
					}

					// If the efferent neuron has spiked or has received a new spike, we need to update its incoming weights. 
					// Note that these incoming weights (afferent to iNeuron_dummy) are stored in w as the efferent weights of iNeuron. 
					if (neurons[iNeuron_dummy].getHasSpiked() or neurons[iNeuron_dummy].getWasSpoken()){

						for (int ii=0; ii<nAfferent[iNeuron_dummy]; ii++){
							if (afferentNeurons[iNeuron_dummy][ii]==iNeuron) iSuperDummy=ii; 
						}
						(void) iSuperDummy; // kept for parity with the original bookkeeping; unused here 

						// We will need to modify w, so start deltaW: 
						deltaW = 0.; 

						// BRANCH 1: post just spiked -- score against every OLDER micro-spike this
						// edge has sent (today's packet, if any, gets appended further below, so it
						// is deliberately not visible here yet). 
						if (neurons[iNeuron_dummy].getHasSpiked()){
							vector<double> &pTimes = packetTimes[iNeuron][jNeuron]; 
							vector<double> &pFracs = packetFractions[iNeuron][jNeuron]; 
							for (size_t k=0; k<pTimes.size(); k++){
								deltaT = pTimes[k] - t; 
								deltaW += pFracs[k] * rwN::deltaW_STDP(deltaT, amplitudeSTDP, sigmaSTDP); 
							}
						}

						// BRANCH 2: pre just spiked, delivering a NEW packet along this edge --
						// score each of its micro-spikes against ALL of post's own past spikes
						// (including one from this very timestep, if simultaneous). 
						if (neurons[iNeuron].getHasSpiked()){
							vector<double> newTimes, newFracs; 
							computePacketShape(iNeuron, jNeuron, t, newTimes, newFracs); 

							if (neurons[iNeuron_dummy].getEfferentSpikeTrain().size()){
								for (size_t k=0; k<newTimes.size(); k++){
									for (iSpike=0; iSpike<neurons[iNeuron_dummy].getEfferentSpikeTrain().size(); iSpike++){
										deltaT = newTimes[k] - neurons[iNeuron_dummy].getEfferentSpike(iSpike); 
										deltaW += newFracs[k] * rwN::deltaW_STDP(deltaT, amplitudeSTDP, sigmaSTDP); 
									}
								}
							}

							// Record this new packet for future lookups by branch 1, once post
							// spikes again later. Appended AFTER scoring above, so this timestep's
							// own branch-1 pass (if it also ran) never sees these entries. 
							for (size_t k=0; k<newTimes.size(); k++){
								packetTimes[iNeuron][jNeuron].push_back(newTimes[k]); 
								packetFractions[iNeuron][jNeuron].push_back(newFracs[k]); 
							}
						}

						// After all contributions have been calculated, correct the weights accordingly: 
						w[iNeuron][jNeuron] += deltaW; 
						w[iNeuron][jNeuron] = max(0., min(1., w[iNeuron][jNeuron])); 
					}

					// Drop retained micro-spike records for this edge that have aged out of the
					// STDP window, same cutoff neurons[].cleanEfferentSpikeTrain() uses in update(). 
					cleanPacketHistory(iNeuron, jNeuron, t - 5*sigmaSTDP); 
				}
			}

			return; 
		}

		void RWNetwork::performSTDP_verbose(int t){
			/* performSTDP function: 

				This function implements the STDP step. 

				ACHTUNG!! 

				This is the same function as above, with running on verbose more. This produces a detailed
				output of the spikes involved and what changes each weight undergoes. 

				This function is meant for debugging purposes only. 

				NOT UPDATED for Gaussian spike packets: this still uses the original single-nominal-
				delay STDP formulas (one deltaT per spike event, not per micro-spike), so its printed
				deltaT/deltaW trace will NOT match performSTDP()'s actual per-micro-spike corrections
				whenever packetSigma > 0. It's still correct/useful for debugging with packetSigma<=0
				(single discrete spike), but don't rely on it to inspect packet-based STDP behavior --
				extend it the same way performSTDP() was extended (computePacketShape() +
				packetTimes/packetFractions) if you need a verbose trace of packet-based corrections. 


				Inputs: 
					>> t: Time. 

			*/


			cout << "Time: " << t << endl;
			int iSuperDummy;  

			// Loop over all neurons to update the weights: 
			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){

				// Loop over efferent neurons to iNeuon: 
				for (jNeuron=0; jNeuron<nEfferent[iNeuron]; jNeuron++){

					// Save external index of efferent neuron, thus we have access to local (jNeuron) and global (iNeuron_summy) ID: 
					iNeuron_dummy = efferentNeurons[iNeuron][jNeuron]; 

					// If the efferent neuron has spiked or has received a new spike, we need to update its incoming weights. 
					// Note that these incoming weights (afferent to iNeuron_dummy) are stored in w as the efferent weights of iNeuron. 
					if (neurons[iNeuron_dummy].getHasSpiked() or neurons[iNeuron_dummy].getWasSpoken()){


						for (int ii=0; ii<nAfferent[iNeuron_dummy]; ii++){
							if (afferentNeurons[iNeuron_dummy][ii]==iNeuron) iSuperDummy=ii; 
						}

						cout << "Spike involving neurons " << iNeuron << " and " << iNeuron_dummy << ": " << endl; 
						cout << "\tW " << iNeuron << " --> " << iNeuron_dummy << ": " << w[iNeuron][jNeuron] << endl; 
						cout << "\tW " << iNeuron_dummy << " --> " << iNeuron << ": " << w[iNeuron_dummy][iSuperDummy] << endl << endl; 
						
						// We will need to modify w, so start deltaW: 
						deltaW = 0.; 

						// If it has spiked, we compare the new spike to its input spikes: 
						if (neurons[iNeuron_dummy].getHasSpiked()){
							cout << "\t\tEfferent spike from " << iNeuron_dummy << ": " << endl; 
							cout << "\t\t\tEmitted at time " << t << endl; 
							// New efferent spike! 
							// 	We need to compare all afferent spikes to the new efferent spike. 
							//	For each afferent spike, an STDP correction must be done. 
							if (neurons[iNeuron].getEfferentSpikeTrain().size()){
								for (iSpike=0; iSpike<neurons[iNeuron].getEfferentSpikeTrain().size(); iSpike++){
									cout << "\t\t\tCompared with spike at time " << neurons[iNeuron].getEfferentSpike(iSpike); 
									cout << " arriving at " << neurons[iNeuron].getEfferentSpike(iSpike) + delays[iNeuron][jNeuron] << endl; 
									deltaT = neurons[iNeuron].getEfferentSpike(iSpike) + delays[iNeuron][jNeuron] - t; 
									deltaW += rwN::deltaW_STDP(deltaT, amplitudeSTDP, sigmaSTDP); 
									cout << "\t\t\tdeltaT: " << deltaT << " and deltaW: " << deltaW << endl; 
								}
							}
						}

						// If, instead, it has received a new spike, it needs to compare this new input with spikes that
						// the neuron previously produced -- but only from the afferent neurons that spiked in this round: 
						if (neurons[iNeuron_dummy].getWasSpoken() and neurons[iNeuron].getHasSpiked()){
							cout << "\t\tAfferent spike from " << iNeuron << " to " << iNeuron_dummy << ": " << endl; 
							cout << "\t\t\tEmitted at time " << t << " arriving at " << t + delays[iNeuron][jNeuron] << endl; 
							// New afferent spike! 
							// 	We need to compare all efferent spikes to the new afferent spike. 
							// 	For each spike, an STDP correction must be done. 
							if (neurons[iNeuron_dummy].getEfferentSpikeTrain().size()){
								for (iSpike=0; iSpike<neurons[iNeuron_dummy].getEfferentSpikeTrain().size(); iSpike++){
									cout << "\t\t\tCompared with spike at time " << neurons[iNeuron_dummy].getEfferentSpike(iSpike) << endl; 
									deltaT = t + delays[iNeuron][jNeuron] - neurons[iNeuron_dummy].getEfferentSpike(iSpike); 
									deltaW += rwN::deltaW_STDP(deltaT, amplitudeSTDP, sigmaSTDP); 
									cout << "\t\t\tdeltaT: " << deltaT << " and deltaW: " << deltaW << endl; 
								}

								// If the neuron both spiked and received a spike, this contribution was scored twice. 
								// We need to remove one of the contributions: 
								if (neurons[iNeuron_dummy].getHasSpiked()){
									cout << "\t\tDouble score! Removing one fo the contributions! " << endl; 
									deltaW -= rwN::deltaW_STDP(deltaT, amplitudeSTDP, sigmaSTDP); 
									cout << "\t\t\tdeltaT: " << deltaT << " and deltaW: " << deltaW << endl; 
								}
							}
						}

						// After all contributions have been calculated, correct the weights accordingly: 
						w[iNeuron][jNeuron] += deltaW; 
						w[iNeuron][jNeuron] = max(0., min(1., w[iNeuron][jNeuron])); 

						cout << "\tW " << iNeuron << " --> " << iNeuron_dummy << ": " << w[iNeuron][jNeuron] << endl; 
						cout << "\tW " << iNeuron_dummy << " --> " << iNeuron << ": " << w[iNeuron_dummy][iSuperDummy] << endl << endl; 
					}
				}
			}

			return; 
		}

		void RWNetwork::setSynapticPacketShape(double packetSigma_, double packetWindowSigmas_){
			/* setSynapticPacketShape function: 

				Sets the temporal shape of internally-delivered efferent spikes. Instead of arriving
				as a single discrete spike at t+delay, each efferent spike is spread out into a
				volley of micro-spikes discretized from a Gaussian of std. dev. packetSigma_ (in
				timesteps), truncated at +/- packetWindowSigmas_ sigmas, and centered at t+delay. The
				micro-spike biases are normalized so they sum to w[iNeuron][jNeuron] -- i.e. this
				changes the temporal profile of a synaptic event, not its total integrated strength. 

				Inputs: 
					>> packetSigma_: Std. dev. (timesteps) of the Gaussian micro-spike packet. Pass
						0 (or leave at the default) to keep the original single discrete spike. 
					>> packetWindowSigmas_: Truncate the packet at +/- this many sigmas. 

			*/

			packetSigma = packetSigma_; 
			packetWindowSigmas = packetWindowSigmas_; 

			return; 
		}

		void RWNetwork::computePacketShape(int iNeuron_, int jNeuron_, double t, vector<double> &times, vector<double> &fractions){
			/* computePacketShape function: 

				Deterministically computes the discretized-Gaussian micro-spike offsets for the
				efferent spike from iNeuron_ along its jNeuron_-th connection, given the spike was
				detected at time t. Returns FRACTIONS (each micro-spike's share of the packet,
				summing to 1), not biases -- deliberately independent of w[iNeuron_][jNeuron_], so
				this same shape can be reused both to schedule the actual delivery (deliverSpikePacket,
				which multiplies by the CURRENT w to get biases) and to weight STDP corrections in
				performSTDP (which cares about the temporal profile, not the raw synaptic strength). 
				Calling this twice with the same arguments always returns the same times/fractions. 

				Inputs: 
					>> iNeuron_: Index of the spiking (afferent) neuron. 
					>> jNeuron_: Local index of the target within iNeuron_'s efferent list. 
					>> t: Time the spike was detected. 
					>> times, fractions: Output vectors (cleared and filled by this function). 

			*/

			times.clear(); 
			fractions.clear(); 
			double centerTime = t + delays[iNeuron_][jNeuron_]; 

			// packetSigma<=0: single discrete "packet" of one micro-spike carrying the whole
			// packet (fraction 1.0) -- reduces every downstream computation to the original
			// single-discrete-spike behavior. 
			if (packetSigma <= 0.){
				times.push_back(centerTime); 
				fractions.push_back(1.0); 
				return; 
			}

			int halfWidth = (int) ceil(packetWindowSigmas * packetSigma); 
			vector<double> gaussWeights; 
			double sumG = 0.; 
			for (int dt=-halfWidth; dt<=halfWidth; dt++){
				double spikeTime = centerTime + dt; 
				if (spikeTime < t) continue; // no delivering "before now" -- causality guard 

				double gaussVal = exp(-(double(dt)*dt) / (2.0*packetSigma*packetSigma)); 
				gaussWeights.push_back(gaussVal); 
				times.push_back(spikeTime); 
				sumG += gaussVal; 
			}

			// Degenerate case (e.g. the whole window got clipped by the causality guard): fall
			// back to a single micro-spike at the center time. 
			if (sumG <= 0. or gaussWeights.empty()){
				times.clear(); 
				times.push_back(centerTime); 
				fractions.push_back(1.0); 
				return; 
			}

			for (size_t k=0; k<gaussWeights.size(); k++){
				fractions.push_back(gaussWeights[k] / sumG); 
			}

			return; 
		}

		void RWNetwork::deliverSpikePacket(int iNeuron_, int jNeuron_, int iNeuron_dummy_, double t){
			/* deliverSpikePacket function: 

				Delivers the efferent spike from iNeuron_ to iNeuron_dummy_ (its jNeuron_-th efferent
				neuron) as a Gaussian-shaped micro-spike packet centered at t+delay, instead of a
				single discrete spike. Uses computePacketShape() for the (deterministic) temporal
				shape, then scales each micro-spike's fraction by the CURRENT w[iNeuron_][jNeuron_]
				to get its actual bias -- so the whole packet still sums to exactly
				w[iNeuron_][jNeuron_] (weight-conserving), same as before. 

				performSTDP() independently recomputes the identical shape (same t, same edge, same
				deterministic function) to score each micro-spike against the postsynaptic spike
				history -- see performSTDP() for how that lets different micro-spikes within this
				SAME packet potentiate or depress depending on their own arrival time. 

				Inputs: 
					>> iNeuron_: Index of the spiking (afferent) neuron. 
					>> jNeuron_: Local index of the target within iNeuron_'s efferent list. 
					>> iNeuron_dummy_: Global index of the target (efferent) neuron. 
					>> t: Current simulation time (the spike was just detected at this timestep). 

			*/

			vector<double> times, fractions; 
			computePacketShape(iNeuron_, jNeuron_, t, times, fractions); 

			double totalWeight = w[iNeuron_][jNeuron_]; 
			for (size_t k=0; k<times.size(); k++){
				neurons[iNeuron_dummy_].receiveSpike(times[k], totalWeight * fractions[k]); 
			}

			return; 
		}

		void RWNetwork::cleanPacketHistory(int iNeuron_, int jNeuron_, double threshold){
			/* cleanPacketHistory function: 

				Drops retained micro-spike records for edge (iNeuron_ -> its jNeuron_-th efferent)
				older than threshold. Mirrors RWNeuron::cleanEfferentSpikeTrain(), and is called every
				timestep for every edge from performSTDP() with the same 5*sigmaSTDP window already
				used there, so retained history never grows unbounded. 

				Inputs: 
					>> iNeuron_: Index of the presynaptic neuron. 
					>> jNeuron_: Local index of the efferent connection. 
					>> threshold: Time before which all records should be dropped. 

			*/

			if (not packetTimes or (int)packetTimes[iNeuron_].size() <= jNeuron_) return; 

			vector<double> &pTimes = packetTimes[iNeuron_][jNeuron_]; 
			vector<double> &pFracs = packetFractions[iNeuron_][jNeuron_]; 
			while (pTimes.size() and pTimes[0] <= threshold){
				pTimes.erase(pTimes.begin()); 
				pFracs.erase(pFracs.begin()); 
			}

			return; 
		}

		void RWNetwork::deliverExternalSpike(double t,  double externalBias=1.){
			/* deliverExternalSpike function: 

				This function delivers an external spike to all neurons. 

				Inputs: 
					>> t: Time. 
					>> externalBias: Size of the external signal. 

			*/

			for (iNeuron=0; iNeuron<nNeurons; iNeuron++){
					
				neurons[iNeuron].receiveExternalSpike(t+externalDelays[iNeuron], externalBias*externalW[iNeuron]); 
			}
				
			return; 
		}

		void RWNetwork::deliverExternalSpike(double t, double externalBias, vector<int> &spokenNeurons){
			/* deliverExternalSpike function: 

				This function delivers an external spike of size externalBias to neurons in spokenNeurons. This will
				allow us to make experiments in which different neurons are differently disconnected from the
				externala stimulus. 

				Inputs: 
					>> t: Time. 
					>> externalBias: Size of the external signal. 
					>> spokenNeurons: Neurons that receive the external signal. 

			*/

			for (int i=0; i<spokenNeurons.size(); i++){
				iNeuron = spokenNeurons[i]; 
				neurons[iNeuron].receiveExternalSpike(t+externalDelays[iNeuron], externalBias*externalW[iNeuron]); 
			}

			return; 
		}

		RWNetwork::~RWNetwork(){
    		// delete neuron array
    		if (neurons){
        		delete [] neurons;
        		neurons = nullptr;
    		}

   			 // delete integer arrays
    		if (nAfferent){
        		delete [] nAfferent;
        		nAfferent = nullptr;
    		}
    		if (nEfferent){
        		delete [] nEfferent;
        		nEfferent = nullptr;
    		}

    		// delete arrays of vectors
    		if (afferentNeurons){
        		delete [] afferentNeurons;
        		afferentNeurons = nullptr;
    		}
    		if (efferentNeurons){
        		delete [] efferentNeurons;
        		efferentNeurons = nullptr;
    		}

    		// delete arrays of vector<double>
    		if (w){
        		// clear inner vectors (not strictly necessary before delete[])
        		for (int i = 0; i < nNeurons; ++i) w[i].clear();
        		delete [] w;
        		w = nullptr;
    		}
    		if (delays){
        		for (int i = 0; i < nNeurons; ++i) delays[i].clear();
       			delete [] delays;
        		delays = nullptr;
    		}

    		// delete external arrays if allocated
    		if (externalW){
        		delete [] externalW;
        		externalW = nullptr;
    		}
    		if (externalDelays){
        		delete [] externalDelays;
        		externalDelays = nullptr;
    		}

    		// delete wMatrix if it was allocated and still points to memory
    		if (wMatrix){
        		// wMatrix is a double** where each row was allocated with new double[nNeurons]
        		for (int i = 0; i < nNeurons; ++i){
            		if (wMatrix[i]) {
                		delete [] wMatrix[i];
                		wMatrix[i] = nullptr;
            		}
        		}
        		delete [] wMatrix;
       			wMatrix = nullptr;
    		}

    		// delete retained micro-spike packet history, if it was ever allocated (lazily, on
    		// first performSTDP() call)
    		if (packetTimes){
        		delete [] packetTimes;
        		packetTimes = nullptr;
    		}
    		if (packetFractions){
        		delete [] packetFractions;
        		packetFractions = nullptr;
    		}

    		// reset bookkeeping
    		nNeurons = 0;
    		amplitudeSTDP = sigmaSTDP = 0.0;
		}



}