"""

	script_analExperiment_externalDelay.py: 

		This script analysis the results of the experiment with chaging broadness of external delays. 

"""

# Imports: 
import numpy as np; 
import matplotlib.pyplot as plt; 
from mpl_toolkits import mplot3d; 
import os, sys; 
from copy import copy; 
from scipy import stats; 


dataPathIn = "Codigo/output/code_rwNeurons/Experiment_ExternalDelay/"; 
dataPathOut = "Codigo/output/code_rwNeurons/Experiment_ExternalDelay/"; 
picsPathOut = "Codigo/output/code_rwNeurons/Pics/"; 


# Init long delays: 
sigmaExternalDelayMin = 10; 
sigmaExternalDelayMax = 200; 
sigmaExternalDelayStep = 10; 
sigmaExternalDelays = [ss for ss in range(sigmaExternalDelayMin, sigmaExternalDelayMax+sigmaExternalDelayStep, sigmaExternalDelayStep)]; 


## Loop over rightExternalBias: 

# Init variables to store results: 
meanLL = []; meanLR = []; meanRL = []; meanRR = []; 
stdLL = []; stdLR = []; stdRL = []; stdRR = []; 
wMeanLL = []; wMeanLR = []; wMeanRL = []; wMeanRR = []; 
# Actual loop: 
for sigmaExternalDelay in sigmaExternalDelays: 
	# Reading summary files: 
	thisSummary = np.genfromtxt(dataPathIn + "summary_" + str(sigmaExternalDelay) + ".csv", delimiter=', '); 

	# Computing means and stds: 
	wMeanLL += [thisSummary[:,0]]; 
	wMeanLR += [thisSummary[:,1]]; 
	wMeanRL += [thisSummary[:,2]]; 
	wMeanRR += [thisSummary[:,3]]; 
	meanLL += [np.mean(wMeanLL[-1])]; stdLL += [np.std(wMeanLL[-1])]; 
	meanLR += [np.mean(wMeanLR[-1])]; stdLR += [np.std(wMeanLR[-1])]; 
	meanRL += [np.mean(wMeanRL[-1])]; stdRL += [np.std(wMeanRL[-1])]; 
	meanRR += [np.mean(wMeanRR[-1])]; stdRR += [np.std(wMeanRR[-1])]; 


## Plotting all summaries together: 

fig = plt.figure(); 
# Plotting LL: 
plt.plot(sigmaExternalDelays, wMeanLL, 'ok', color = 'grey', alpha = 0.2); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL), color = 'grey'); 
plt.plot(sigmaExternalDelays, meanLL, color = 'black', label = "L-->L"); 
# Plotting LR: 
plt.plot(sigmaExternalDelays, wMeanLR, 'ok', color = 'lightcoral', alpha = 0.2); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR), color = 'red'); 
plt.plot(sigmaExternalDelays, meanLR, color = 'red', label = "L-->R"); 
# Plotting RL: 
plt.plot(sigmaExternalDelays, wMeanRL, 'ok', color = 'lightblue',alpha = 0.2);
plt.fill_between(sigmaExternalDelays, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL), color = 'lightblue'); 
plt.plot(sigmaExternalDelays, meanRL, color = 'blue', label = "R-->L"); 
# Plotting RR: 
plt.plot(sigmaExternalDelays, wMeanRR, 'ok', color = 'lightgreen', alpha = 0.2);
plt.fill_between(sigmaExternalDelays, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR), color = 'green'); 
plt.plot(sigmaExternalDelays, meanRR, color = 'green', label = "R-->R"); 
plt.ylim([0,0.7]); 
plt.xlabel("Dispersion of external delay"); 
plt.ylabel("Weights"); 
plt.legend();
fig.savefig(picsPathOut + "fig_sigmaExternalBias_all.pdf"); 



## Plotting each summary separatedly: 

# Plotting LL: 
fig = plt.figure(); 
plt.plot(sigmaExternalDelays, wMeanLL, 'ok'); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL)); 
plt.plot(sigmaExternalDelays, meanLL, 'k'); 
plt.ylim([0,0.7]); 
plt.xlabel("Long delay"); 
plt.ylabel("Weights"); 
plt.title("L-->L"); 
fig.savefig(picsPathOut + "fig_sigmaExternalBias_LL.pdf"); 


# Plotting LR: 
fig = plt.figure(); 
plt.plot(sigmaExternalDelays, wMeanLR, 'ok'); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR)); 
plt.plot(sigmaExternalDelays, meanLR, 'r'); 
plt.ylim([0,0.7]); 
plt.xlabel("Long delay"); 
plt.ylabel("Weights"); 
plt.title("L-->R"); 
fig.savefig(picsPathOut + "fig_sigmaExternalBias_LR.pdf"); 


# Plotting RL: 
fig = plt.figure(); 
plt.plot(sigmaExternalDelays, wMeanRL, 'ok'); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL)); 
plt.plot(sigmaExternalDelays, meanRL, 'y'); 
plt.ylim([0,0.7]); 
plt.xlabel("Long delay"); 
plt.ylabel("Weights"); 
plt.title("R-->L"); 
fig.savefig(picsPathOut + "fig_sigmaExternalBias_RL.pdf"); 


# Plotting RR: 
fig = plt.figure(); 
plt.plot(sigmaExternalDelays, wMeanRR, 'ok'); 
plt.fill_between(sigmaExternalDelays, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR)); 
plt.plot(sigmaExternalDelays, meanRR, 'g'); 
plt.ylim([0,0.7]); 
plt.xlabel("Long delay"); 
plt.ylabel("Weights"); 
plt.title("R-->R"); 
fig.savefig(picsPathOut + "fig_sigmaExternalBias_RR.pdf"); 




plt.show(); 







