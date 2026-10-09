"""

	script_analExperiment_distance.py: 

		This script analysis the results of the experiment with chaging distances. 

"""

# Imports: 
import numpy as np; 
import matplotlib.pyplot as plt; 
from mpl_toolkits import mplot3d; 
import os, sys; 
from copy import copy; 
from scipy import stats; 


dataPathIn = "Codigo/output/code_rwNeurons/Experiment_Distance/"; 
dataPathOut = "Codigo/output/code_rwNeurons/Experiment_Distance/"; 
picsPathOut = "Codigo/output/code_rwNeurons/Pics/"; 


# Init long delays: 
longDelayMin = 0; 
longDelayMax = 300; 
longDelayStep = 10; 
longDelays = [dd for dd in range(longDelayMin, longDelayMax+longDelayStep, longDelayStep)]; 


## Loop over rightExternalBias: 

# Init variables to store results: 
meanLL = []; meanLR = []; meanRL = []; meanRR = []; 
stdLL = []; stdLR = []; stdRL = []; stdRR = []; 
wMeanLL = []; wMeanLR = []; wMeanRL = []; wMeanRR = []; 
# Actual loop: 
for longDelay in longDelays: 
	# Reading summary files: 
	thisSummary = np.genfromtxt(dataPathIn + "summary_" + str(longDelay) + ".csv", delimiter=', '); 

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
plt.plot(longDelays, wMeanLL, 'ok', color = 'grey', alpha = 0.2); 
plt.fill_between(longDelays, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL), color = 'grey'); 
plt.plot(longDelays, meanLL, color = 'black', label = "L-->L"); 

# Plotting LR: 
plt.plot(longDelays, wMeanLR, 'ok', color = 'lightcoral', alpha = 0.2); 
plt.fill_between(longDelays, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR), color = 'lightcoral'); 
plt.plot(longDelays, meanLR, color = 'red', label = "L-->R"); 

# Plotting RL: 
plt.plot(longDelays, wMeanRL, 'ok', color = 'lightblue',alpha = 0.2);
plt.fill_between(longDelays, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL), color = 'lightblue'); 
plt.plot(longDelays, meanRL, color = 'blue', label = "R-->L"); 

# Plotting RR: 
plt.plot(longDelays, wMeanRR, 'ok', color = 'lightgreen', alpha = 0.2);
plt.fill_between(longDelays, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR), color = 'lightgreen'); 
plt.plot(longDelays, meanRR, color = 'green', label = "R-->R"); 

plt.ylim([0,0.7]); 
plt.xlabel("Inter-barrel distance"); 
plt.ylabel("Weights"); 
plt.legend();
fig.savefig(picsPathOut + "fig_distance_all.pdf"); 



## Plotting each summary separatedly: 

# Plotting LL: 
fig = plt.figure(); 
plt.plot(longDelays, wMeanLL, 'ok'); 
plt.fill_between(longDelays, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL)); 
plt.plot(longDelays, meanLL, 'k'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("L-->L"); 
fig.savefig(picsPathOut + "fig_distance_LL.pdf"); 


# Plotting LR: 
fig = plt.figure(); 
plt.plot(longDelays, wMeanLR, 'ok'); 
plt.fill_between(longDelays, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR)); 
plt.plot(longDelays, meanLR, 'r'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("L-->R"); 
fig.savefig(picsPathOut + "fig_distance_LR.pdf"); 


# Plotting RL: 
fig = plt.figure(); 
plt.plot(longDelays, wMeanRL, 'ok'); 
plt.fill_between(longDelays, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL)); 
plt.plot(longDelays, meanRL, 'y'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("R-->L"); 
fig.savefig(picsPathOut + "fig_distance_RL.pdf"); 


# Plotting RR: 
fig = plt.figure(); 
plt.plot(longDelays, wMeanRR, 'ok'); 
plt.fill_between(longDelays, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR)); 
plt.plot(longDelays, meanRR, 'g'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("R-->R"); 
fig.savefig(picsPathOut + "fig_distance_RR.pdf"); 




plt.show(); 







