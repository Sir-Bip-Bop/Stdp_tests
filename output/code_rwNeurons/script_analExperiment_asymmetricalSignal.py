"""

	script_analExperiment_asymmetricalSignal.py: 

		This script analysis the results of the experiment with asymmetrical signals. 

"""

# Imports: 
import numpy as np; 
import matplotlib.pyplot as plt; 
from mpl_toolkits import mplot3d; 
import os, sys; 
from copy import copy; 
from scipy import stats; 


dataPathIn = "Codigo/output/code_rwNeurons/Experiment_AsymmetricSignal/"; 
dataPathOut = "Codigo/output/code_rwNeurons/Experiment_AsymmetricSignal/"; 
picsPathOut = "Codigo/output/code_rwNeurons/Pics/"; 


# My name is Python and I don't know how to deal with doubles properly: 
externalBiases = np.arange(0,1,0.01); 
externalBiases = [round(ee, 2) for ee in externalBiases]; 

## Loop over rightExternalBias: 

# Init variables to store results: 
meanLL = []; meanLR = []; meanRL = []; meanRR = []; 
stdLL = []; stdLR = []; stdRL = []; stdRR = []; 
wMeanLL = []; wMeanLR = []; wMeanRL = []; wMeanRR = []; 
# Actual loop: 
for thisRightExternalBias in externalBiases: 
	# Reading summary files: 
	if (thisRightExternalBias!=0): 
		thisSummary = np.genfromtxt(dataPathIn + "summary_" + str(thisRightExternalBias) + ".csv", delimiter=', '); 
	else: 
		thisSummary = np.genfromtxt(dataPathIn + "summary_" + str(int(thisRightExternalBias)) + ".csv", delimiter=', '); 

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
plt.plot(externalBiases, wMeanLL, 'ok', color = 'grey', alpha = 0.2); 
plt.fill_between(externalBiases, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL), color= 'grey'); 
plt.plot(externalBiases, meanLL, color = 'black', label = "L-->L"); 
# Plotting LR: 
plt.fill_between(externalBiases, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR), color= 'lightcoral'); 
plt.plot(externalBiases, meanLR, color = 'red', label = "L-->R"); 
# Plotting RL: 
plt.fill_between(externalBiases, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL), color= 'lightblue'); 
plt.plot(externalBiases, meanRL, color = 'blue', label = "R-->L"); 
# Plotting RR: 
plt.plot(externalBiases, wMeanRR, 'ok', color = 'lightgreen', alpha = 0.2);
plt.fill_between(externalBiases, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR), color= 'lightgreen'); 
plt.plot(externalBiases, meanRR, color = 'green', label = "R-->R"); 
plt.ylim([0,0.7]); 
plt.xlabel("External bias on right barrel"); 
plt.ylabel("Weights"); 
plt.legend();
fig.savefig(picsPathOut + "fig_asymmetricBias_all.pdf"); 


## Plotting each summary separatedly: 

# Plotting LL: 
fig = plt.figure(); 
plt.plot(externalBiases, wMeanLL, 'ok'); 
plt.fill_between(externalBiases, np.subtract(meanLL, stdLL), np.add(meanLL, stdLL)); 
plt.plot(externalBiases, meanLL, 'k'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("L-->L"); 
fig.savefig(picsPathOut + "fig_asymmetricBias_LL.pdf"); 

# Plotting LR: 
fig = plt.figure(); 
plt.plot(externalBiases, wMeanLR, 'ok'); 
plt.fill_between(externalBiases, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR)); 
plt.plot(externalBiases, meanLR, 'r'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("L-->R"); 
fig.savefig(picsPathOut + "fig_asymmetricBias_LR.pdf"); 


# Plotting RL: 
fig = plt.figure(); 
plt.plot(externalBiases, wMeanRL, 'ok', color = 'lightblue', alpha = 0.2); 
plt.fill_between(externalBiases, np.subtract(meanRL, stdRL), np.add(meanRL, stdRL), color= 'lightblue'); 
plt.plot(externalBiases, meanRL, color = 'blue', label = "R-->L"); 
plt.plot(externalBiases, wMeanLR, 'ok', alpha = 0.2, color = 'lightcoral'); 
plt.fill_between(externalBiases, np.subtract(meanLR, stdLR), np.add(meanLR, stdLR),color= 'lightcoral'); 
plt.plot(externalBiases, meanLR, color = 'red', label = "L-->R"); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("R-->L"); 
plt.legend();
fig.savefig(picsPathOut + "fig_asymmetricBias_RL.pdf"); 

# Plotting RR: 
fig = plt.figure(); 
plt.plot(externalBiases, wMeanRR, 'ok'); 
plt.fill_between(externalBiases, np.subtract(meanRR, stdRR), np.add(meanRR, stdRR)); 
plt.plot(externalBiases, meanRR, 'g'); 
plt.ylim([0,0.7]); 
plt.xlabel("Right external bias"); 
plt.ylabel("Weights"); 
plt.title("R-->R"); 
fig.savefig(picsPathOut + "fig_asymmetricBias_RR.pdf"); 




plt.show(); 







