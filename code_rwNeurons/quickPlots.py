"""

	quickPlots.py: 

		This is a quick script to generate plots that test random neuroms. 

"""


# Imports: 
import numpy as np; 
import matplotlib.pyplot as plt; 
from mpl_toolkits import mplot3d; 
import os, sys; 
from copy import copy; 
from scipy import stats; 



dataPathIn = "./TestOut/"; 
picsPathOut = "../Documents/Pics/PreliminaryPics/Raw/"; 


## Uncomment for STDP: 

# x = np.arange(-3,3.01,0.01); 
# sigmaSTDP = 1.; 
# fx = -x*np.exp( -0.5*(x**2)/(sigmaSTDP**2) )/((sigmaSTDP**3)*np.sqrt(2*np.pi)); 

# fig = plt.figure(); 
# plt.plot(x, fx); 
# fig.savefig(picsPathOut + "fSTDP.pdf"); 

# plt.show(); 
# sys.exit(0); 




externalDelays = np.loadtxt(dataPathIn + "externalDelays.csv", delimiter=','); 
internalDelays = np.loadtxt(dataPathIn + "internalDelays.csv", delimiter=','); 
startWeights = np.loadtxt(dataPathIn + "startW.csv", delimiter=','); 
endWeights = np.loadtxt(dataPathIn + "endW.csv", delimiter=','); 
v = np.loadtxt(dataPathIn + "runV.csv", delimiter=','); 


plt.figure(); 
plt.plot(externalDelays); 

fig = plt.figure(); 
plt.hist(externalDelays, 50); 

fig = plt.figure(); 
plt.imshow(internalDelays); 
plt.colorbar(); 


fig = plt.figure(); 
plt.imshow(startWeights, vmin=0, vmax=1); 
plt.colorbar(); 
fig.savefig(picsPathOut + "startWeights.pdf"); 

fig = plt.figure(); 
plt.imshow(endWeights, vmin=0, vmax=1); 
plt.colorbar(); 
fig.savefig(picsPathOut + "endWeights_noExternalInput.pdf"); 
# fig.savefig(picsPathOut + "endWeights_withExternalInput.pdf"); 

# plt.figure(); 
# (fig, ax) = plt.subplot(3, 1, 1); 
# ax[0].plt.imshow(np.transpose(v[0:500,:])); 
# ax[1].plt.imshow(np.transpose(v[501:1000,:])); 
# ax[2].plt.imshow(np.transpose(v[1000:1500,:])); 
# # plt.imshow(np.transpose(v)); 
# plt.colorbar(); 

plt.show(); 


