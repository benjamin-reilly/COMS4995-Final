#Just a little workspace for playing around with the parsed data
#output by data_parser.py.
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re #regex
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

#Load in data:
data = np.load('aggregateData_raw.npy')

fontsize=20

#First plot tau vs. theta:
x = data[:,0]
y = data[:,-1]/(1e-9)

fig, ax = plt.subplots(figsize=(7,7))
#plt.plot(x,y,'k.')
ax.plot(x,y,'k.') #ax.semilogy(x,y,'k.') #
ax.grid(visible=True)
#ax.set_xlabel(r'Melt depth, $\delta$ (nm)',fontsize=fontsize)
ax.set_xlabel(r'Contact angle, $\theta$',fontsize=fontsize)
ax.set_ylabel(r'Melt lifetime, $\tau$ (ns)',fontsize=fontsize)
#ax.set_title(r'$\tau$ vs. $\delta$')
ax.set_title(r'$\tau$ vs. $\theta$')
ax.set_xticks([60,75,90,105,120])

plt.savefig('tau_vs_theta.png',bbox_inches='tight',pad_inches = 0.25)
plt.close()


#Now plot delta vs. theta:
x = data[:,0]
y = data[:,-5]/(1e-9)

fig, ax = plt.subplots(figsize=(7,7))
#plt.plot(x,y,'k.')
ax.plot(x,y,'k.') #ax.semilogy(x,y,'k.') #
ax.grid(visible=True)
#ax.set_xlabel(r'Melt depth, $\delta$ (nm)',fontsize=fontsize)
ax.set_xlabel(r'Contact angle, $\theta$',fontsize=fontsize)
ax.set_ylabel(r'Melt depth, $\delta$ (nm)',fontsize=fontsize)
#ax.set_title(r'$\tau$ vs. $\delta$')
ax.set_title(r'$\delta$ vs. $\theta$')
ax.set_xticks([60,75,90,105,120])

plt.savefig('delta_vs_theta.png',bbox_inches='tight',pad_inches = 0.25)
plt.close()


#Now plot tau vs. delta, with different markers for different
#theta:
x = data[:,-5]/(1e-9)
y = data[:,-1]/(1e-9)

I_60 = data[:,0] == 60
x_60 = x[I_60]
y_60 = y[I_60]

I_75 = data[:,0] == 75
x_75 = x[I_75]
y_75 = y[I_75]

I_90 = data[:,0] == 90
x_90 = x[I_90]
y_90 = y[I_90]

I_105 = data[:,0] == 105
x_105 = x[I_105]
y_105 = y[I_105]

I_120 = data[:,0] == 120
x_120 = x[I_120]
y_120 = y[I_120]


fig, ax = plt.subplots(figsize=(7,7))
#plt.plot(x,y,'k.')
ax.plot(x_60,y_60,'.',color='red')
ax.plot(x_75,y_75,'v',color='limegreen')
ax.plot(x_90,y_90,'^',color='black')
ax.plot(x_105,y_105,'s',color='dodgerblue')
ax.plot(x_120,y_120,'*',color='darkviolet')
ax.grid(visible=True)
#ax.set_xlabel(r'Melt depth, $\delta$ (nm)',fontsize=fontsize)
ax.set_xlabel(r'Melt depth, $\delta$ (nm)',fontsize=fontsize)
ax.set_ylabel(r'Melt lifetime, $\tau$ (ns)',fontsize=fontsize)
#ax.set_title(r'$\tau$ vs. $\delta$')
ax.set_title(r'$\tau$ vs. $\delta$')
ax.legend([r'$60^{\circ}$',r'$75^{\circ}$',r'$90^{\circ}$',r'$105^{\circ}$',r'$120^{\circ}$'],loc='lower right')

plt.savefig('tau_vs_delta.png',bbox_inches='tight',pad_inches = 0.25)
plt.close()


print('# of datapoints = ' + str(np.shape(x_60)))