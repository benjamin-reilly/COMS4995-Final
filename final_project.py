"""bsr2138 -- HW2"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import final_classes_and_functions as fin
import dgh
import time
import scipy.spatial.distance as spd
import matplotlib as mpl
import matplotlib.tri as mtri
plt.rcParams['text.usetex'] = True
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

tol = 1e-12 #tolerance for LLE zero eigenvalues

"""Q8"""
theta_list = [60,75,90,105,120]
for theta in theta_list:
    filename = 'theta_' + str(theta) + '_zScore.npy'
    #filename = 'aggregateData_zScore.npy'
    eps = 1.5 #3.5 #4
    sigma = 0.5 #0.75
    m = 10 #number of nearest neighbors when constructing nearest-neighbor weights (if desired)

    #Function which performs single-linkage clustering for the
    #second and third datasets over ranges of epsilon values, in
    #order to see if there is a range of epsilon values which appears
    #to give stable clusterings (and would therefore be a good guess
    #to use for the IsoMap embeddings):
    def determine_epsilon():
        global filename

        N = 50 #number of distinct epsilons to sample SLC over
        #epsilon_vec = np.linspace(20,160,N)
        epsilon_vec = np.linspace(0,3,N)

        X = np.load(filename)
        data_now = fin.data(X)
        n_SLC = np.zeros(N) #pre-allocating number of clusters returned by SLC for each epsilon
        for ind in range(0,len(epsilon_vec)):
            (clusters_SLC,cluster_ind_SLC) = data_now.singleLinkageClustering(epsilon_vec[ind])
            n_SLC[ind] = len(clusters_SLC)
        fig, ax = plt.subplots(figsize=(7,7))
        plt.plot(epsilon_vec,n_SLC,'k.-')
        plt.plot(epsilon_vec[n_SLC==2],n_SLC[n_SLC==2],'r.')
        ax.set_title('Number of SLC clusters vs. $\epsilon$ for ' + filename,fontsize=20)
        ax.set_xlabel('$\epsilon$',fontsize=20)
        ax.set_ylabel('$n_{c}$',fontsize=20)
        plt.grid(visible=True)
        plt.savefig('n_vs_epsilon_theta_' + str(theta) + '.png',bbox_inches='tight',pad_inches = 0.25)
        plt.close()




    #Defining the embedding of the full datasets
    #as its own function, since this takes a while
    #for the second dataset ('ps2-data-1.txt'):
    def full_embedded_plots():
        fontsize=20 #fontsize for plots

        global filename, eps, sigma, m

        #Load data (omit first column, which simply numbers the datapoints):
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)

        #print(np.min(data_now.dX[np.nonzero(data_now.dX)]))
        #print(np.max(data_now.dX))
        #print(np.mean(data_now.dX))

        #Use the four methods to embed the data in k = 1-dimensional space:
        k = 1
        #MDS embedding:
        MDS = data_now.MDS(k)
        #IsoMap embedding:
        epsilon = eps #epsilon parameter for the single-linkage clustering steep of IsoMap
        #ISO = data_now.IsoMap(k,epsilon) 
        ISO = data_now.IsoMap_kNN(k,m)
        #Laplacian eigenmaps embedding (with Gaussian weights):
        #data_now.constructGaussianWeights(sigma=sigma)
        data_now.constructNearestNeighborWeights(m=m)
        LEM = data_now.LaplacianEigenmaps(k)
        #LLE embedding:
        LLE = data_now.LLE(k,tol=tol)

        data_list = [MDS,ISO,LEM,LLE]
        titles = [r'MDS embedding with $k$ = ' + str(k), \
                r'IsoMap embedding with k = ' + str(k) + ' and $m$ = ' + str(m), \
                #r'Eigenmap embedding with $k$ = ' + str(k) + ' and $\sigma$ = ' + str(sigma), \
                r'Eigenmap embedding with $k$ = ' + str(k) + ' and $m$ = ' + str(m), \
                r'Locally Linear embedding with $k$ = ' + str(k)]

        #Plot results:
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,7))
        ind = 0
        for i in range(0,2):
            for j in range(0,2):
                ax[i,j].plot(data_list[ind],np.zeros(np.shape(data_list[ind])[0]),'r.')
                ax[i,j].set_title(titles[ind])
                ax[i,j].grid(visible=True)
                ax[i,j].set_xlabel(r'$y_{i}$',fontsize=fontsize,labelpad=15)
                ax[i,j].set_yticks([])
                ind += 1
        fig.tight_layout(pad=5.0)
        plt.savefig('k=1_embedding_theta_' + str(theta) + '.png',bbox_inches='tight',pad_inches = 0.25)
        plt.close()


        #Now let's repeat the same process, but with k = 2:

        k = 2
        #MDS embedding:
        MDS = data_now.MDS(k)
        #IsoMap embedding:
        epsilon = eps #epsilon parameter for the single-linkage clustering steep of IsoMap
        #ISO = data_now.IsoMap(k,epsilon) 
        ISO = data_now.IsoMap_kNN(k,m)
        #Laplacian eigenmaps embedding (with Gaussian weights):
        #data_now.constructGaussianWeights(sigma=sigma)
        data_now.constructNearestNeighborWeights(m=m)
        LEM = data_now.LaplacianEigenmaps(k)
        #LLE embedding:
        LLE = data_now.LLE(k,tol=tol)

        data_list = [MDS,ISO,LEM,LLE]
        titles = [r'MDS embedding with $k$ = ' + str(k), \
                r'IsoMap embedding with k = ' + str(k) + ' and $m$ = ' + str(m), \
                #r'Eigenmap embedding with $k$ = ' + str(k) + ' and $\sigma$ = ' + str(sigma), \
                r'Eigenmap embedding with $k$ = ' + str(k) + ' and $m$ = ' + str(m), \
                r'Locally Linear embedding with $k$ = ' + str(k)]

        #Plot results:
        fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8,8))
        ind = 0
        for i in range(0,2):
            for j in range(0,2):
                ax[i,j].plot(data_list[ind][:,0],data_list[ind][:,1],'r.')
                ax[i,j].set_title(titles[ind])
                ax[i,j].grid(visible=True)
                ax[i,j].set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
                ax[i,j].set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
                ind += 1
        fig.tight_layout(pad=6.0)
        plt.savefig('k=2_embedding_theta_' + str(theta) + '.png',bbox_inches='tight',pad_inches = 0.25)
        plt.close()


        #Now let's repeat the same process, but with k = 3:
        k = 3
        #MDS embedding:
        MDS = data_now.MDS(k)
        #IsoMap embedding:
        epsilon = eps #epsilon parameter for the single-linkage clustering steep of IsoMap
        #ISO = data_now.IsoMap(k,epsilon) 
        ISO = data_now.IsoMap_kNN(k,m)
        #Laplacian eigenmaps embedding (with Gaussian weights):
        #data_now.constructGaussianWeights(sigma=sigma)
        data_now.constructNearestNeighborWeights(m=m)
        LEM = data_now.LaplacianEigenmaps(k)
        #LLE embedding:
        LLE = data_now.LLE(k,tol=tol)

        data_list = [MDS,ISO,LEM,LLE]
        titles = [r'MDS embedding with $k$ = ' + str(k), \
                r'IsoMap embedding with k = ' + str(k) + ' and $m$ = ' + str(m), \
                #r'Eigenmap embedding with $k$ = ' + str(k) + ' and $\sigma$ = ' + str(sigma), \
                r'Eigenmap embedding with $k$ = ' + str(k) + ' and $m$ = ' + str(m), \
                r'Locally Linear embedding with $k$ = ' + str(k)]

        #Plot results:
        #fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,7))
        #fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(7,7))
        fig = plt.figure(figsize=(8,8))
        ind = 0
        for i in range(0,2):
            for j in range(0,2):


                ax = fig.add_subplot(2, 2, ind+1, projection='3d')

                #Plotting clusters from single-linkage clustering:
                ax.view_init(25,40,0)
                my_cmap = plt.get_cmap('winter')
                ax.scatter3D(data_list[ind][:,0],data_list[ind][:,1],data_list[ind][:,2],c=data_list[ind][:,2],cmap=my_cmap)
                ax.set_title(titles[ind])
                ax.grid(visible=True)
                ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
                ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
                ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
                ind += 1
        fig.tight_layout(pad=6.0)
        plt.savefig('k=3_embedding_theta_' + str(theta) + '.png',bbox_inches='tight',pad_inches = 0.35)
        plt.close()


    #Function to animate the isomap embeddings over a range of
    #different epsilon values used to construct the single-linkage
    #connectivity:
    def isomap_animation_single():
        fontsize=20 #fontsize for plots

        global filename, eps, sigma

        #Desired epsilon values:
        eps_vec = np.linspace(0.1,5,100) #np.linspace(27,35,100)


        #First animate for k = 1:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 1
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(eps_vec)))
        ind = 0
        for eps in eps_vec:
            ISO = data_now.IsoMap(k,eps)
            Isomaps[:,:,ind] = ISO
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Isomaps[:,:,0],np.zeros(np.shape(ISO)[0]),'r.')
        ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[0]))
        ax.grid(visible=True)
        ax.set_yticks([])
        ax.set_xlabel(r'$y_{i}$',fontsize=fontsize,labelpad=15)
        ax.set_xlim((np.min(Isomaps),np.max(Isomaps)))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            line.set_xdata(Iso_frame)
            ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(eps_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 2
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(eps_vec)))
        ind = 0
        for eps in eps_vec:
            ISO = data_now.IsoMap(k,eps)
            Isomaps[:,:,ind] = ISO
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Isomaps[:,0,0],Isomaps[:,1,0],'r.')
        ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[0]))
        ax.grid(visible=True)
        #ax.set_yticks([])
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_xlim((np.min(Isomaps[:,0,:]),np.max(Isomaps[:,0,:])))
        ax.set_ylim((np.min(Isomaps[:,1,:]),np.max(Isomaps[:,1,:])))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            line.set_xdata(Iso_frame[:,0])
            line.set_ydata(Iso_frame[:,1])
            ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(eps_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 3
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(eps_vec)))
        ind = 0
        for eps in eps_vec:
            ISO = data_now.IsoMap(k,eps)
            Isomaps[:,:,ind] = ISO
            ind += 1



        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(25,40,0)
        data_3D = ax.scatter3D(Isomaps[:,0,0],Isomaps[:,1,0],Isomaps[:,2,0],'r.')
        ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[0]))
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(Isomaps[:,0,:]),np.max(Isomaps[:,0,:])))
        ax.set_ylim((np.min(Isomaps[:,1,:]),np.max(Isomaps[:,1,:])))
        ax.set_zlim((np.min(Isomaps[:,2,:]),np.max(Isomaps[:,2,:])))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            #ax.set_xdata(Iso_frame[:,0])
            #ax.set_ydata(Iso_frame[:,1])
            #ax.set_zdata(Iso_frame[:,2])
            data_3D._offsets3d = (Iso_frame[:,0], Iso_frame[:,1], Iso_frame[:,2])
            ax.set_title('Isomap for k = ' + str(k) + ', eps = ' + str(eps_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(eps_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')

    #Function to animate the isomap embeddings over a range of
    #different m values used to construct the m-NN connectivity:
    def isomap_animation_kNN():
        fontsize=20 #fontsize for plots

        global filename

        #Desired epsilon values:
        m_vec = np.linspace(1,100,100,dtype=int) #np.linspace(1,100,100,dtype=int)


        #First animate for k = 1:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 1
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            ISO = data_now.IsoMap_kNN(k,m)
            Isomaps[:,:,ind] = ISO
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Isomaps[:,:,0],np.zeros(np.shape(ISO)[0]),'r.')
        ax.set_title(r'Isomap for $k$ = ' + str(k) + r', $\epsilon$ = ' + str(m_vec[0]) + r', $\theta$ = ' + str(theta))
        ax.grid(visible=True)
        ax.set_yticks([])
        ax.set_xlabel(r'$y_{i}$',fontsize=fontsize,labelpad=15)
        ax.set_xlim((np.min(Isomaps),np.max(Isomaps)))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            line.set_xdata(Iso_frame)
            ax.set_title('Isomap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 2
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            ISO = data_now.IsoMap_kNN(k,m)
            Isomaps[:,:,ind] = ISO
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Isomaps[:,0,0],Isomaps[:,1,0],'r.')
        ax.set_title(r'Isomap for $k$ = ' + str(k) + r', $\epsilon$ = ' + str(m_vec[0]) + r', $\theta$ = ' + str(theta))
        ax.grid(visible=True)
        #ax.set_yticks([])
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_xlim((np.min(Isomaps[:,0,:]),np.max(Isomaps[:,0,:])))
        ax.set_ylim((np.min(Isomaps[:,1,:]),np.max(Isomaps[:,1,:])))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            line.set_xdata(Iso_frame[:,0])
            line.set_ydata(Iso_frame[:,1])
            ax.set_title('Isomap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 3
        #IsoMap embeddings:

        Isomaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            ISO = data_now.IsoMap_kNN(k,m)
            Isomaps[:,:,ind] = ISO
            ind += 1



        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(25,40,0)
        data_3D = ax.scatter3D(Isomaps[:,0,0],Isomaps[:,1,0],Isomaps[:,2,0],'r.')
        ax.set_title(r'Isomap for $k$ = ' + str(k) + r', $\epsilon$ = ' + str(m_vec[0]) + r', $\theta$ = ' + str(theta))
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(Isomaps[:,0,:]),np.max(Isomaps[:,0,:])))
        ax.set_ylim((np.min(Isomaps[:,1,:]),np.max(Isomaps[:,1,:])))
        ax.set_zlim((np.min(Isomaps[:,2,:]),np.max(Isomaps[:,2,:])))
        fig.tight_layout(pad=5.0)
        def animate_Isomaps(ind): 
            Iso_frame = Isomaps[:,:,ind]
            #ax.set_xdata(Iso_frame[:,0])
            #ax.set_ydata(Iso_frame[:,1])
            #ax.set_zdata(Iso_frame[:,2])
            data_3D._offsets3d = (Iso_frame[:,0], Iso_frame[:,1], Iso_frame[:,2])
            ax.set_title('Isomap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Isomaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_isomap_theta_' + str(theta) + '.gif')


    #Function to animate the Laplacian eigenamps embeddings over a range of
    #different sigma values for the Gaussian weights:
    def eigenmaps_animation_Gaussian():
        fontsize=20 #fontsize for plots

        global filename

        #Desired epsilon values:
        sigma_vec = np.linspace(0.1,5,100) #np.linspace(1,20,100)



        #First animate for k = 1:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 1
        #Laplacian eigenmap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(sigma_vec)))
        ind = 0
        for sigma in sigma_vec:
            data_now.constructGaussianWeights(sigma=sigma)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Eigenmaps[:,:,0],np.zeros(np.shape(LEM)[0]),'r.')
        ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[0]))
        ax.grid(visible=True)
        ax.set_yticks([])
        ax.set_xlabel(r'$y_{i}$',fontsize=fontsize,labelpad=15)
        ax.set_xlim((np.min(Eigenmaps),np.max(Eigenmaps)))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            line.set_xdata(Eig_frame)
            ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(sigma_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 2
        #IsoMap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(sigma_vec)))
        ind = 0
        for sigma in sigma_vec:
            data_now.constructGaussianWeights(sigma=sigma)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Eigenmaps[:,0,0],Eigenmaps[:,1,0],'r.')
        ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[0]))
        ax.grid(visible=True)
        #ax.set_yticks([])
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_xlim((np.min(Eigenmaps[:,0,:]),np.max(Eigenmaps[:,0,:])))
        ax.set_ylim((np.min(Eigenmaps[:,1,:]),np.max(Eigenmaps[:,1,:])))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            line.set_xdata(Eig_frame[:,0])
            line.set_ydata(Eig_frame[:,1])
            ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(sigma_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 3
        #IsoMap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(sigma_vec)))
        ind = 0
        for sigma in sigma_vec:
            data_now.constructGaussianWeights(sigma=sigma)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1




        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(25,40,0)
        data_3D = ax.scatter3D(Eigenmaps[:,0,0],Eigenmaps[:,1,0],Eigenmaps[:,2,0],'r.')
        ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[0]))
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(Eigenmaps[:,0,:]),np.max(Eigenmaps[:,0,:])))
        ax.set_ylim((np.min(Eigenmaps[:,1,:]),np.max(Eigenmaps[:,1,:])))
        ax.set_zlim((np.min(Eigenmaps[:,2,:]),np.max(Eigenmaps[:,2,:])))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            #ax.set_xdata(Iso_frame[:,0])
            #ax.set_ydata(Iso_frame[:,1])
            #ax.set_zdata(Iso_frame[:,2])
            data_3D._offsets3d = (Eig_frame[:,0], Eig_frame[:,1], Eig_frame[:,2])
            ax.set_title('Eigenmap for k = ' + str(k) + ', sigma = ' + str(sigma_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(sigma_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


    #Function to animate the Laplacian eigenamps embeddings over a range of
    #different m values for the k-NN weights:
    def eigenmaps_animation_kNN():
        fontsize=20 #fontsize for plots

        global filename

        #Desired epsilon values:
        m_vec = np.linspace(1,100,100,dtype=int)



        #First animate for k = 1:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 1
        #Laplacian eigenmap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            data_now.constructNearestNeighborWeights(m=m)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Eigenmaps[:,:,0],np.zeros(np.shape(LEM)[0]),'r.')
        ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[0]))
        ax.grid(visible=True)
        ax.set_yticks([])
        ax.set_xlabel(r'$y_{i}$',fontsize=fontsize,labelpad=15)
        ax.set_xlim((np.min(Eigenmaps),np.max(Eigenmaps)))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            line.set_xdata(Eig_frame)
            ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


        #Now for k = 2:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 2
        #IsoMap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            data_now.constructNearestNeighborWeights(m=m)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1

        #Animate results:
        fig, ax = plt.subplots(figsize=(7,7))
        line, = ax.plot(Eigenmaps[:,0,0],Eigenmaps[:,1,0],'r.')
        ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[0]))
        ax.grid(visible=True)
        #ax.set_yticks([])
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_xlim((np.min(Eigenmaps[:,0,:]),np.max(Eigenmaps[:,0,:])))
        ax.set_ylim((np.min(Eigenmaps[:,1,:]),np.max(Eigenmaps[:,1,:])))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            line.set_xdata(Eig_frame[:,0])
            line.set_ydata(Eig_frame[:,1])
            ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


        #Now for k = 3:

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)
        k = 3
        #IsoMap embeddings:

        Eigenmaps = np.zeros((np.shape(data_now.X)[0],k,len(m_vec)))
        ind = 0
        for m in m_vec:
            data_now.constructNearestNeighborWeights(m=m)
            LEM = data_now.LaplacianEigenmaps(k)
            Eigenmaps[:,:,ind] = LEM
            ind += 1




        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(25,40,0)
        my_cmap = plt.get_cmap('winter')
        data_3D = ax.scatter3D(Eigenmaps[:,0,0],Eigenmaps[:,1,0],Eigenmaps[:,2,0],c=Eigenmaps[:,2,0],cmap=my_cmap)
        ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[0]))
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(Eigenmaps[:,0,:]),np.max(Eigenmaps[:,0,:])))
        ax.set_ylim((np.min(Eigenmaps[:,1,:]),np.max(Eigenmaps[:,1,:])))
        ax.set_zlim((np.min(Eigenmaps[:,2,:]),np.max(Eigenmaps[:,2,:])))
        fig.tight_layout(pad=5.0)
        def animate_Eigenmaps(ind): 
            Eig_frame = Eigenmaps[:,:,ind]
            #ax.set_xdata(Iso_frame[:,0])
            #ax.set_ydata(Iso_frame[:,1])
            #ax.set_zdata(Iso_frame[:,2])
            data_3D._offsets3d = (Eig_frame[:,0], Eig_frame[:,1], Eig_frame[:,2])
            data_3D._facecolor3d = Eig_frame[:,2]
            data_3D._edgecolor3d = Eig_frame[:,2]
            ax.set_title('Eigenmap for k = ' + str(k) + ', m = ' + str(m_vec[ind]))

        #Animate and save phi-field evolution:
        anim = FuncAnimation(fig,animate_Eigenmaps,frames=len(m_vec),interval=100)
        anim.save('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.gif')


    #Function to make big 3D plots of single eigenmap embeddings
    #with k-NN connectivity:
    def plot_single_eigenmaps_3D_kNN(m,sigma=None):
        #Inputs:
        #   m-- number of nearest neighbors in k-NN graph
        #   sigma-- value of sigma, if we desire to weight
        #           the edges in the k-NN graph with Gaussian/
        #           heat kernel weights based off of the
        #           Euclidean distances of adjacent points.
        #           If None, then this weighting is not done,
        #           and all edges are weighted 1.
        fontsize=20 #fontsize for plots

        global filename

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)

        k = 3
        data_now.constructNearestNeighborWeights(m=m,sigma=sigma)
        LEM = data_now.LaplacianEigenmaps(k)

        tri = mtri.Triangulation(LEM[:,0],LEM[:,1])


        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(30,-115 + 90,0) #ax.view_init(25,-10,0) #ax.view_init(25,40,0)
        my_cmap = plt.get_cmap('winter') #plt.get_cmap('inferno')
        #data_3D = ax.scatter3D(LEM[:,0],LEM[:,1],LEM[:,2],c=LEM[:,2],cmap=my_cmap)
        data_3D = ax.plot_trisurf(LEM[:,0], LEM[:,1], LEM[:,2], triangles=tri.triangles, cmap=my_cmap)
        ax.set_title('m-NN Laplacian Eigenmaps embedding for m = ' + str(m) + ', k = ' + str(k) + r', $\theta$ = ' + str(theta)) #inverting k and m just to adhere to more standard notation
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(LEM[:,0]),np.max(LEM[:,0])))
        ax.set_ylim((np.min(LEM[:,1]),np.max(LEM[:,1])))
        ax.set_zlim((np.min(LEM[:,2]),np.max(LEM[:,2])))
        fig.tight_layout(pad=5.0)
        plt.savefig('k='+str(k)+'_eigenmap_theta_' + str(theta) + '.png')
        plt.close()


    #Function to make big 3D plots of single isomap embeddings
    #with k-NN connectivity:
    def plot_single_isomaps_3D_kNN(m,sigma=None):
        #Inputs:
        #   m-- number of nearest neighbors in k-NN graph
        #   sigma-- value of sigma, if we desire to weight
        #           the edges in the k-NN graph with Gaussian/
        #           heat kernel weights based off of the
        #           Euclidean distances of adjacent points.
        #           If None, then this weighting is not done,
        #           and all edges are weighted 1.
        fontsize=20 #fontsize for plots

        global filename

        #Load data:
        X = np.load(filename)
        #Initialize data class with this data:
        data_now = fin.data(X)

        k = 3
        ISO = data_now.IsoMap_kNN(k,m)

        tri = mtri.Triangulation(ISO[:,0],ISO[:,1])


        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.view_init(25,-10,0) #ax.view_init(25,40,0)
        my_cmap = plt.get_cmap('winter') #plt.get_cmap('inferno')
        #data_3D = ax.scatter3D(LEM[:,0],LEM[:,1],LEM[:,2],c=LEM[:,2],cmap=my_cmap)
        data_3D = ax.plot_trisurf(ISO[:,0], ISO[:,1], ISO[:,2], triangles=tri.triangles, cmap=my_cmap)
        ax.set_title('m-NN IsoMap embedding for m = ' + str(m) + ', k = ' + str(k) + r', $\theta$ = ' + str(theta)) #inverting k and m just to adhere to more standard notation
        ax.grid(visible=True)
        ax.set_xlabel(r'$\vec{y}_{i}\cdot\hat{e}_{1}$',fontsize=fontsize,labelpad=15)
        ax.set_ylabel(r'$\vec{y}_{i}\cdot\hat{e}_{2}$',fontsize=fontsize)
        ax.set_zlabel(r'$\vec{y}_{i}\cdot\hat{e}_{3}$',fontsize=fontsize)
        ax.set_xlim((np.min(ISO[:,0]),np.max(ISO[:,0])))
        ax.set_ylim((np.min(ISO[:,1]),np.max(ISO[:,1])))
        ax.set_zlim((np.min(ISO[:,2]),np.max(ISO[:,2])))
        fig.tight_layout(pad=5.0)
        plt.savefig('k='+str(k)+'_isomap_theta_' + str(theta) + '.png')
        plt.close()




    #Run the desired functions:
    
    determine_epsilon(); print('determine_epsilon() complete.\n');
    full_embedded_plots(); print('full_embedded_plots() complete.\n');
    #isomap_animation_single(); print('isomap_animation_single() complete.\n');
    isomap_animation_kNN(); print('isomap_animation_kNN() complete.\n');
    #eigenmaps_animation_kNN(); print('eigenmaps_animation_kNN() complete.\n');
    plot_single_eigenmaps_3D_kNN(10,sigma=None)
    plot_single_isomaps_3D_kNN(10,sigma=None)

#Function to plot 4 different n_c vs. epsilon subplots:
def determine_epsilon_subplots():
    fontsize = 20
    filenames = []
    for theta in [60,75,90,120]:
        filenames.append('theta_' + str(theta) + '_zScore.npy')


    N = 50 #number of distinct epsilons to sample SLC over
    epsilon_vec = np.linspace(0,3,N)




    #Calculate and plot results:
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    ind = 0
    for i in range(0,2):
        for j in range(0,2):
            print(ind)
            filename = filenames[ind]
            X = np.load(filename)
            data_now = fin.data(X)
            n_SLC = np.zeros(N) #pre-allocating number of clusters returned by SLC for each epsilon
            for ind2 in range(0,len(epsilon_vec)):
                (clusters_SLC,cluster_ind_SLC) = data_now.singleLinkageClustering(epsilon_vec[ind2],tol=1e-9)
                n_SLC[ind2] = len(clusters_SLC) #len(np.unique(cluster_ind_SLC))

            #Determine the sizes of the 2 clusters when
            #epsilon = 1.5:
            (clusters_SLC,cluster_ind_SLC) = data_now.singleLinkageClustering(1.5,tol=1e-9)
            n1 = np.shape(clusters_SLC[0])[0]
            n2 = np.shape(clusters_SLC[1])[0]

            print(n1)
            print(n2)
            print('\n')

            ax[i,j].plot(epsilon_vec,n_SLC,'k.-')
            ax[i,j].plot(epsilon_vec[n_SLC==2],n_SLC[n_SLC==2],'r.')
            ax[i,j].set_title('Number of SLC clusters vs. $\epsilon$ for ' + filename,fontsize=10)
            ax[i,j].grid(visible=True)
            ax[i,j].set_xlabel('$\epsilon$',fontsize=20)
            ax[i,j].set_ylabel('$n_{c}$',fontsize=20)
            ind += 1

    plt.grid(visible=True)
    fig.tight_layout(pad=5.0)
    plt.savefig('n_vs_epsilon_subplots.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()

#Function to plot 4 different k-means subplots:
def kMeans_subplots():
    fontsize = 20
    filenames = []
    for theta in [60,75,90,105]:
        filenames.append('theta_' + str(theta) + '_zScore.npy')


    N = 12 #number of distinct epsilons to sample SLC over
    k_vec = np.linspace(1,N,N,dtype=int)




    #Calculate and plot results:
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    ind = 0
    for i in range(0,2):
        for j in range(0,2):
            print(ind)
            filename = filenames[ind]
            X = np.load(filename)
            data_now = fin.data(X)
            N_max = np.zeros(N) #pre-allocating max sizes of clusters
            N_min = np.zeros(N) #pre-allocating min sizes of clusters
            N_mean = np.zeros(N) #pre-allocating mean sizes of clusters
            N_std = np.zeros(N) #pre-allocating std of sizes of clusters
            avg_diam = np.zeros(N) #pre-allocating average diameters of clusters
            std_diam = np.zeros(N) #pre-allocating std of diameters of clusters
            avg_centr = np.zeros(N) #pre-allocating average inter-centroid distances of clusters
            std_centr = np.zeros(N) #pre-allocating std of inter-centroid distances of clusters
            for ind2 in range(0,len(k_vec)):
                (clusters_k,cluster_ind_k,centroids) = data_now.kMeansClustering(k_vec[ind2])
                A = np.bincount(cluster_ind_k)
                N_max[ind2] = np.max(A)
                N_min[ind2] = np.min(A[A>0])
                N_mean[ind2] = np.mean(A[A>0])
                N_std[ind2] = np.std(A[A>0])

                #Calculate the average diameter of each cluster:
                diams = np.zeros(len(clusters_k))
                for ind3 in range(0,len(clusters_k)):
                    if np.size(clusters_k[ind3]) > 0: #if non-empty cluster
                        diam_data = fin.data(clusters_k[ind3])
                        diams[ind3] = np.max(diam_data.dX)
                    else:
                        diams[ind3] = np.nan
                #print('size(diams) = ' + str(np.size(diams)))
                avg_diam[ind2] = np.nanmean(diams)
                std_diam[ind2] = np.nanstd(diams)

                #Also calculate the average inter-centroid distance:
                nonempty_centroids = centroids[np.unique(cluster_ind_k),:] #eliminate all centroids corresponding to empty clusters
                #print('np.shape(nonempty_centroids) = ' + str(np.shape(nonempty_centroids)))
                centr_data = fin.data(nonempty_centroids)
                #Calculate average and std inter-centroid distance:
                U = np.triu(centr_data.dX)
                #print('np.shape(U) = ' + str(np.shape(U)))
                uniqueInterCentroidDistances = U[np.nonzero(U)]
                #print('np.shape(uniqueInterCentroidDistances) = ' + str(np.shape(uniqueInterCentroidDistances)))
                avg_centr[ind2] = np.mean(uniqueInterCentroidDistances)
                std_centr[ind2] = np.std(uniqueInterCentroidDistances)

            ax[i,j].plot(k_vec,N_max,'r.-')
            ax[i,j].plot(k_vec,N_min,'b.-')
            ax[i,j].errorbar(k_vec,N_mean,yerr=N_std,fmt='g.-',ecolor='k')
            ax[i,j].set_title('$k$-means cluster sizes vs. $k$ for ' + filename,fontsize=10)
            ax[i,j].grid(visible=True)
            ax[i,j].set_xlabel('$k$',fontsize=20)
            ax[i,j].set_ylabel('Cluster size statistics',fontsize=20)
            ax[i,j].legend(['Max size','Min size','Mean size'])
            ind += 1


    plt.grid(visible=True)
    fig.tight_layout(pad=5.0)
    plt.savefig('maxmin_vs_k_means.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()



def plot_diam_centroids():
    fontsize = 20
    filenames = []
    for theta in [60,75,90,105]:
        filenames.append('theta_' + str(theta) + '_zScore.npy')


    N = 12 #number of distinct epsilons to sample SLC over
    k_vec = np.linspace(1,N,N,dtype=int)




    #Calculate and plot results:
    fig2, ax2 = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    ind = 0
    for i in range(0,2):
        for j in range(0,2):
            print(ind)
            filename = filenames[ind]
            X = np.load(filename)
            data_now = fin.data(X)
            N_max = np.zeros(N) #pre-allocating max sizes of clusters
            N_min = np.zeros(N) #pre-allocating min sizes of clusters
            N_mean = np.zeros(N) #pre-allocating mean sizes of clusters
            N_std = np.zeros(N) #pre-allocating std of sizes of clusters
            avg_diam = np.zeros(N) #pre-allocating average diameters of clusters
            std_diam = np.zeros(N) #pre-allocating std of diameters of clusters
            avg_centr = np.zeros(N) #pre-allocating average inter-centroid distances of clusters
            std_centr = np.zeros(N) #pre-allocating std of inter-centroid distances of clusters
            for ind2 in range(0,len(k_vec)):
                (clusters_k,cluster_ind_k,centroids) = data_now.kMeansClustering(k_vec[ind2])
                A = np.bincount(cluster_ind_k)
                N_max[ind2] = np.max(A)
                N_min[ind2] = np.min(A[A>0])
                N_mean[ind2] = np.mean(A[A>0])
                N_std[ind2] = np.std(A[A>0])

                #Calculate the average diameter of each cluster:
                diams = np.zeros(len(clusters_k))
                for ind3 in range(0,len(clusters_k)):
                    if np.size(clusters_k[ind3]) > 0: #if non-empty cluster
                        diam_data = fin.data(clusters_k[ind3])
                        diams[ind3] = np.max(diam_data.dX)
                    else:
                        diams[ind3] = np.nan
                #print('size(diams) = ' + str(np.size(diams)))
                avg_diam[ind2] = np.nanmean(diams)
                std_diam[ind2] = np.nanstd(diams)

                #Also calculate the average inter-centroid distance:
                nonempty_centroids = centroids[np.unique(cluster_ind_k),:] #eliminate all centroids corresponding to empty clusters
                #print('np.shape(nonempty_centroids) = ' + str(np.shape(nonempty_centroids)))
                centr_data = fin.data(nonempty_centroids)
                #Calculate average and std inter-centroid distance:
                U = np.triu(centr_data.dX)
                #print('np.shape(U) = ' + str(np.shape(U)))
                uniqueInterCentroidDistances = U[np.nonzero(U)]
                #print('np.shape(uniqueInterCentroidDistances) = ' + str(np.shape(uniqueInterCentroidDistances)))
                avg_centr[ind2] = np.mean(uniqueInterCentroidDistances)
                std_centr[ind2] = np.std(uniqueInterCentroidDistances)

            ax2[i,j].plot(k_vec,avg_diam,'r.-')
            ax2[i,j].plot(k_vec,avg_centr,'b.-')
            ax2[i,j].set_title('Diameters/centroids vs. $k$ for ' + filename,fontsize=10)
            ax2[i,j].grid(visible=True)
            ax2[i,j].set_xlabel('$k$',fontsize=20)
            ax2[i,j].set_ylabel('Clustering statistics',fontsize=20)
            ax2[i,j].legend(['Avg. cluster diameter','Avg. inter-centroid distance'])
            ind += 1
    plt.grid(visible=True)
    fig2.tight_layout(pad=5.0)
    plt.savefig('diam_vs_centroid.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()

#Function to plot 4 different k-medians subplots:
def kMedians_subplots():
    fontsize = 20
    filenames = []
    for theta in [60,75,90,105]:
        filenames.append('theta_' + str(theta) + '_zScore.npy')


    N = 12 #number of distinct epsilons to sample SLC over
    k_vec = np.linspace(1,N,N,dtype=int)




    #Calculate and plot results:
    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    ind = 0
    for i in range(0,2):
        for j in range(0,2):
            print(ind)
            filename = filenames[ind]
            X = np.load(filename)
            data_now = fin.data(X)
            N_max = np.zeros(N) #pre-allocating max sizes of clusters
            N_min = 100*np.ones(N) #pre-allocating min sizes of clusters
            N_mean = 100*np.ones(N) #pre-allocating mean sizes of clusters
            N_std = 100*np.ones(N) #pre-allocating std of sizes of clusters
            for ind2 in range(0,len(k_vec)):
                (clusters_k,cluster_ind_k,centroids) = data_now.kMediansClustering(k_vec[ind2])
                A = np.bincount(cluster_ind_k)
                N_max[ind2] = np.max(A)
                N_min[ind2] = np.min(A[A>0])
                N_mean[ind2] = np.mean(A[A>0])
                N_std[ind2] = np.std(A[A>0])

            N_min[N_min==0] = 1000



            ax[i,j].plot(k_vec,N_max,'r.-')
            ax[i,j].plot(k_vec,N_min,'b.-')
            ax[i,j].errorbar(k_vec,N_mean,yerr=N_std,fmt='g.-',ecolor='k')
            ax[i,j].set_title('$k$-medians cluster sizes vs. $k$ for ' + filename,fontsize=10)
            ax[i,j].grid(visible=True)
            ax[i,j].set_xlabel('$k$',fontsize=20)
            ax[i,j].set_ylabel('Cluster size statistics',fontsize=20)
            ax[i,j].legend(['Max size','Min size','Mean size'])
            ind += 1

    plt.grid(visible=True)
    fig.tight_layout(pad=5.0)
    plt.savefig('maxmin_vs_k_medians.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()


#Function to calculate the Hausdorff distances between the
#datasets for each theta, using the Euclidean metric to compute
#distances between points in a given dataset:
def calculate_dH(fileExtension='zScore.npy',directed=False,ignore120=False):
    #Input:
    #   fileExtension-- file extension to control which datasets are used.
    #   directed-- boolean specifying whether we wish to compute
    #              the DIRECTED Hausdorff distances (directed=True)
    #              or the symmetrized Hausdorff distance which we
    #              learned in class (directed=False).
    #   ignore120-- boolean controlling whether or not we ignore
    #               the theta = 120 dataset.

    fontsize=20

    start_time = time.time()
    X_60 = np.load('theta_60_' + fileExtension)
    X_75 = np.load('theta_75_' + fileExtension)
    X_90 = np.load('theta_90_' + fileExtension)
    X_105 = np.load('theta_105_' + fileExtension)
    if not ignore120:
        X_120 = np.load('theta_120_' + fileExtension)

    #Arrange all data in a list:
    if ignore120:
        X = [X_60,X_75,X_90,X_105]
    else:
        X = [X_60,X_75,X_90,X_105,X_120]

    #Iterate through each pair of spaces and compute the directed
    #Hausdorff distance for each:
    dH = np.zeros((len(X),len(X))) #pre-allocate
    for i in range(0,len(X)):
        for j in range(0,len(X)):
            if not i==j:
                if directed: #compute directed Hausdorff distance
                    dH[i,j] = spd.directed_hausdorff(X[i],X[j])[0] #directed
                else: #compute symmetric Hausdorff distance
                    dH[i,j] = max(spd.directed_hausdorff(X[i],X[j])[0],spd.directed_hausdorff(X[j],X[i])[0]) #symmetric

            else:
                dH[i,j] = np.NaN #set diagonals to NaN just so colorscale in plt.imshow() is not distorted


    #Plot and save results:
    fig, ax = plt.subplots(figsize=(7,7))
    cmap = mpl.colormaps['plasma']
    cmap.set_bad('k')
    im = ax.imshow(dH, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('Hausdorff Distances',fontsize=fontsize)
    ax.set_xlabel(r'$\theta_{2}$',fontsize=fontsize)
    ax.set_ylabel(r'$\theta_{1}$',fontsize=fontsize)
    if ignore120:
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['60','75','90','105'],fontsize=fontsize)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(['60','75','90','105'],fontsize=fontsize)
    else:
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(['60','75','90','105','120'],fontsize=fontsize)
        ax.set_yticks([0,1,2,3,4])
        ax.set_yticklabels(['60','75','90','105','120'],fontsize=fontsize)
    plt.savefig('dH.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()


    end_time = time.time()
    print('calculate_dH() executed in ' + str(end_time-start_time) + ' seconds.\n')
    return(dH)

#Function to calculate the Gromov-Hausdorff distances between the
#datasets for each theta, using the Euclidean metric to compute
#distances between points in a given dataset:
def calculate_dGH(fileExtension='zScore.npy',c=None,ignore120=False,iter_budget=100):
    #Inputs:
    #   fileExtension-- file extension to control which datasets are used.
    #   c-- the relaxation parameter for 
    #   ignore120-- boolean controlling whether or not we ignore
    #               the theta = 120 dataset.

    fontsize=20

    start_time = time.time()
    X_60 = np.load('theta_60_' + fileExtension)
    X_75 = np.load('theta_75_' + fileExtension)
    X_90 = np.load('theta_90_' + fileExtension)
    X_105 = np.load('theta_105_' + fileExtension)
    if not ignore120:
        X_120 = np.load('theta_120_' + fileExtension)


    #Compute Euclidean distances between points within each dataset:
    data_temporary = fin.data(X_60)
    d_60_Euclidean = data_temporary.d_Euclidean()
    data_temporary = fin.data(X_75)
    d_75_Euclidean = data_temporary.d_Euclidean()
    data_temporary = fin.data(X_90)
    d_90_Euclidean = data_temporary.d_Euclidean()
    data_temporary = fin.data(X_105)
    d_105_Euclidean = data_temporary.d_Euclidean()
    if not ignore120:
        data_temporary = fin.data(X_120)
        d_120_Euclidean = data_temporary.d_Euclidean()

    #Construct list of Euclidean distance matrices:
    if ignore120:
        d_Euclidean = [d_60_Euclidean, d_75_Euclidean, d_90_Euclidean, d_105_Euclidean]
    else:
        d_Euclidean = [d_60_Euclidean, d_75_Euclidean, d_90_Euclidean, d_105_Euclidean, d_120_Euclidean]

    #Iterate through each pair of spaces and compute the GH distance for each:
    dGH = np.zeros((len(d_Euclidean),len(d_Euclidean))) #pre-allocate
    for i in range(0,len(d_Euclidean)):
        for j in range(i,len(d_Euclidean)):
            if not i==j:
                if c is None:
                    dGH[i,j] = dgh.upper(d_Euclidean[i],d_Euclidean[j],verbose=1,iter_budget=iter_budget)
                    print('\n')
                else:
                    dGH[i,j] = dgh.upper(d_Euclidean[i],d_Euclidean[j],c=c,verbose=1,iter_budget=iter_budget)
                    print('\n')
            else:
                dGH[i,j] = np.NaN #set diagonals to NaN just so colorscale in plt.imshow() is not distorted

    dGH += dGH.T #symmetrize

    #Plot and save results:
    fig, ax = plt.subplots(figsize=(7,7))
    cmap = mpl.colormaps['plasma']
    cmap.set_bad('k')
    im = ax.imshow(dGH, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title('Gromov-Hausdorff Distances',fontsize=fontsize)
    ax.set_xlabel(r'$\theta_{2}$',fontsize=fontsize)
    ax.set_ylabel(r'$\theta_{1}$',fontsize=fontsize)
    if ignore120:
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(['60','75','90','105'],fontsize=fontsize)
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(['60','75','90','105'],fontsize=fontsize)
    else:
        ax.set_xticks([0,1,2,3,4])
        ax.set_xticklabels(['60','75','90','105','120'],fontsize=fontsize)
        ax.set_yticks([0,1,2,3,4])
        ax.set_yticklabels(['60','75','90','105','120'],fontsize=fontsize)
    plt.savefig('dGH.png',bbox_inches='tight',pad_inches = 0.25)
    plt.close()

    end_time = time.time()
    print('calculate_dGH() executed in ' + str(end_time-start_time) + ' seconds.\n')

    return(dGH)




#Run all relevant functions:
determine_epsilon_subplots()
kMeans_subplots()
kMedians_subplots()
plot_diam_centroids()
calculate_dH()
calculate_dGH(iter_budget=100) #change iter_budget to 1000 for tighter bound-- yields equivalent results, though!