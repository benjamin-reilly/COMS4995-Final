"""bsr2138 -- Final Project"""

#Built off of HW2_classes_and_functions.py.
#The only notable changes that I have made are that I have altered
#the MDS() method to more accurately reflect the algorithm
#I am seeing online (i.e. right-multiply the evecs matrix by
#the sqrt of the evals matrix, rather than returning simply
#the un-multiplied evecs matrix). Similarly, I did not ignore
#the first eigenvalue/eigenvector in the Laplacian Eigenmaps
#embedding, even though it appears as though this is standard
#(again, I found this by poking around online). Similarly
#for Spectral Clustering, which uses the same embedding procedure
#as is done in Laplacian Eigenmaps.

import numpy as np
import networkx as nx
import scipy.spatial.distance as spd

#Define a class for our data:
class data:
    def __init__(data_self, X: np.ndarray, dX=None):
        #Inputs:
        #   X-- n-by-N array, where n is the number of datapoints 
        #       and N is the dimensionality of the data (i.e.
        #       the datapoints x_i are the rows of X).
        #   dX-- n-by-n array where the i,j entry is the
        #        distance between the datapoints x_i and
        #        x_j. If None, then dX is constructed
        #        using the standard N-dimensional Euclidean 
        #        metric.

        #Define the datapoints as an instance attribute:
        data_self.X = X

        #Define dX as Euclidean metric if not given:
        if dX is None:
            #Compact and vectorized way to compute the Euclidean
            #distances between every pair of points, inspired by
            #the following stackoverflow response:
            #https://stackoverflow.com/questions/63673658/compute-distances-between-all-points-in-array-efficiently-using-python
            M = X[np.newaxis,:,:] - X[:,np.newaxis,:]
            data_self.dX = np.sqrt(np.sum(np.square(M),axis=2))
        else:
            #Check to ensure that the given distances are valid for
            #the dataset X (this is mostly for myself, because I
            #imagine I'll mess this up at some point otherwise):
            if not (np.shape(dX) == (np.shape(X)[0],np.shape(X)[0])):
                data_self.dX = dX #define instance attribute as given distances
            else:
                raise Exception('Given distances are incompatible with the number of datapoints provided.')
            
        #Initialize the adjacency matrix, w, as None:
        data_self.w = None

    #Method to explicitly calculate and return the Euclidean distance matrix:
    def d_Euclidean(data_self):
        M = data_self.X[np.newaxis,:,:] - data_self.X[:,np.newaxis,:]
        return(np.sqrt(np.sum(np.square(M),axis=2)))

    #Method to explicitly calculate and return the geodesic distance matrix,
    #after single-linkage clustering is performed (e.g. what is done in
    #the Isomap algorithm):
    def d_geo(data_self,epsilon:float):

        #Just copied-and-pasted from my IsoMap() method, without
        #squaring D at the end (since we are interested here in
        #the actual geodesic distances themselves, rather than the
        #squares of these to be passed into e.g. MDS):
        n = np.shape(data_self.X)[0]
        A = (data_self.d_Euclidean() < epsilon).astype(int)
        G = nx.from_numpy_array(A)
        L = dict(nx.all_pairs_shortest_path_length(G))
        D = np.zeros((n,n))
        for ind1 in range(0,n):
            for ind2 in range(ind1+1,n):
                K = L[ind1].keys()
                if ind2 in K:
                    D[ind1,ind2] = L[ind1][ind2]
        D = D + np.transpose(D)
        return(D)



    #Method to construct the edge weights as a Gaussian
    #of the distances with some desired standard deviation sigma:
    def constructGaussianWeights(data_self,sigma=1):
        #Inputs:
        #   sigma-- standard deviation used in the Gaussian
        #           when calculating edge weights
        data_self.w = np.exp(-np.square(data_self.dX)/(2*(sigma**2)))

        #Remove diagonal elements so that vertices are not connected
        #to themselves:
        data_self.w -= np.diag(np.diag(data_self.w))

    #Method to construct w as the the m-nearest neighbor adjacency
    #matrix:
    def constructNearestNeighborWeights(data_self,m=1,sigma=None):
        #Inputs:
        #   m-- vertices will be adjacent to all
        #       m-nearest neighbors
        #   sigma-- if None, then the m-NN adjacency matrix
        #           is constructed (i.e. all edge weights
        #           are 1). If sigma is not None, then the
        #           value of sigma is used in the heat kernel
        #           to weight the edges based off of the
        #           Euclidean distance between adjacent points.

        #First extract number of datapoints/dimension
        #of the adjacency matrix:
        n = np.shape(data_self.dX)[0]

        #First sort the indices of all datapoints in each row of
        #data_self.dX (i.e. we are ordering the indices based on
        #their distance to the datapoint corresponding to the
        #respective row):
        ind_sorted = np.argsort(data_self.dX,axis=1)
        
        #Now, we wish to construct the adjacency matrix, w, such that
        #all points are only adjacent to their m-nearest neighbors.
        #Since the diagonal elements of dX are all 0, we have to
        #ignore the first column of ind_sorted when extracting
        #which indices we wish to use to construct edges:
        ind_connect = ind_sorted[:,1:(m+1)]

        #Now we wish to simply set w_{i,j} = 1 whenever
        #the index j exists in row i of ind_connect (or,
        #equivalently, whenever the index i exists in column
        #j of ind_connect), and w_{i,j} = 0 otherwise.

        #There's probably a vectorized way to do this without
        #using a for-loop, but for now I'm just going to use
        #a for-loop since it only scales like O(m) and we
        #are probably never going to use huge values of m:
        data_self.w = np.zeros((n,n))
        i = np.array(range(0,n))
        for ind in range(0,m):
            j = ind_connect[i,ind]
            data_self.w[i,j] = 1

        #Now we need to symmetrize the adjacency matrix
        #(since x_{i} being one of the m-nearest neighbors
        #of x_{j} does not imply that x_{j} is one of the
        #m-nearest neighbors of x_{i})

        #First save a copy of the (possibly) non-symmetric connectivity 
        #(to be used e.g. for LLE), such that the ith row encodes the
        #m nearest-neighbors of datapoint x_{i}:
        data_self.w_nonsym = np.copy(data_self.w)
        #Now proceeed with symmetrizing w:
        data_self.w = data_self.w + np.transpose(data_self.w)

        #Now we just wish to reduce all entries with value 2
        #(which may exist in the event that two points ARE in fact
        #mutually among the m-nearest neighbors of one another),
        #since we want our m-nearest neighbor adjacency matrix
        #to simply have 1's where vertices are adjacent and 0's
        #otherwise:
        data_self.w[data_self.w>1] = 1

        #Replace this with the Gaussian weights if sigma is given:
        if not (sigma is None):
            I = (data_self.w==1) #indicator array
            data_self.w[I] = np.exp(-np.square(data_self.dX[I])/(2*(sigma**2)))


    #Method to return clusters as generated via single-linkage
    #clustering:
    def singleLinkageClustering(data_self,epsilon: float,tol=1e-12):
        #Inputs:
        #   epsilon-- feature scale of the clustering
        #             (an edge is added between two
        #              datapoints iff the distance
        #              between them is < epsilon)
        #   tol-- tolerance within which we consider
        #         eigenvalues to be 0.
        #Output:
        #   clusters-- a list of numpy arrays, in
        #              which each array contains
        #              all of the datapoints in
        #              a certain cluster.
        #   cluster_ind-- n-dimensional array containing
        #                 the cluster assignments
        #                 of each datapoint.

        #Construct adjacency matrix (1's where an edge
        #exists between two datapoints/vertices, 0's
        #otherwise):
        A = (data_self.dX < epsilon).astype(int)

        #Remove all edges connecting vertices to themselves:
        A -= np.eye(np.shape(data_self.dX)[0]).astype(int)

        D = np.diag(np.sum(A,axis=1)) #D matrix

        #Construct graph Laplacian
        L = D - A

        #Solve for eigenvalues and eigenvectors (using eigh
        #since we have a symmetric matrix by construction):
        (evals,evecs) = np.linalg.eig(L)
        #Actually, eigh() was giving some strange results,
        #but when I changed it to use eig() instead, everything
        #looks better.

        #NOTE: It appears as though by default, eigh() returns a set of
        #orthonormal eigenvectors, which is good because (I believe)
        #that is precisely the condition required for our zero
        #eigenvectors to be indicator arrays which have 0's whenever
        #a vertex is not in the cluster. However, I am not sure if
        #eigh() is GUARANTEED to return an orthonormal basis, or
        #if I'm just getting lucky. So, I may have to revisit this
        #and make it more robust in the future.

        #Zero eigenvectors give us the connected components
        #in our graph, i.e. the clusters:
        zero_evecs = evecs[:,np.abs(evals)<tol].astype(bool).astype(int)
        #Casting as bool and then casting as int so that all 0's
        #stay 0 and all of the identical nonzero values become 1.


        #Now use these zero eigenvectors to partition our datapoints
        #into clusters, and also to construct the cluster_ind output:
        clusters = [] #pre-allocating empty list
        cluster_ind = np.zeros(np.shape(data_self.X)[0])
        for ind in range(0,np.shape(zero_evecs)[1]):
            isInCurrentCluster = np.abs(zero_evecs[:,ind])>0
            cluster_ind[isInCurrentCluster] = ind
            clusters.append(data_self.X[isInCurrentCluster,:])

        return(clusters,cluster_ind)
    
    #k-means clustering method:
    def kMeansClustering(data_self, k: int, centroids=None, iter=None):
        #Inputs:
        #   k-- the number of desired clusters
        #   centroids-- initial centroid guesses,
        #               given as a k-by-N array
        #               in which the rows are the
        #               initial centroids.
        #               If None, then the initial
        #               guesses are chosen as a
        #               random set of existing
        #               datapoints which are
        #               mutually far apart.
        #   iter-- max number of iterations in the
        #          k-means clustering algorithm.
        #          If None, then the algorithm
        #          proceeds until the new result
        #          is identical to that of the 
        #          previous iteration.
        #Output:
        #   clusters-- a list of numpy arrays, in
        #              which each array contains
        #              all of the datapoints in
        #              a certain cluster.
        #   cluster_ind-- n-dimensional array containing
        #                 the cluster assignments
        #                 of each datapoint.

        n = np.shape(data_self.X)[0] #number of datapoints
        N = np.shape(data_self.X)[1] #dimensionality of data

        if centroids is None:
            #If no initial centroid guesses are given, 
            #let's start by making some "intelligent"
            #guesses at the centroids, by picking the
            #centroids as datapoints that are far apart
            #from one another:

            #Pre-allocate row-stacked array of centroids:
            centroids = np.zeros((k,N))

            #Also pre-allocate array of indices of the
            #datapoints that have been seleected as centroids:
            c_ind = np.zeros(k,dtype=int)

            #Pick a random datapoint as the first centroid:
            c_ind[0] = np.random.randint(0,n)
            centroids[0,:] = data_self.X[c_ind[0],:]

            #Now, for the remaining centroids, successively pick
            #the datapoint which has the largest average distance
            #from all previous centroids:
            for ind in range(1,k):
                #Pre-allocate array of sum of distances to all 
                #previously-chosen centroids:
                distancesToCentroids = np.zeros(n)
                for ind2 in range(0,ind): #loop through all previously-chosen centroids
                    #Add distance to centroid indexed by ind2 
                    #(which is the datapoint indexed by c_ind[ind2]):
                    distancesToCentroids += data_self.dX[c_ind[ind2],:]

                #The next centroid is chosen as the datapoint with the
                #maximum (average) distance to all pre-existing centroids:
                c_ind[ind] = np.argmax(distancesToCentroids)
                centroids[ind,:] = data_self.X[c_ind[ind],:]

        #Okay, now that we have our initial centroids, let's
        #proceed with the actual k-means clustering:

        cluster_ind_old = -1*np.ones(n,dtype=int) #initialize bogus old value
        cluster_ind = -2*np.ones(n,dtype=int) #initialize different bogus value
        if iter is None:
            #if no iteration limit is given, simply iterate until we 
            #reach a stable solution:
            while True:
                #Calculate the distances between all datapoints in X
                #and all centroids (here I am going to reuse the
                #name distancesToCentroids, even though this is
                #a slightly different object than it was before):
                M = data_self.X[np.newaxis,:,:] - centroids[:,np.newaxis,:]
                distancesToCentroids = np.sqrt(np.sum(np.square(M),axis=2))

                #Note that distancesToCentroids is now a k-by-n array such
                #that the [i,j] entry is the distance between centroid i
                #and datapoint j.

                #So, to assign each datapoint to a cluster, we can
                #simply take the argmin along axis=0:
                cluster_ind = np.argmin(distancesToCentroids,axis=0)

                #Break if we have reached a stable solution:
                if (cluster_ind_old == cluster_ind).all():
                    break

                #Update cluster_ind_old for comparison with the next
                #iteration of the loop:
                cluster_ind_old = np.copy(cluster_ind)

                #Now we need to update the centroids as the mean of each
                #cluster:
                for ind in range(0,k):
                    inCluster = (cluster_ind == ind) #boolean array denoting which points are in the current cluster
                    if inCluster.any(): #if cluster is non-empty
                        centroids[ind,:] = np.mean(data_self.X[inCluster,:],axis=0)
                        #If cluster is empty, don't update centroid position.

        else: #else only carry out a maximum of iter iterations:
            for dummy_ind in range(0,iter):
                M = data_self.X[np.newaxis,:,:] - centroids[:,np.newaxis,:]
                distancesToCentroids = np.sqrt(np.sum(np.square(M),axis=2))
                cluster_ind = np.argmin(distancesToCentroids,axis=1)
                if (cluster_ind_old == cluster_ind).all():
                    break
                cluster_ind_old = np.copy(cluster_ind)
                for ind in range(0,k):
                    inCluster = (cluster_ind == ind)
                    if inCluster.any():
                        centroids[ind,:] = np.mean(data_self.X[inCluster,:],axis=0)

        #Now, our clusters are given by the partite sets of our
        #data which all have the same cluster_ind values:
        clusters = [] #pre-allocating empty list
        for ind in range(0,k):
            clusters.append(data_self.X[cluster_ind==ind,:])

        return(clusters,cluster_ind,centroids)
    
    #k-medians clustering method:
    #(note that this is practically identical to the k-means
    #clustering method, but where the new centroids are
    #updated as the median of all data in the cluster instead
    #of the mean. So, I have removed all comments, since the
    #only change is the np.mean() is replaced with np.median())
    def kMediansClustering(data_self, k: int, centroids=None, iter=None):
        #Inputs:
        #   k-- the number of desired clusters
        #   centroids-- initial centroid guesses,
        #               given as a k-by-N array
        #               in which the rows are the
        #               initial centroids.
        #               If None, then the initial
        #               guesses are chosen as a
        #               random set of existing
        #               datapoints which are
        #               mutually far apart.
        #   iter-- max number of iterations in the
        #          k-means clustering algorithm.
        #          If None, then the algorithm
        #          proceeds until the new result
        #          is identical to that of the 
        #          previous iteration.
        #Output:
        #   clusters-- a list of numpy arrays, in
        #              which each array contains
        #              all of the datapoints in
        #              a certain cluster.
        #   cluster_ind-- n-dimensional array containing
        #                 the cluster assignments
        #                 of each datapoint.
        n = np.shape(data_self.X)[0]
        N = np.shape(data_self.X)[1]
        if centroids is None:
            centroids = np.zeros((k,N))
            c_ind = np.zeros(k,dtype=int)
            c_ind[0] = np.random.randint(0,n)
            centroids[0,:] = data_self.X[c_ind[0],:]
            for ind in range(1,k):
                distancesToCentroids = np.zeros(n)
                for ind2 in range(0,ind):
                    distancesToCentroids += data_self.dX[c_ind[ind2],:]
                c_ind[ind] = np.argmax(distancesToCentroids)
                centroids[ind,:] = data_self.X[c_ind[ind],:]
        cluster_ind_old = -1*np.ones(n,dtype=int)
        cluster_ind = -2*np.ones(n,dtype=int)
        if iter is None:
            while True:
                M = data_self.X[np.newaxis,:,:] - centroids[:,np.newaxis,:]
                distancesToCentroids = np.sqrt(np.sum(np.square(M),axis=2))
                cluster_ind = np.argmin(distancesToCentroids,axis=0)
                if (cluster_ind_old == cluster_ind).all():
                    break
                cluster_ind_old = np.copy(cluster_ind)
                for ind in range(0,k):
                    inCluster = (cluster_ind == ind)
                    if inCluster.any():
                        centroids[ind,:] = np.median(data_self.X[inCluster,:],axis=0)
        else:
            for dummy_ind in range(0,iter):
                M = data_self.X[np.newaxis,:,:] - centroids[:,np.newaxis,:]
                distancesToCentroids = np.sqrt(np.sum(np.square(M),axis=2))
                cluster_ind = np.argmin(distancesToCentroids,axis=1)
                if (cluster_ind_old == cluster_ind).all():
                    break
                cluster_ind_old = np.copy(cluster_ind)
                for ind in range(0,k):
                    inCluster = (cluster_ind == ind)
                    if inCluster.any():
                        centroids[ind,:] = np.median(data_self.X[inCluster,:],axis=0)
        clusters = []
        for ind in range(0,k):
            clusters.append(data_self.X[cluster_ind==ind,:])
        return(clusters,cluster_ind,centroids)

    #Spectral clustering method:
    def spectralClustering(data_self, M: int, k: int, iter=None):
        #Inputs:
        #   M-- the dimensionality of the subspace onto which we
        #       wish to project our data
        #   k-- the desired number of clusters (used as the k
        #       input in the k-means clustering algorithm)
        #   iter-- used as the iter input in the kMeansClustering()
        #          method once we cluster the embedded data using
        #          k-means clustering.

        N = np.shape(data_self.X)[1] #dimensionality of our data

        #If we have not yet constructed an adjacency matrix
        #(data_self.w), then throw an error:
        if data_self.w is None:
            raise Exception('Before calling the spectralClustering() method, you must first construct an adjacency matrix, w, by calling a constructor method (e.g. constructGaussianWeights() or constructNearestNeighborWeights()).')

        #Construct graph Laplacian, L:
        D = np.diag(np.sum(data_self.w,axis=1))
        L = D - data_self.w

        #Compute first M eigenvectors of L (ignoring the first
        #eigenvector):
        (evals,evecs) = np.linalg.eigh(L)
        evecs = evecs[:,1:(M+1)]

        #Now, the projections of our n N-dimensional datapoints onto 
        #M-dimensional space are exactly given by the rows of our n-by-M
        #evecs array. So, now we just need to perform k-means clustering
        #on the embedded data, and then our spectral clustering is
        #complete:
        embedded_data = data(evecs)
        (clusters,cluster_ind) = embedded_data.kMeansClustering(k,iter=iter)

        #Now define the actual clusters of the non-embedded data:
        clusters = []
        for ind in range(0,k):
            clusters.append(data_self.X[cluster_ind==ind,:])
        return(clusters,cluster_ind)
    
    #Laplacian eigenmaps embedding (just the embedding step
    #from spectral clustering):
    def LaplacianEigenmaps(data_self, k: int):
        #Inputs:
        #   k-- the dimensionality of the subspace onto which we
        #       wish to project our data
        #Output:
        #   Returns an n-by-k array in which the n rows are the
        #   n datapoints embedded in k-dimensional space.

        N = np.shape(data_self.X)[1] #dimensionality of our data

        #If we have not yet constructed an adjacency matrix
        #(data_self.w), then throw an error:
        if data_self.w is None:
            raise Exception('Before calling the LaplacianEigenmaps() method, you must first construct an adjacency matrix, w, by calling a constructor method (e.g. constructGaussianWeights() or constructNearestNeighborWeights()).')

        #Construct graph Laplacian, L:
        D = np.diag(np.sum(data_self.w,axis=1))
        L = D - data_self.w

        #Compute first k eigenvectors of L (IGNORING the first one!):
        (evals,evecs) = np.linalg.eigh(L)
        evecs = evecs[:,1:(k+1)]

        #Now, the projections of our n N-dimensional datapoints onto 
        #k-dimensional space are exactly given by the rows of our n-by-k
        #evecs array:
        return(evecs)
    
    #Locally Linear Embedding (LLE) method:
    def LLE(data_self, k: int, tol=1e-10):
        #Inputs:
        #   k-- the dimensionality of the subspace onto which we
        #       wish to project/embed our data, which is also
        #       the number of nearest-neighbors to use when
        #       computing the nearest-neighbor connectivity
        #       of our data.
        #   tol-- tolerance below which eigenvalues are
        #         considered 0.

        #Output:
        #   Returns an n-by-k array in which the n rows are the
        #   n datapoints embedded in k-dimensional space.

        #Pre-allocate the n-by-n matrix of weight coefficients, W:
        n = np.shape(data_self.X)[0]
        W = np.zeros((n,n),dtype=float)

        #Compute indices of all m nearest-neighbors to each
        #datapoint:
        data_self.constructNearestNeighborWeights(m=k)

        #The array data_self.w_nonsym now encodes in its
        #ith row the indices of the m nearest-neighbors
        #of datapoint x_{i}. In particular, the i,j element
        #of data_self.w_nonsym is a 1 if x_{j} is one of the
        #m nearest-neighbors of x_{i}, and 0 otherwise.

        #Now loop through every datapoint and fill out W
        #with the desired values (see HW2 Q1(a)):
        ones = np.ones(k)
        for ind in range(0,n):
            ind_nearest = data_self.w_nonsym[ind,:].astype(bool)
            x_i = data_self.X[ind,:]
            X_i = np.transpose(data_self.X[ind_nearest,:])
            G_half = np.outer(x_i,ones) - X_i
            G = np.matmul(np.transpose(G_half),G_half)
            if np.linalg.det(G) == 0: #if G is singular
                #Perturb the diagonals of G by some small amount, where
                #"small" here is taken relative to the minimum distance
                #between all datapoints in data_self.X:
                G += 1e-9*np.min(np.abs(data_self.dX[data_self.dX>0]))*np.eye(k)
            G_inv = np.linalg.inv(G)
            w_i = np.matmul(G_inv,ones)/(np.matmul(np.transpose(ones),np.matmul(G_inv,ones)))
            #Now insert these values into W:
            W[ind,ind_nearest] = w_i

        #Now simply determine the n eigenvectors of L^{T}L
        #in order to obtain our n embedded datapoints:
        L = np.eye(n) - W
        (evals,evecs) = np.linalg.eigh(np.matmul(np.transpose(L),L))
        #Take the first k eigenvectors, ignoring the first one, and
        #this gives us an array in which our n embedded datapoints
        #are the rows:
        #Discard all eigenvectors with eigenvalues below a certain
        #tolerance:
        #print(evals)
        zeroEvals = (evals < tol)
        evecs = evecs[:,np.logical_not(zeroEvals)]


        #Now construct our embedded data off of the first
        #k non-zero eigenvectors:
        return(evecs[:,0:k])
    
    #Multidimensional scaling (MDS) method:
    def MDS(data_self,k: int, D=None):
        #Inputs:
        #   k-- the dimensionality of the subspace onto which we
        #       wish to project/embed our data
        #   D-- the n-by-n (n being the number of datapoints)
        #       array encoding the SQUARED distances between points.
        #       #So, for "classical"/"metric" MDS we want D to
        #       be an array containing the squares of the
        #       Euclidean distances between points in the
        #       original high-dimensional space.
        #       If None, then D = np.square(data_self.dX) is 
        #       used by default.
        #Output:
        #   Returns an n-by-k array in which the n rows are the
        #   n datapoints embedded in k-dimensional space.

        n = np.shape(data_self.X)[0]

        if D is None:
            D = np.square(data_self.dX)

        #Construct the centering matrix:
        A = np.eye(n) - (1/n)*np.ones((n,n))

        #Construct the matrix from which we will compute
        #the eigenvectors in order to embed our data:
        M = (-1/2)*np.matmul(np.matmul(A,D),A)

        #Now compute evals and evecs:
        (evals,evecs) = np.linalg.eigh(M)

        #Now take the first k eigenvectors and arrange them as
        #the columns of an array. The n rows of this array gives us
        #our embedded datapoints:
        #return(evecs[:,0:k])

        #Actually, according to multiple online sources, what we
        #want is to actually take the rows of a related, but different,
        #matrix: we instead multiply evecs by np.sqrt(np.diag(evals))
        #(which is equivalent to the sqrt of the evals matrix, since it
        #is diagonal) and then take the rows of THAT matrix:
        #Since np.linalg.eigh() sorts things in ASCENDING order
        #of eigenvalue, we want to take the LAST k columns of this
        #matrix:
        out = np.matmul(evecs[:,-k:],np.sqrt(np.diag(evals[-k:])))
        return(out)


    
    #Isomap method:
    def IsoMap(data_self,k: int,epsilon: float):
        #Inputs:
        #   epsilon-- feature scale of the single-linkage
        #             adjacency matrix
        #   k-- the dimensionality of the subspace onto which we
        #       wish to project/embed our data

        n = np.shape(data_self.X)[0]

        #Construct single-linkage adjacency matrix:
        A = (data_self.dX < epsilon).astype(int)

        #Now construct a graph object in NetworkX
        #with connectivity specified by A:
        G = nx.from_numpy_array(A)
        L = dict(nx.all_pairs_shortest_path_length(G))

        #L is now an iterator (first index) with dictionaries keyed by 
        #the target node (second index). So, the shortest path between
        #vertices i and j is given by L[i][j].

        #Unfortunately, as far as I can tell, there is no quick
        #and easy way to convert this to a numpy array, which we
        #want to do because we want to use these path lengths
        #as the input distances in our MDS algorithm. So, after
        #a bit of Googling, it looks like we are stuck looping
        #through the dictionary manually:
        D = np.zeros((n,n)) #graph metric is zero unless we find a path
        for ind1 in range(0,n):
            for ind2 in range(ind1+1,n):
                K = L[ind1].keys()
                if ind2 in K:
                    D[ind1,ind2] = L[ind1][ind2]

        #Now symmetrize D, which at the moment is lower triangular:
        D = D + np.transpose(D)

        #Now square D elementwise, so that we can pass it into our
        #MDS algorithm:
        D = np.square(D)

        #Now simply return the data embedded by the MDS algorithm
        #using these squared graph metric distances:
        out = data_self.MDS(k=k,D=D)
        return(out)


    #Isomap method which uses k-NN connectivity, rather
    #than single-linkage connectivity:
    def IsoMap_kNN(data_self,k: int,m: int):
        #Inputs:
        #   m-- number of nearest neighbors to connect to
        #       each point when constructing the connectivity
        #       of the graph.
        #   k-- the dimensionality of the subspace onto which we
        #       wish to project/embed our data

        n = np.shape(data_self.X)[0]

        #Construct single-linkage adjacency matrix:
        data_self.constructNearestNeighborWeights(m=m)
        A = data_self.w

        #Now construct a graph object in NetworkX
        #with connectivity specified by A:
        G = nx.from_numpy_array(A)
        L = dict(nx.all_pairs_shortest_path_length(G))

        #L is now an iterator (first index) with dictionaries keyed by 
        #the target node (second index). So, the shortest path between
        #vertices i and j is given by L[i][j].

        #Unfortunately, as far as I can tell, there is no quick
        #and easy way to convert this to a numpy array, which we
        #want to do because we want to use these path lengths
        #as the input distances in our MDS algorithm. So, after
        #a bit of Googling, it looks like we are stuck looping
        #through the dictionary manually:
        D = np.zeros((n,n)) #graph metric is zero unless we find a path
        for ind1 in range(0,n):
            for ind2 in range(ind1+1,n):
                K = L[ind1].keys()
                if ind2 in K:
                    D[ind1,ind2] = L[ind1][ind2]

        #Now symmetrize D, which at the moment is lower triangular:
        D = D + np.transpose(D)

        #Now square D elementwise, so that we can pass it into our
        #MDS algorithm:
        D = np.square(D)

        #Now simply return the data embedded by the MDS algorithm
        #using these squared graph metric distances:
        out = data_self.MDS(k=k,D=D)
        return(out)







        




            



#Function which samples the desired number of datapoints
#from a probability distribution which is a mixture of
#spherical Gaussians in Euclidean space:
def sampleGaussians(N,M,mu,sigma,n):
    #Inputs:
    #   N-- dimensionality of datapoints to be generated
    #   M-- the number of Gaussians contributing to the mixed 
    #       distribution
    #   mu-- M-by-N array where the columns are the mean vectors
    #        for each of the M Gaussians in the distribution
    #   sigma-- M-dimensional numpy array of the standard deviation
    #           of each Gaussian (here we assume spherical Gaussians
    #           and so our covariance matrices are just the identity
    #           matrix multiplied by the respective sigmas)
    #   n-- number of datapoints to generate from the distribution

    #NOTE: Sampling from a mixed distribution is equivalent to
    #first sampling from the mixing weights, and then sampling
    #from the distribution corresponding to the sampled weight.

    #Assume equal mixing weights:
    m = np.random.randint(0,high=M,size=n) 
    #The entries in m are the indices of the sampled Gaussians.

    #Pre-allocate array of sampled data:
    samples = np.zeros((1,N))

    #Now actually sample datapoints from these Gaussians:
    for ind in range(0,M): #ind indexes which Gaussian we are currently sampling from
        n_m = len(m[m==ind]) #number of points to sample from current Gaussian
        samples_now = np.random.multivariate_normal(mu[ind,:],(sigma[ind]**2)*np.eye(N),size=n_m)
        samples = np.append(samples,samples_now,axis=0)
    
    #Delete first row of samples, which was only present to initialize
    #the samples array so we could append our datapoints to it:
    samples = np.delete(samples,(0),axis=0)

    #Return samples:
    return(samples)

#Function which samples from an annulus in R^2 centered at the origin:
def sampleAnnulus(r1,r2,n):
    #Inputs:
    #   r1-- the inner radius of the annulus
    #   r2-- the outer radius of the annulus
    #   n-- number of datapoints to generate from the distribution

    #Uniformly sample n random angles in the interval [0,2pi):
    angles = 2*np.pi*np.random.rand(n)

    #Uniformly sample n random lengths in the interval [r1,r2):
    lengths = r1 + (r2-r1)*np.random.rand(n)

    #Now convert all data from polar form to cartesian coordinates:
    x = lengths*np.cos(angles)
    y = lengths*np.sin(angles)
    samples = np.transpose(np.array([x,y]))
    return(samples)



#Numerical experiment illustrating that the dH and dGH values I am
#seeing are likely indicative of actual structural differences in
#the datasets, rather than simply being a vestige of random
#sampling. The random datasets generated in this experiment
#have every coordinate sampled from the standard normal distribution,
#since the actual data I am using in my project has every feature
#z-scored.
def numerical_experiment(M,N,D,dist='normal'):
    #Inputs:
    #   M-- the number of randomly-generated datasets between which
    #       we will calculate the dH distances.
    #   N-- the number of datapoints in each dataset.
    #   D-- the dimensionality of the data in each dataset.
    #   dist-- A string indicating which type of distribution
    #          the datapoints in each set should be sampled
    #          from. 'normal' uses the standard normal distribution,
    #          whereas 'uniform' uses a z-scored uniform distribution.

    data = [] #pre-allocating an empty list

    #Now generate all datasets and place them into the data list:
    if dist == 'normal':
        for ind in range(0,M):
            #Normal distribution:
            data_now = np.random.normal(size=(N,D))

            #Append data_now to the data list:
            data.append(data_now)
    elif dist == 'uniform':
        for ind in range(0,M):
            #Uniform distribution:
            data_now = np.random.rand(N,D)
            #z-score our uniformly-distributed data:
            data_now = (data_now - 0.5)*np.sqrt(12)

            #Append data_now to the data list:
            data.append(data_now)
    else:
        raise Exception("Invalid 'dist' input passed into numerical_experiment().")


    #Now calculate the Hausdorff distance between each dataset:
    dH = np.zeros((M,M)) #pre-allocate
    for i in range(0,M):
        for j in range(0,M):
            if not i==j:
                #compute symmetric Hausdorff distance:
                dH[i,j] = max(spd.directed_hausdorff(data[i],data[j])[0],spd.directed_hausdorff(data[j],data[i])[0]) #symmetric

            else:
                dH[i,j] = np.NaN #set diagonals to NaN just so colorscale in plt.imshow() is not distorted

    #Extract the upper triangle of the dH array:
    T = np.triu(dH)
    
    #Return the mean and std of all inter-dataset Hausdorff distances:
    return(np.nanmean(T[np.nonzero(T)]),np.nanstd(T[np.nonzero(T)]))
