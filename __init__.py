from __future__ import print_function
import numpy as np,  matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn import cluster

class CPCA(object):
    # Returns data    
    def get_data(self):
        return self.data    
    # Returns the background data 
    def get_bg(self):
        return self.bg    
    # Returns the active data
    def get_fg(self):
        return self.fg
    # Returns the active labels
    def get_active_labels(self):
        return self.active_labels    
    # Returns the pca directions for the active set
    def get_pca_directions(self):
        return self.pca_directions
    # Returns the active set projected in the pca directions
    def get_active_pca_projected(self):
        return self.active_pca
    def get_affinity_matrix(self):
        return self.affinity_matrix
    
    # A helper method to standardize arrays
    def standardize_array(self, array):
        standardized_array =  (array-np.mean(array,axis=0)) / np.std(array,axis=0)
        return np.nan_to_num(standardized_array)
    
    
    def __init__(self, n_components=2, standardize=True):
        self.standardize = standardize
        self.n_components = n_components
        
    def fit_transform(self, foreground, background, plot=False, gui=False, verbose=False, auto=True,  n_alphas=40,  max_log_alpha=3, n_alphas_to_return=4, active_labels = None):
        self.fit(foreground, background, verbose=verbose, active_labels=active_labels)
        return self.transform(dataset=foreground, auto=auto,  n_alphas=n_alphas, max_log_alpha=max_log_alpha, n_alphas_to_return=n_alphas_to_return, plot=plot, gui=gui)
    
    def fit(self, foreground, background, active_labels = None, verbose=False):
        # Housekeeping
        self.pca_directions = None
        self.bg_eig_vals = None
        self.affinity_matrix = None
        self.colors = ['k','r','b','g','c']
        
        # Datasets and dataset sizes
        self.fg = foreground
        self.bg = background
        self.n_fg, _           = foreground.shape
        self.n_bg, self.features_d = background.shape
        if active_labels is None:
            active_labels = np.ones(self.n_fg)
        self.active_labels = active_labels

        if (verbose):
            print("Data loaded")
        
        #Center the background data
        self.bg = self.bg - np.mean(self.bg, axis=0)
        if self.standardize: #Standardize if specified
            self.bg = self.standardize_array(self.bg)

        #Calculate the covariance matrix
        self.bg_cov = self.bg.T.dot(self.bg)/(self.bg.shape[0]-1)
        
        #Center the foreground data
        self.fg = self.fg - np.mean(self.fg, axis=0)
        if self.standardize: #Standardize if specified
            self.fg = self.standardize_array(self.fg)

        #Calculate the covariance matrix
        self.fg_cov = self.fg.T.dot(self.fg)/(self.n_fg-1)
        if (verbose):
            print("Covariance matrices computed")
    
    def transform(self, dataset, auto=True, n_alphas=40, max_log_alpha=3, n_alphas_to_return=4, plot=False, gui=False):
        if gui:
            transformed_data_auto, alphas_auto = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
            transformed_data_manual, alphas_manual = self.manual_cpca(dataset, n_alphas, max_log_alpha)
            try:
                from ipywidgets import widgets, interact, Layout
                from IPython.display import display
            except:
                print("To use the GUI, you must be running this code in a jupyter notebook that supports ipywidgets")
            from matplotlib.gridspec import GridSpec
            
            """
            Handles the plotting
            """
            def graph_foreground(ax,fg, active_labels, alpha):
                for i, l in enumerate(np.sort(np.unique(active_labels))):
                    ax.scatter(fg[np.where(active_labels==l),0],fg[np.where(active_labels==l),1], color=self.colors[i], alpha=0.6)    
                if (alpha==0):
                    ax.annotate(r'$\alpha$='+str(np.round(alpha,2))+" (PCA)", (0.05,0.05), xycoords='axes fraction')
                else:
                    ax.annotate(r'$\alpha$='+str(np.round(alpha,2)), (0.05,0.05), xycoords='axes fraction')        
            
            
            """
            This code gets run whenever the widget slider is moved
            """
            def update(value):
                fig = plt.figure(figsize=[10,4])
                gs=GridSpec(2,4)

                for i in range(4):
                    ax1=fig.add_subplot(gs[int(i//2),i%2]) # First row, first column
                    fg = transformed_data_auto[i]
                    graph_foreground(ax1, fg, self.active_labels, alphas_auto[i])

                    ax5=fig.add_subplot(gs[:,2:]) # Second row, span all columns 

                    alpha_idx = np.abs(alphas_manual-10**value).argmin()
                    fg = transformed_data_manual[alpha_idx]
                    graph_foreground(ax5, fg, self.active_labels, alphas_manual[alpha_idx])
                    
                #if len(np.unique(self.active_labels))>1:
                    #plt.legend()
                
                plt.tight_layout()            
                plt.show()
            
            widg = interact(update, value=widgets.FloatSlider(description=r'\(\log_{10}{\alpha} \)', min=-1, max=3, step=4/40, continuous_update=False, layout=Layout(width='80%')))
            
            return
        
        elif plot:
            transformed_data, best_alphas = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
            plt.figure(figsize=[14,3])
            for j, fg in enumerate(transformed_data):
                plt.subplot(1,4,j+1)
                for i, l in enumerate(np.sort(np.unique(self.active_labels))):
                    idx = np.where(self.active_labels==l)
                    plt.scatter(fg[idx,0],fg[idx,1], color=self.colors[i], alpha=0.6, label='Class '+str(i))
                plt.title('Alpha='+str(np.round(best_alphas[j],2)))
            if len(np.unique(self.active_labels))>1:
                plt.legend()
            plt.show()

        else:
            if auto:
                transformed_data, best_alphas = self.automated_cpca(dataset, n_alphas_to_return, n_alphas, max_log_alpha)
            else:
                transformed_data, all_alphas = self.manual_cpca(dataset, n_alphas, max_log_alpha)
            return transformed_data, best_alphas
    
    """
    This function performs contrastive PCA using the alpha technique on the 
    active and background dataset. It automatically determines n_alphas=4 important values
    of alpha up to based to the power of 10^(max_log_alpha=5) on spectral clustering
    of the top subspaces identified by cPCA.
    The final return value is the data projected into the top (n_components = 2) 
    subspaces, which can be plotted outside of this function
    """
    def automated_cpca(self, dataset, n_alphas_to_return, n_alphas, max_log_alpha):
        best_alphas, all_alphas, _, _ = self.find_spectral_alphas(n_alphas, max_log_alpha, n_alphas_to_return)
        best_alphas = np.concatenate(([0], best_alphas)) #one of the alphas is always alpha=0
        data_to_plot = []
        for alpha in best_alphas:
            transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, best_alphas        
        
    """
    This function performs contrastive PCA using the alpha technique on the 
    active and background dataset. It returns the cPCA-reduced data for all values of alpha specified,
    both the active and background, as well as the list of alphas
    """
    def manual_cpca(self, dataset, n_alphas, max_log_alpha):
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        data_to_plot = []
        for alpha in alphas:
            transformed_dataset = self.cpca_alpha(dataset=dataset, alpha=alpha)
            data_to_plot.append(transformed_dataset)
        return data_to_plot, alphas        

    """
    Returns active and bg dataset projected in the cpca direction, as well as the top c_cpca eigenvalues indices.
    If specified, it returns the top_cpca directions
    """
    def cpca_alpha(self, dataset, alpha=1):
        n_components = self.n_components
        sigma = self.fg_cov - alpha*self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -n_components)[-n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:,eig_idx]
        reduced_dataset = dataset.dot(v_top)
        reduced_dataset[:,0] = reduced_dataset[:,0]*np.sign(reduced_dataset[0,0])
        reduced_dataset[:,1] = reduced_dataset[:,1]*np.sign(reduced_dataset[0,1])
        return reduced_dataset
    
    """
    This method performs spectral clustering on the affinity matrix of subspaces
    returned by contrastive pca, and returns (`=3) exemplar values of alpha
    """
    def find_spectral_alphas(self, n_alphas, max_log_alpha, n_alphas_to_return):
        self.create_affinity_matrix(max_log_alpha, n_alphas)
        affinity = self.affinity_matrix
        spectral = cluster.SpectralClustering(n_clusters=n_alphas_to_return, affinity='precomputed')
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        spectral.fit(affinity)        
        labels = spectral.labels_
        best_alphas = list()
        for i in range(n_alphas_to_return):
            idx = np.where(labels==i)[0]
            if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
                affinity_submatrix = affinity[idx][:, idx]
                sum_affinities = np.sum(affinity_submatrix, axis=0)
                exemplar_idx = idx[np.argmax(sum_affinities)]
                best_alphas.append(alphas[exemplar_idx])
        return np.sort(best_alphas), alphas, affinity[0,:], labels        

    """
    This method creates the affinity matrix of subspaces returned by contrastive pca
    """
    def create_affinity_matrix(self, max_log_alpha, n_alphas):
        from math import pi
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,n_alphas)))
        subspaces = list()
        k = len(alphas)
        affinity = 0.5*np.identity(k) #it gets doubled
        for alpha in alphas:
            space = self.cpca_alpha(dataset=self.fg, alpha=alpha)
            q, r = np.linalg.qr(space)
            subspaces.append(q)
        for i in range(k):
            for j in range(i+1,k):
                q0 = subspaces[i]
                q1 = subspaces[j]
                u, s, v = np.linalg.svd(q0.T.dot(q1))
                affinity[i,j] = s[0]*s[1]
        affinity = affinity + affinity.T
        self.affinity_matrix = np.nan_to_num(affinity)
        