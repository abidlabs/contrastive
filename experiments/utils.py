from copy import deepcopy
import numpy as np, matplotlib.pyplot as plt
from sklearn import cluster
from PIL import Image,ImageOps

def resize_and_crop(img, size=(100,100), crop_type='middle'):
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally
    # depending on the ratio
    if ratio > img_ratio:
        img = img.resize((
            size[0],
            int(round(size[0] * img.size[1] / img.size[0]))),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (
                0,
                int(round((img.size[1] - size[1]) / 2)),
                img.size[0],
                int(round((img.size[1] + size[1]) / 2)))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((
            int(round(size[1] * img.size[0] / img.size[1])),
            size[1]),
            Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (
                int(round((img.size[0] - size[0]) / 2)),
                0,
                int(round((img.size[0] + size[0]) / 2)),
                img.size[1])
        elif crop_type == 'bottom':
            box = (
                img.size[0] - size[0],
                0,
                img.size[0],
                img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((
            size[0],
            size[1]),
            Image.ANTIALIAS)
    # If the scale is the same, we do not need to crop
    return img

class Dataset(object):
    # Returns data    
    def get_data(self):
        return self.data    
    # Returns the background data 
    def get_bg(self):
        return self.bg    
    # Returns the active data
    def get_active(self):
        return self.active
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
    def standardize(self, array):
        standardized_array =  (array-np.mean(array,axis=0)) / np.std(array,axis=0)
        return np.nan_to_num(standardized_array)

    """ 
    Initialization performs the following operations
    1) Centering the active, background seperately
    2) Standardize if to_standardize is specified as True
    3) Calculate the covariance_matrices
    """
    def __init__(self, to_standardize=True):
        #from contrastive import PCA
        
        # Housekeeping
        self.pca_directions = None
        self.bg_eig_vals = None
        self.affinity_matrix = None
        
        # Dataset sizes
        self.n_active, _           = self.active.shape
        self.n_bg, self.features_d = self.bg.shape
        
        #Center the background data
        self.bg = self.bg - np.mean(self.bg, axis=0)
        if to_standardize: #Standardize if specified
            self.bg = self.standardize(self.bg)
        #Calculate the covariance matrix
        self.bg_cov = self.bg.T.dot(self.bg)/(self.bg.shape[0]-1)
        
        #Center the active data
        self.active = self.active - np.mean(self.active, axis=0)
        if to_standardize: #Standardize if specified
            self.active = self.standardize(self.active)
        #Calculate the covariance matrix
        self.active_cov = self.active.T.dot(self.active)/(self.n_active-1)

        #self.cpca = PCA()
        #self.cpca.fit_transform(foreground=self.active, background=self.bg)

    """
    Perfomes plain vanilla pca on the active dataset (TO DO: write the same code for background )
    Not a part of init because this might be time consuming
    """
    def pca_active(self, n_components = 2):
        
        # Perform PCA only once (to save computation time)
        if self.pca_directions is None:
            #print("PCA is being perfomed on the dataset")
            # Calculating the top eigen vectors on the covariance of the active dataset
            w, v = LA.eig(self.active_cov)
            # Sorting the vectors in the order of eigen values
            idx  = w.argsort()[::-1]
            idx  = idx[:n_components]
            w    = w[idx]

            # Storing the top_pca_directions
            self.pca_directions = v[:,idx]
            # Storing the active dataset projected on the top_pca_directions
            self.active_pca     = self.active.dot(self.pca_directions)
        else:
            print("PCA has been previously perfomed on the dataset")

    """
    Returns active and bg dataset projected in the cpca direction, as well as the top c_cpca eigenvalues indices.
    If specified, it returns the top_cpca directions
    """
    def cpca_alpha(self, n_components = 2, alpha=1, return_eigenvectors=False):
        #return None, self.cpca.cpca_alpha(dataset=self.active,alpha=alpha), None
        sigma = self.active_cov - alpha*self.bg_cov
        w, v = LA.eig(sigma)
        eig_idx = np.argpartition(w, -n_components)[-n_components:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        v_top = v[:,eig_idx]
        reduced_foreground = self.active.dot(v_top)
        reduced_background = self.bg.dot(v_top)

        reduced_foreground[:,0] = reduced_foreground[:,0]*np.sign(reduced_foreground[0,0])
        reduced_foreground[:,1] = reduced_foreground[:,1]*np.sign(reduced_foreground[0,1])
        
        
        if (return_eigenvectors):
            #return eig_idx, reduced_foreground, reduced_background, v_top
            print("WHat?")
            return None
        else:
            #return eig_idx, reduced_foreground, reduced_background
            return None, reduced_foreground, None

    """
    This function performs contrastive PCA using the alpha technique on the 
    active and background dataset. It automatically determines n_alphas=4 important values
    of alpha up to based to the power of 10^(max_log_alpha=5) on spectral clustering
    of the top subspaces identified by cPCA.
    The final return value is the data projected into the top (n_components = 2) 
    subspaces, which can be plotted outside of this function
    """
    def automated_cpca(self,  n_alphas=4, max_log_alpha=5, n_components = 2, affinity_metric='determinant', exemplar_method='medoid'):
        best_alphas, all_alphas, angles0, labels = self.find_spectral_alphas(n_components, n_alphas-1, max_log_alpha, affinity_metric, exemplar_method)
        best_alphas = np.concatenate(([0], best_alphas)) #one of the alphas is always alpha=0
        data_to_plot = []
        for alpha in best_alphas:
            _, r_active, r_bg = self.cpca_alpha(n_components=n_components, alpha=alpha)
            data_to_plot.append((r_active, r_bg))
        return data_to_plot, best_alphas

    
    
    """
    This function performs contrastive PCA using the alpha technique on the 
    active and background dataset. It returns the cPCA-reduced data for all values of alpha specified,
    both the active and background, as well as the list of alphas
    """
    def manual_cpca(self, max_log_alpha=5, n_components = 2):
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
        data_to_plot = []
        for alpha in alphas:
            n, r_active, r_bg = self.cpca_alpha(n_components, alpha=alpha)
            data_to_plot.append((r_active, r_bg))
        return data_to_plot, alphas
    
    
    """
    This method performs spectral clustering on the affinity matrix of subspaces
    returned by contrastive pca, and returns (`=3) exemplar values of alpha
    """
    def find_spectral_alphas(self, n_components=2, max_num=3, max_log_alpha=5, affinity_metric='determinant', exemplar_method='medoid'):
        #if (self.affinity_matrix is None): #commented out because different kinds of affinity can be defined
        self.create_affinity_matrix(n_components, max_log_alpha, affinity_metric)
        affinity = self.affinity_matrix
        spectral = cluster.SpectralClustering(n_clusters=max_num+1, affinity='precomputed')
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
        spectral.fit(affinity)
        labels = spectral.labels_
        best_alphas = list()
        for i in range(max_num+1):
            idx = np.where(labels==i)[0]
            if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
                if exemplar_method=='smallest':
                    exemplar_idx = int(np.min(idx)) #min seems to work better than median though I don't know why exactly
                elif exemplar_method=='medoid':
                    affinity_submatrix = affinity[idx][:, idx]
                    sum_affinities = np.sum(affinity_submatrix, axis=0)
                    exemplar_idx = idx[np.argmax(sum_affinities)]
                else:
                    raise ValueError("Invalid specification of exemplar method")
                best_alphas.append(alphas[exemplar_idx])
        return np.sort(best_alphas), alphas, affinity[0,:], labels

    """
    This method creates the affinity matrix of subspaces returned by contrastive pca
    """
    def create_affinity_matrix(self, n_components=2, max_log_alpha=5, affinity_metric='determinant'):
        from math import pi
        alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
        print(alphas)
        subspaces = list()
        k = len(alphas)
        if affinity_metric=='principal':
            affinity = pi/4*np.identity(k)
        elif affinity_metric=='determinant':
            affinity = 0.5*np.identity(k) #it gets doubled
        for alpha in alphas:
            n, space, _ = self.cpca_alpha(n_components, alpha=alpha)
            q, r = np.linalg.qr(space)
            subspaces.append(q)
        for i in range(k):
            for j in range(i+1,k):
                q0 = subspaces[i]
                q1 = subspaces[j]
                u, s, v = np.linalg.svd(q0.T.dot(q1))
                if affinity_metric=='principal':
                    angle = np.arccos(s[0])
                    affinity[i,j] = pi/2 - angle
                elif affinity_metric=='determinant':
                    affinity[i,j] = np.prod(s)
        affinity = affinity + affinity.T
        self.affinity_matrix = np.nan_to_num(affinity)
        
        
        

        
        
        
        
