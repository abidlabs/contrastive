import numpy as np
from copy import deepcopy
from numpy import *
from numpy import linalg as LA
import numpy as np, matplotlib.pyplot as plt
from sklearn import cluster

def find_ap_alphas(active,background, n_components=2, max_num=3, max_log_alpha=2, damping=0.5):
    affinity = create_affinity_matrix(active, background, n_components, max_num, max_log_alpha)
    ap = cluster.AffinityPropagation(affinity='precomputed', damping=damping)
    alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
    ap.fit(affinity)
    labels = ap.labels_
    print(damping)
    print(labels)
    best_alphas = list()
    for i in range(np.max(labels)+1):
        idx = np.where(labels==i)[0]
        if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
            middle_idx = int(np.min(idx)) #min seems to work better than median though I don't know why exactly
            best_alphas.append(alphas[middle_idx])
    
    return np.sort(best_alphas), alphas, affinity[0,:], labels

def create_affinity_matrix(active,background, n_components=2, max_num=3, max_log_alpha=2):
    from math import pi
    alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
    subspaces = list()
    k = len(alphas)
    affinity = pi/4*np.identity(k)
    for alpha in alphas:
        n, space, _ = contrast_pca_alpha(active,background, n_components, alpha=alpha)
        q, r = np.linalg.qr(space)
        subspaces.append(q)
    for i in range(k):
        for j in range(i+1,k):
            q0 = subspaces[i]
            q1 = subspaces[j]
            u, s, v = np.linalg.svd(q0.T.dot(q1))
            angle = np.arccos(s[0])
            affinity[i,j] = pi/2 - angle
    affinity = affinity + affinity.T
    return np.nan_to_num(affinity)
        
    #calculate distance between each subspace, subtracting pi
    
def find_spectral_alphas(active,background, n_components=2, max_num=3, max_log_alpha=2):
    affinity = create_affinity_matrix(active, background, n_components, max_num, max_log_alpha)
    spectral = cluster.SpectralClustering(n_clusters=max_num+1,
                                          affinity='precomputed')
    
    alphas = np.concatenate(([0],np.logspace(-1,max_log_alpha,40)))
    spectral.fit(affinity)
    labels = spectral.labels_
    best_alphas = list()
    for i in range(max_num+1):
        idx = np.where(labels==i)[0]
        if not(0 in idx): #because we don't want to include the cluster that includes alpha=0
            middle_idx = int(np.min(idx)) #min seems to work better than median though I don't know why exactly
            best_alphas.append(alphas[middle_idx])
    
    return np.sort(best_alphas), alphas, affinity[0,:], labels

def standardize(array):
    standardized_array =  (array-np.mean(array,axis=0)) / np.std(array,axis=0)
    return np.nan_to_num(standardized_array)


"""
    Returns eigen values, vectors sorted by the *magnitude* of eigen values
    The eigen vectors are in the columns
"""
def eigen_values_abs_sorted(M):
    #Calculating the eigen values
    eig_values, eig_vecs = linalg.eigh(M)
    
    #Taking their absolute values
    eig_values = np.abs(eig_values)
    
    #Sorting the (abs) eigen values
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vecs = eig_vecs[:,idx]
    
    return eig_values, eig_vecs

def contrast_pca_alpha(foreground, background, n_components = 1, alpha=0.5, return_eigenvectors=False):
    n1, d = foreground.shape
    n2, d = background.shape
    x1c = (foreground - np.mean(foreground,axis=0))
    x2c = (background - np.mean(background,axis=0))
    sigma = x1c.T.dot(x1c)/n1 - alpha*x2c.T.dot(x2c)/n2
    w, v = LA.eig(sigma)
    idx = np.argpartition(w, -n_components)[-n_components:]
    idx = idx[np.argsort(-w[idx])]
    v_top = v[:,idx]
    reduced_foreground = x1c.dot(v_top)
    reduced_background = x2c.dot(v_top)
    if (return_eigenvectors):
        return idx, reduced_foreground, reduced_background, v_top
    else:
        return idx, reduced_foreground, reduced_background

def subspace_angles(active, background, alphas, n_components=2):
    _, space0, _ = contrast_pca_alpha(active,background, n_components, alpha=0)
    angles1 = list()
    angles2 = list()
    nums = list()
    for alpha in alphas:
        n, space1, _ = contrast_pca_alpha(active,background, n_components, alpha=alpha)
        q0, r0 = np.linalg.qr(space0)
        q1, r1 = np.linalg.qr(space1)
        u, s, v = np.linalg.svd(q0.T.dot(q1))
        angles1.append(np.arccos(s[0]))
        angles2.append(np.arccos(s[1]))
        nums.append(n)
    return angles1, angles2, nums

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind  = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        plt.plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def find_optimal_alphas(active, background, n_components=2, max_num=3, max_log_alpha=2):
    alphas = np.logspace(-1,max_log_alpha,25)
    angles1, angles2, nums = subspace_angles(active, background, alphas, n_components)
    idx_peaks = detect_peaks(np.diff(angles1)+np.array(range(1,len(angles1)))/len(angles1)/10)
    alpha_peaks = alphas[1+idx_peaks]
    if len(alpha_peaks) > max_num:
        ind = np.argpartition(alpha_peaks, -max_num)[-max_num:]
        alpha_peaks = alpha_peaks[ind]
    return alpha_peaks

def find_optimal_alphas_and_plot(active,background, A_labels, n_components=2, max_num=3, max_log_alpha=2):
    from matplotlib import pyplot as plt 
    alphas = find_optimal_alphas(active,background, n_components, max_num, max_log_alpha)
    alphas = np.concatenate(([0], alphas))
    for alpha in alphas:
        _, reduced_foreground, reduced_background = contrast_pca_alpha(active, background, n_components=2, alpha=alpha)
        plt.figure()
        plt.title(str(alpha))
        plt.scatter(reduced_foreground[:,0], reduced_foreground[:,1], color=A_labels)