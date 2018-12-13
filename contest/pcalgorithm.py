import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

setOfParts = lambda x : chain.from_iterable(combinations(list(x),n) for n in range(len(list(x))+1))
complete = lambda n : [[1 for i in range(n)] for i in range(n)]

def get_skeleton(dataset, alpha, labels,corr_matrix = None):
    var_covar = dataset.cov().values
    return pc_algorithm(np.linalg.pinv(var_covar), dataset.values.shape[0], alpha, labels, corr_matrix = corr_matrix)

def pc_algorithm(sigma_inverse, N, alpha, labels, corr_matrix = None):
    def tocor(vcv):
        cm = []
        for i in range(0,len(vcv)):
            cm_row = []
            for j in range(0,len(vcv[i])):
                cov_ij = vcv[i][j]
                sd_i = math.sqrt(vcv[i][i])
                sd_j = math.sqrt(vcv[j][j])
                corr_ij = cov_ij/(sd_i*sd_j)
                cm_row.append(corr_ij)
            cm.append(cm_row)
        return np.array(cm)
    if corr_matrix is None: 
        sigma = np.linalg.pinv(sigma_inverse)
        corr_matrix = tocor(sigma)
    n = len(corr_matrix[0])
    G = complete(n)
    for i in range(n):
        G[i][i] = 0
    sep_set = [[[] for i in range(n)] for j in range(n)]
    stop = False
    l = 0
    under_test = {0: 0}
    def adj(x,G):
        adjacents = list()
        for j in range(0,len(G[x])):
            if G[x][j] == 1:
                adjacents.append(j)
        return adjacents
    while stop == False and any(G):
        l1 = l + 1
        stop = True
        act_ind = []
        for i in range(len(G)):
            for j in range(len(G[i])):
                if G[i][j] == 1:
                    act_ind.append((i,j))
        for x,y in act_ind:
            if G[x][y] ==1 :      
                neighbors = adj(x,G)
                neighbors.remove(y)
                if len(neighbors) >= l:
                    if len(neighbors) > l:
                        stop = False
                    for K in set(itertools.combinations(neighbors, l)):
                        p_value = indep_test(corr_matrix, N, x, y, list(K))
                        if p_value >= alpha:
                            G[x][y] = 0
                            G[y][x] = 0
                            sep_set[x][y] = list(K)
                            break
        l = l + 1
    return (np.array(G),sep_set)

def indep_test(CM, n, i, j, K):
    if len(K) == 0:
        r = CM[i, j]
    elif len(K) == 1:
        r = (CM[i, j] - CM[i, K] * CM[j, K]) / math.sqrt((1 - math.pow(CM[j, K], 2)) * (1 - math.pow(CM[i, K], 2)))
    else:
        CM_SUBSET = CM[np.ix_([i]+[j]+K, [i]+[j]+K)]
        PM_SUBSET = np.linalg.pinv(CM_SUBSET)
        r = -1 * PM_SUBSET[0, 1] / math.sqrt(abs(PM_SUBSET[0, 0] * PM_SUBSET[1, 1]))
    r = min(0.999999, max(-0.999999,r))
    res = math.sqrt(n - len(K) - 3) * 0.5 * math.log1p((2*r)/(1-r))
    return 2 * (1 - norm.cdf(abs(res)))

def plot(toplot, labels):
    G = nx.DiGraph()
    for i in range(len(toplot)):
        G.add_node(labels[i])
        for j in range(len(toplot[i])):
            if toplot[i][j] == 1:
                G.add_edges_from([(labels[i], labels[j])])
    nx.draw(G, with_labels = True)
    plt.savefig("graph.png")
    plt.show()



def butterfly_model():
    alpha = .05
    varnames = varnames = ["mec","vec","alg","ana","stat"]
    sigma_inverse=[[1.000,0.331,0.235,0.000,0.000],
                   [0.553,1.000,0.327,0.000,0.000],
                   [0.546,0.610,1.000,0.451,0.364],
                   [0.388,0.433,0.711,1.000,0.256],
                   [0.363,0.405,0.665,0.607,1.000]]
    (g,sep_set) = pc_algorithm(np.array(sigma_inverse),100,alpha,varnames)
    print(g)
    plot(g,varnames)

def from_file(filename, separator = ","):
    alpha = .05
    dataset = pd.read_csv(filename,sep = separator)
    (g,sep_set) = get_skeleton(dataset, alpha,dataset.columns,corr_matrix = dataset.corr().values)
    print(g)
    plot(g,dataset.columns)

def generated():
    alpha = .05
    varnames = ["x1","x2","x3","x4","y"]
    N = 500
    (x1,x2,x3,x4,y) = generate(N)
    dataset = [x1,x2,x3,x4,y]
    dataset = np.array(transpose(dataset))
    sigma = np.cov(dataset)
    sigma_inverse = np.linalg.pinv(sigma)
    (g,sep_set) = pc_algorithm(sigma_inverse,N, alpha, varnames)
    print(g)
    plot(g, varnames)

if __name__ == '__main__':
    #butterfly_model()
    from_file('guPrenat.dat',separator = '\t')
    
        
           
    
