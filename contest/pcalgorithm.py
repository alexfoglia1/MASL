from itertools import combinations
from scipy.stats import norm
import pandas as pd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt

complete = lambda n : [[1 for i in range(n)] for i in range(n)]
rnorm = lambda n : np.random.normal(size=n) 


def get_skeleton(dataset, alpha, labels):
    corr_matrix = dataset.corr().values
    N = dataset.values.shape[0]
    n = len(corr_matrix[0])
    G = complete(n)
    for i in range(n):
        G[i][i] = 0
    sep_set = [[[] for i in range(n)] for j in range(n)]
    stop = False
    l = 0
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
                    for K in set(combinations(neighbors, l)):
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

def to_cpdag(skeleton, sep_set):
    def getIndependents(cpdag,reqij,reqji):
        ind = []
        for i in range(len(cpdag)):
            for j in range(len(cpdag)):
                if cpdag[i][j] == reqij and (reqji == None or cpdag[j][i] == reqji):
                    ind.append((i,j))
        return sorted(ind, key = lambda z:(z[1],z[0]))
    cpdag = skeleton.tolist()
    ind = getIndependents(skeleton,1,None)
    for x, y in ind:
        allZ = []
        for z in range(len(cpdag)):
            if skeleton[y][z] == 1 and z != x:
                allZ.append(z)
        for z in allZ:
            if skeleton[x][z] == 0 and sep_set[x][z] != None and sep_set[z][x] != None and not (
                    y in sep_set[x][z] or y in sep_set[z][x]):
                cpdag[x][y] = cpdag[z][y] = 1
                cpdag[y][x] = cpdag[y][z] = 0
    #rule 1
    search = list(cpdag)
    ind = getIndependents(cpdag,1,0)
    for a,b in ind:
        found = []
        for i in range(len(search)):
            if (search[b][i] == 1 and search[i][b] == 1) and (search[a][i] == 0 and search[i][a] == 0):
                 found.append(i)
        if len(found) > 0:
            for c in found:
                if cpdag[b][c] == 1 and cpdag[c][b] == 1:
                    cpdag[b][c] = 1
                    cpdag[c][b] = 0
                elif cpdag[b][c] == 0 and cpdag[c][b] == 1:
                    cpdag[b][c] = 2
                    cpdag[c][b] = 2
    #rule2
    search = list(cpdag)
    ind = getIndependents(cpdag,1,1)
    for a, b in ind:
        found = []
        for i in range(len(search)):
            if (search[a][i] == 1 and search[i][a] == 0) and (search[i][b] == 1 and search[b][i] == 0):
                found.append(i)
        if len(found) > 0:
            if cpdag[a][b] == 1 and cpdag[b][a] == 1:
                cpdag[a][b] = 1
                cpdag[b][a] = 0
            elif cpdag[a][b] == 0 and cpdag[b][a] == 1:
                cpdag[a][b] = cpdag[b][a] = 2    
    #rule3
    search = list(cpdag)
    ind = getIndependents(cpdag,1,1)
    for a, b in ind:
        found = []
        for i in range(len(search)):
            if (search[a][i] == 1 and search[i][a] == 1) and (search[i][b] == 1 and search[b][i] == 0):
                found.append(i)
        if len(found) >= 2:
            for c1, c2 in combinations(found, 2):
                if search[c1][c2] == 0 and search[c2][c1] == 0:
                    if search[a][b] == 1 and search[b][a] == 1:
                        cpdag[a][b] = 1
                        cpdag[b][a] = 0
                        break
                    elif search[a][b] == 0 and search[b][a] == 1:
                        cpdag[a][b] = cpdag[b][a] = 2
                        break
    return np.array(cpdag)

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

def test_butterfly_model():
    alpha = .10
    dataset = pd.read_csv("marks.dat",sep=",")
    varnames = dataset.columns
    import time
    t0 = time.time()
    (g,sep_set) = get_skeleton(dataset, alpha, dataset.columns)
    tf = time.time()
    print "Elapsed "+str(tf-t0)+" sec"
    print(g)
    plot(g,varnames)
    g = to_cpdag(g,sep_set)
    print(g)
    plot(g,varnames)

def from_file(filename, separator = ","):
    alpha = .05
    dataset = pd.read_csv(filename,sep = separator)
    (g,sep_set) = get_skeleton(dataset, alpha,dataset.columns)
    print(g)
    plot(g,dataset.columns)
    g = to_cpdag(g,sep_set)
    print(g)
    plot(g,dataset.columns)

def gen_lin_reg_model(a,b,N):
    alpha = .05
    import time
    t0 = time.time()
    x1 = rnorm(N)
    x2 = a*x1+rnorm(N)
    x3 = a*x1+rnorm(N)
    x4 = a*x1+rnorm(N)
    y = b*x2 + b*x3 + b*x4 + rnorm(N) + 5
    tf = time.time()
    delta_t0 = (tf-t0)
    dataset = pd.DataFrame({'x1': x1.tolist(),
                       'x2': x2.tolist(),
                       'x3': x3.tolist(),
                       'x4': x4.tolist(),
                        'y': y.tolist()
                        })
    t0 = time.time()
    (g,sep_set) = get_skeleton(dataset, alpha,dataset.columns)
    tf = time.time()
    delta_t1 = (tf-t0)
    print(g)
    #plot(g,dataset.columns)
    t0 = time.time()
    g = to_cpdag(g,sep_set)
    tf = time.time()
    print(g)
    delta_t2 = (tf-t0)
    #plot(g,dataset.columns)
    return (delta_t0,delta_t1,delta_t2)

def gen_lrg2():
    alpha = .05
    N = 99999
    x1 = rnorm(N)
    x2 = 3*x1+rnorm(N)
    y = 3*x2+rnorm(N)
    dataset = pd.DataFrame({
                       'x1': x1.tolist(),
                       'x2': x2.tolist(),
                        'y': y.tolist()
                        })
    (g,sep_set) = get_skeleton(dataset,alpha,dataset.columns)
    g = to_cpdag(g,sep_set)
    plot(g,dataset.columns)

def test_linear_regression():
    (t0,t1,t2) = gen_lin_reg_model(3,5,999999)
    print "Time to generate: "+str(t0)+" sec"
    print "Time to estimate skeleton: "+str(t1)+" sec"
    print "Time to estimate dag: "+str(t2)+" sec"
    print "Total time: "+str(t1+t0+t2)+" sec"
    
if __name__ == '__main__':
    test_butterfly_model()
    #from_file('marks.dat',separator = ',')


    
        
           
    
