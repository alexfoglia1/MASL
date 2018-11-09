import itertools
import copy
import numpy
import graphtools

def matstr(A):
    tos = ""
    for row in A:
        tos = tos + str(row)+"\n"
    return tos

def complete(n):
    A = list()
    for i in range(0,n):
        row = list()
        for j in range(0,n):
            if i == j:
                row.append(0)
            else:
                row.append(1)
        A.append(row)
    return A

def setdiff(A,B):
    ret_set = copy.copy(A)
    for x in B:
        if x in A:
            ret_set.remove(x)
    return ret_set

def adj(i,G):
    adjacents = list()
    for j in range(0,len(G[i])):
        if i!=j and G[i][j] == 1:
            adjacents.append(j)
    return adjacents

def findsubsets(S,l):
    return list(itertools.combinations(S,l))

def erfinv(x):
    sgn = 1
    a = 0.147
    PI = numpy.pi
    if x<0:
        sgn = -1
    temp = 2/(PI*a) + numpy.log(1-x**2)/2
    add_1 = temp**2
    add_2 = numpy.log(1-x**2)/a
    add_3 = temp
    rt1 = (add_1-add_2)**0.5
    rtarg = rt1 - add_3
    return sgn*(rtarg**0.5)

def phi(p):
    return (2**0.5)*erfinv(2*p-1)

def test(n,z,i,j,a,K):
    root = (n-len(K)-3)**0.5
    return root*abs(z(i,j)) <= phi(1-a/2)

def meanof(dataset):
    n = len(dataset[0])
    m = []
    for i in range(0,n):
        m.append(0.0)
    datasize = len(dataset)
    for i in range(0,datasize):
        for j in range(0,n):
            m[j] = m[j] + float(dataset[i][j])
    for i in range(0,n):
        m[i] = m[i] / datasize
    return m

def zeros(n,m):
    zer = []    
    for i in range(0,n):
        row = []
        for j in range(0,m):
            row.append(0)
        zer.append(row)
    return zer

def getcol(i,matrix):
    col = []
    for row in matrix:
        col.append(row[i])
    return col

def sigma(dataset,means):
    n = len(means)
    sigma = zeros(n,n)
    for i in range(0,n):
        for j in range(0,n):
            dset_i = getcol(i,dataset)
            dset_j = getcol(j,dataset)
            means_i = means[i]
            means_j = means[j]
            sigma[i][j] = covar(dset_i,dset_j,means_i,means_j)
    return sigma

def covar(X,Y,ux,uy):
    n = len(X)
    s = 0
    for i in range(0,n):
        s = s +(X[i] - ux)*(Y[i] - uy);
    return float(s)/n

def getSigma(dataset):
    means = meanof(dataset)
    return sigma(dataset,means)

def getInverse(A):
    return numpy.linalg.inv(A)

def pc_algorithm(a,sigma_inverse):
    l = - 1
    n = len(sigma_inverse)
    z = lambda i,j : -sigma_inverse[i][j]/((sigma_inverse[i][i]*sigma_inverse[j][j])**0.5)
    act_g = complete(n)
    act = 0
    while l<n-1:
        l = l + 1
        for i in range(0,n):
            for j in range(i+1,n):
                if(act_g[i][j]!=1):
                    continue
                adjacents = adj(i,act_g)
                if len(adjacents)==0:
                    continue
                act_set = setdiff(adjacents,[j])
                all_k = findsubsets(act_set, l)
                counter = 0
                while(act_g[i][j]!=0):
                    if(counter >= len(all_k)):
                        break
                    K = all_k[counter];
                    counter = counter+1
                    if test(n,z,i,j,a,K):
                        act_g[i][j] = 0
    for i in range(0,n):
        for j in range(0,n):
            if i>j:
                act_g[i][j] = act_g[j][i]
    return act_g;

def rand_set(X,Var):
    n = len(X)
    rset = []
    for i in range(0,n):
        rset.append(numpy.random.normal(X[i],Var[i],n))
    return numpy.transpose(rset); 

def make_graph_from_dataset(dataset,alpha):
    sigma = getSigma(dataset)
    sigma_inverse = getInverse(sigma)
    return pc_algorithm(alpha,sigma_inverse)

def plot(adj_matrix,varnames):
    graphtools.plot_graph(adj_matrix,varnames)

def get_rand_dataset(dim):
    return rand_set(range(0,dim),range(1,dim+1))

def butterfly():
    return [[0.8, 0.5, 0, 0.6],[0.5, 1.4, -0.6, 0.4],[0, -0.6, 1.2, -0.3],[0.6, 0.4, -0.3, 1]]

if __name__ == '__main__':
    dataset = get_rand_dataset(7)
    varnames = ['X1','X2','X3','X4','X5','X6','X7']
    alpha = 0.50
    G = make_graph_from_dataset(dataset,alpha)
    print "Graph results: "
    print matstr(G)  
    plot(G,varnames)
            
