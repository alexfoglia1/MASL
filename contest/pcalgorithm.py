import itertools
import copy
import numpy
import graphtools
from gsq.ci_tests import ci_test_bin, ci_test_dis
#from gsq.gsq_testdata import bin_data, dis_data


def generate(n):
    rnorm = lambda x : numpy.random.normal(size=n)    
    x1 = rnorm(n)
    x2 = rnorm(n)
    x3 = rnorm(n)
    x4 = rnorm(n)
    y  = x1 + x2 + x3 + x4 + 5
    return y

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

def adj(i,G):
    adjacents = list()
    for j in range(0,len(G[i])):
        if G[i][j] == 1:
            adjacents.append(j)
    return adjacents

def test(dataset,i,j,k):
    return ci_test_dis(dataset,i,j,k)



def _test(n,z,i,j,a,K):
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
    
    phi = lambda p : (2**0.5)*erfinv(2*p-1)
    root = (n-len(K)-3)**0.5
    print str(root*abs(z(i,j)))+"<="+str(phi(1-a/2))+"?"
    return root*abs(z(i,j)) <= phi(1-a/2)

def pc_algorithm(a,dataset):
    l = -1
    n = range(dataset.shape[1])
    sep_set = [ [set() for i in n] for j in n]
    act_g = complete(len(n))
    sigma = numpy.cov(dataset)
    sigma_inverse = numpy.linalg.inv(sigma)
    z = lambda i,j : -sigma_inverse[i][j]/((sigma_inverse[i][i]*sigma_inverse[j][j])**0.5)
    while True:
        l = l+1
        flag = False
        to_remove = list()
        for (i,j) in itertools.permutations(n,2):
            adjacents_of_i = adj(i,act_g)
            if j not in adjacents_of_i:
                continue
            else:
                adjacents_of_i.remove(j)
                pass
            if len(adjacents_of_i) >=l:
                for k in itertools.combinations(adjacents_of_i,l):
                    #p_val = test(dataset,i,j,set(k))
                    #if p_val > a:
                    if _test(dataset.shape[1],z,i,j,a,set(k)):              
                        to_remove.append((i,j))
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                    else:
                        pass
                flag = True
        for (i,j) in to_remove:
            act_g[i][j] = 0
        l = l + 1
        if flag is False:
            break
    return (act_g,sep_set)

def plot(adj_matrix,varnames):
    graphtools.plot_graph(adj_matrix,varnames)

def to_dag(g,sep_set):
    n = len(g[0])
    for (i,j) in itertools.combinations(range(n),2):
        adj_i = adj(i,g)
        if j in adj_i:
            continue
        adj_j = adj(j,g)
        if i in adj_j:
            continue
        intersection = set(adj_i) & set(adj_j)
        for var in intersection:
            if var not in sep_set[i][j]:
                if g[var][i]>0:
                    g[var][i] = 0
                    pass
                if g[var][j]>0:
                    g[var][j] = 0
                    pass
    def predecessors(i):
        preds = list()
        for j in range(n):
            if g[j][i] > 0:
                preds.append(j)
        return preds
    
    for (i,j) in itertools.combinations(range(n),2):
        #rule 1
        if g[i][j] > 0 and g[j][i] > 0:
             for pred in predecessors(i):
                if g[i][pred] > 0:
                    continue
                if g[j][pred] > 0 or g[pred][j] > 0:
                    continue
                g[j][i] = 0
        #rule 2
        if g[i][j] > 0 and g[j][i] > 0:
            succs = set()
            for succ in adj(i,g):
                if g[succ][i] == 0: 
                    succs.add(succ)
            preds = set()
            for pred in predecessors(j):
                if g[j][pred] == 0:
                    preds.add(pred)
            if len(succs & preds) > 0:
                g[j][i] = 0
        #rule3
        if g[i][j] > 0 and g[j][i] > 0:
            adjacents = set()
            for adjacent in adj(i,g):
                if g[adjacent][i] > 0:
                    adjacents.add(adjacent)
            for (k, l) in itertools.combinations(adjacents, 2):
                if g[k][l] > 0 or g[l][k] > 0:
                    continue
                if g[j][k] > 0 or g[k][j] == 0:
                    continue
                if g[j][l] > 0 or g[l][j] == 0:
                    continue
                g[j][i] = 0
    return g        
            

def read(filename):
    lines = [line.rstrip('\r\n') for line in open(filename)]
    varnames = lines[0]
    spl_varnames = varnames.split()
    rows = list()
    i = 0
    for line in lines[1:]:
        values = line.split()
        fvalues = list()
        for strvalue in values:
            floatval = float(strvalue)*1000000
            intval = int(floatval)
            fvalues.append(intval)
        rows.append(fvalues)
        print(fvalues)
    return (numpy.array(rows),spl_varnames)
        

if __name__ == '__main__':
    N = 1000
    VARIABLES = 5
    dataset = generate(N)
    dataset = dataset.reshape(N/VARIABLES,VARIABLES)
    varnames = ["x1","x2","x3","x4","y"]
    #(dataset,varnames) = read("aircraft.dat")
    alpha = 0.01
    (G,sep_set) = pc_algorithm(alpha,dataset)
    print("Skeleton: ")
    print (matstr(G))  
    plot(G,varnames)
    G = to_dag(G,sep_set)
    print("Directed acyclic graph: ")
    plot(G,varnames)
    print (matstr(G))
            
