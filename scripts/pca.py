import numpy
from load import *
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')

def compute_pca(D, m):
    mu = D.mean(1)
    
    # print(mu)
    DC = D - vcol(mu)
    
    C = (1/numpy.shape(D)[1]) * numpy.dot(DC, DC.T)

    s, U = numpy.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]
    
    DP = numpy.dot(P.T, D)

    # scatter plot possible if # of dimension is 2 or 
     
    return DP

def compute_pca_direction(D, m):
    mu = D.mean(1)
    
    # print(mu)
    DC = D - vcol(mu)
    
    C = (1/numpy.shape(D)[1]) * numpy.dot(DC, DC.T)

    s, U = numpy.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]   
    return P



def explained_variance(D):
    mu = D.mean(1)
    # print(mu)
    DC = D - vcol(mu)
    
    C = (1/numpy.shape(D)[1]) * numpy.dot(DC, DC.T)

    s, U = numpy.linalg.eigh(C)

    
    ordered_s = numpy.flip(s)
    eig_variance = numpy.divide(ordered_s, ordered_s.sum())
    print(eig_variance)
    for i in range(0, 9):
        eig_variance[i+1] += eig_variance[i]
    
    return numpy.concatenate([[0], eig_variance])
    # P = U[:, ::-1][:, 0:m]
    