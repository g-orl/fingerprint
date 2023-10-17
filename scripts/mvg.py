import numpy
from load import *
import math
import scipy

import numpy
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')


def compute_mu(DL):
    MU = vcol(DL.mean(1))
    return MU

def compute_c_tied(D, L, MU):
    # need to compute for each class the covariance matrix
    DL_0 =  D[:, L==0]
    DL_1 =  D[:, L==1]

    # covariance class 0
    DL_0C = DL_0 - MU[0]
    CC0 = numpy.dot(DL_0C, DL_0C.T)

    # covariance class 1
    DL_1C = DL_1 - MU[1]
    CC1 = numpy.dot(DL_1C, DL_1C.T)

    # need to sum all matrixes
    C = CC0 + CC1
    C = C / D.shape[1]
    return C

def compute_mu_c(D, L, label):
    DL = D[:, L==label]
    MU = compute_mu(DL)
    DLC = DL - MU
    C = numpy.dot(DLC, DLC.T)/(DLC.shape[1])
    return (MU, C)

def correlation_matrix(D, L, label, path):
    mu, C = compute_mu_c(D, L, 0)
    print(C)
    correlation = numpy.zeros(C.shape)
    for i in range(0, D.shape[0]):
        for j in range(0, D.shape[0]):
            correlation[i][j] = C[i][j] / (numpy.sqrt(C[i][i] * C[j][j]))
    plt.figure()
    plt.title("Correlation matrix for target class")
    plt.imshow(correlation, cmap="autumn")
    plt.xticks(numpy.arange(0, D.shape[0]))
    plt.yticks(numpy.arange(0, D.shape[0]))
    plt.colorbar()
    plt.savefig(path)

def compute_mu_c_nb(D, L, label):
    DL = D[:, L==label]
    MU = compute_mu(DL)
    DLC = DL - MU
    C = numpy.dot(DLC, DLC.T)/(DLC.shape[1])
    # need to set to zero all the elements out of the diagonal
    identityMatrix = numpy.identity(DL.shape[0])
    # multipy element wise C and identity
    C = numpy.multiply(C, identityMatrix)
    return (MU, C)


def logpdf_GAU_ND2(X, mu, C):
    # X array of shape(M, N)
    # mu array of shape (M, 1)
    # C array of shape (M, M) that represents the covariance matrix
    M = C.shape[0] #number of features
    # N = X.shape[1] #number of samples
    invC = numpy.linalg.inv(C) #C^-1
    logDetC = numpy.linalg.slogdet(C)[1] #log|C|
    
    # with the for loop:
    # logN = np.zeros(N)
    # for i, sample in enumerate(X.T):
    #     const = -0.5*M*np.log(2*np.pi)
    #     dot1 = np.dot((sample.reshape(M, 1) - mu).T, invC)
    #     dot2 = np.dot(dot1, sample.reshape(M, 1) - mu)
    #     logN[i] = const - 0.5*logDetC - 0.5*dot2

    XC = (X - mu).T # XC has shape (N, M)
    const = -0.5*M*numpy.log(2*numpy.pi)

    # sum(1) sum elements of the same row togheter
    # multiply make an element wise multiplication
    logN = const - 0.5*logDetC - 0.5*numpy.multiply(numpy.dot(XC, invC), XC).sum(1)

    # logN is an array of length N (# of samples)
    # each element represents the log-density of each sample
    return logN

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0] # size of the feature vector

    det = numpy.linalg.slogdet(C)[1]

    result = [-(M/2) * math.log(2 * math.pi) - (1/2)*det - 1/2 * numpy.dot((y.reshape(M, 1) - mu).T, numpy.dot(numpy.linalg.inv(C), y.reshape(M, 1) - mu))[0][0] for y in x.T]
    return result

def mvg_classifier(DTR, LTR, DTE, PC):

    (MU0, C0) = compute_mu_c(DTR, LTR, 0)
    (MU1, C1) = compute_mu_c(DTR, LTR, 1)

    S0 = logpdf_GAU_ND2(DTE, MU0, C0)
    S1 = logpdf_GAU_ND2(DTE, MU1, C1)

    # return the loglikelihood ratio
    return numpy.subtract(S1, S0)
    
    # S0 = S0 + numpy.log(1-PC)
    # S1 = S1 + numpy.log(PC)
    
    # LogSJoint = numpy.vstack([S0, S1])

    # LogSMarginal = vrow(scipy.special.logsumexp(LogSJoint, axis=0))
    # LogSPost = LogSJoint - LogSMarginal

    # SPost = numpy.exp(LogSPost)

    # PL = numpy.argmax(SPost, 0)

    # # the function returns the arrow of predicted labels
    # return PL

def mvg_classifier_nb(DTR, LTR, DTE, PC):

    (MU0, C0) = compute_mu_c_nb(DTR, LTR, 0)
    (MU1, C1) = compute_mu_c_nb(DTR, LTR, 1)

    S0 = logpdf_GAU_ND2(DTE, MU0, C0)
    S1 = logpdf_GAU_ND2(DTE, MU1, C1)

    return numpy.subtract(S1, S0)
    # S = numpy.vstack([S0, S1])
 
    # LogSJoint = S + numpy.log(PC)

    # LogSMarginal = vrow(scipy.special.logsumexp(LogSJoint, axis=0))
    # LogSPost = LogSJoint - LogSMarginal

    # SPost = numpy.exp(LogSPost)
    # PL = numpy.argmax(SPost, 0) 

    # return PL


def mvg_classifier_tied(DTR, LTR, DTE, PC):

    MU0 = compute_mu(DTR[:, LTR==0])
    MU1 = compute_mu(DTR[:, LTR==1])
    
    C = compute_c_tied(DTR, LTR, [MU0, MU1])

    S0 = logpdf_GAU_ND2(DTE, MU0, C)
    S1 = logpdf_GAU_ND2(DTE, MU1, C)

    return numpy.subtract(S1, S0)

    # S = numpy.vstack([S0, S1])

    # LogSJoint = S + numpy.log(1/3)

    # LogSMarginal = vrow(scipy.special.logsumexp(LogSJoint, axis=0))
    # LogSPost = LogSJoint - LogSMarginal

    # SPost = numpy.exp(LogSPost)

    # PL = numpy.argmax(SPost, 0)
    # return PL


