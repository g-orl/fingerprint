import numpy
from load import *
import scipy.linalg


def compute_lda(D, L, n_classes, m):
    #SB
    SB = numpy.zeros((D.shape[0],D.shape[0]))
    mu = D.mean(1)
    mu = vcol(mu)

    for i in range(n_classes):
        nc = D[:,L==i].shape[1]

        mc = D[:,L==i].mean(1)
        mc = vcol(mc)

        SB += nc * numpy.dot(mc - mu, (mc - mu).T)
    
    SB /= D.shape[1]

    #SW
    SW = numpy.zeros((D.shape[0],D.shape[0]))
    for i in range(n_classes):
        mc = vcol(D[:, L==i].mean(1))
        SW += numpy.dot(D[:,L==i]-mc, (D[:,L==i]-mc).T)
    SW /= D.shape[1]

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    UW, _, _ = numpy.linalg.svd(W)
    U = UW[:, 0:m]

    U, s, _ = numpy.linalg.svd(SW)

    P1 = -numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    SBT = numpy.dot(P1, numpy.dot(SB, P1.T))

    # need to find the eigen vectors of SBT
    U, s, Vh = numpy.linalg.svd(SBT)

    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    
    DP = numpy.dot(W.T, D)

    # DP_0 = DP[:, L==0]
    # DP_1 = DP[:, L==1]
# 
    # plt.figure()
    # plt.scatter(DP_0[0], DP_0[1], label="Spoofed")
    # plt.scatter(DP_1[0], DP_1[1], label="True")
    # plt.legend()
    # plt.show() 

    return DP
