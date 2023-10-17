from load import *
from pca import *
from lda import *

import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg') 

def center(D):
    mu = numpy.mean(D, axis=1)
    mu = mu.reshape((D.shape[0],1))

    return D - mu

def z_norm(D):
    mu = numpy.mean(D, axis=1)
    mu = mu.reshape((D.shape[0],1))

    DC = D - mu
    C = numpy.dot(DC, DC.T)
    C = C / D.shape[1]

    V = numpy.diag(C)
    SV = numpy.sqrt(V)
    return DC / SV.reshape(D.shape[0], 1)

def whiten(D, C):
    E = scipy.linalg.fractional_matrix_power(C, 0.5)
    return numpy.dot(E, D)

def l2(D):
    norm = numpy.linalg.norm(D, axis=0)
    return D / norm.reshape((1, D.shape[1]))


if __name__=="__main__":
    [D, L] = load_data("../data/Train.txt", 10)

    # print(D[:, L==0].shape)
    # print(D[:, L==1].shape)
    # # will print the correlation between all the features
    # print(D.shape)
    # # print for each attribute the corresponding attribtue for each class
    # D0 = D[:, L == 0]
    # D1 = D[:, L == 1]

    # for i in range(0, D.shape[0]):    
    #     plt.figure()
    #     plt.title("Feature" + i.__str__())
    #     plt.hist(D0[i, :], 50, label="Spoofed", alpha=0.5, density=True)
    #     plt.hist(D1[i, :], 50, label="Authentic", alpha=0.5, density=True)
    #     plt.legend()
    #     plt.savefig("../assets/feature-distribution/feature" + i.__str__())

    # for i in range(0, D.shape[0]):
    #     for j in range(0, D.shape[0]):
    #         if(i != j):
    #             plt.figure()
    #             plt.scatter(D0[i, :], D0[j, :], label="Spoofed", alpha=0.5)
    #             plt.scatter(D1[i, :], D1[j, :], label="Authentic", alpha=0.5)
    #             plt.legend()
    #             plt.xlabel("Feature " + i.__str__())
    #             plt.xlabel("Feature " + j.__str__())
    #             plt.savefig("../assets/feature-pair-distribution/feature" + i.__str__() + "-" + j.__str__())
    #             plt.close()

    # DPCA = compute_pca(D, 2)
    # DP_0 = DPCA[:, L==0]
    # DP_1 = DPCA[:, L==1]

    # for i in range(0, DPCA.shape[0]):    
    #     plt.figure()
    #     plt.title("Feature" + i.__str__())
    #     plt.hist(DP_0[i, :], 50, label="Spoofed", alpha=0.5, density=True)
    #     plt.hist(DP_1[i, :], 50, label="Authentic", alpha=0.5, density=True)
    #     plt.legend()
    #     plt.savefig("../assets/pca-feature-distribution/feature" + i.__str__())

    DLDA = compute_lda(D, L, 2, 1)
    D0 = DLDA[:, L==0]
    D1 = DLDA[:, L==1]

    for i in range(0, DLDA.shape[0]):    
        plt.figure()
        plt.title("Feature" + i.__str__())
        plt.hist(D0[i, :], 50, label="Spoofed", alpha=0.5, density=True)
        plt.hist(D1[i, :], 50, label="Authentic", alpha=0.5, density=True)
        plt.legend()
        plt.savefig("../assets/lda-feature-distribution/feature" + i.__str__())

    # for i in range(0, DPCA.shape[0]):
    #     for j in range(0, DPCA.shape[0]):
    #         if(i != j):
    #             plt.figure()
    #             plt.scatter(DP_0[i, :], DP_0[j, :], label="Spoofed", alpha=0.3)
    #             plt.scatter(DP_1[i, :], DP_1[j, :], label="Authentic", alpha=0.3)
    #             plt.legend()
    #             plt.xlabel("Feature " + i.__str__())
    #             plt.xlabel("Feature " + j.__str__())
    #             plt.savefig("../assets/pca-pair-distribution/feature" + i.__str__() + "-" + j.__str__())
    #             plt.close()
    
    # cumulative_variance = explained_variance(D)
    # plt.figure()
    # plt.title("PCA - explained variance")
    # plt.plot(numpy.linspace(0, 10, 11), cumulative_variance)
    # plt.xlabel("PCA dimensions")
    # plt.ylabel("Fraction of cumulative explained variance")
    # plt.xticks(numpy.arange(0,11))
    # plt.yticks(numpy.arange(0.0, 1.1, 0.1))
    # plt.grid(visible=True)
    # plt.savefig("../assets/pca-explained-variance/expvariance")
    # plt.close()