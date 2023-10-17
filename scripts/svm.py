import scipy
import numpy
import math

def poly_kernel(x1, x2, c, d, e):
    return numpy.power((numpy.dot(x1.T, x2) + c), d) + e

def rbf_kernel(x1, x2, g, e):
    return numpy.exp((numpy.linalg.norm(x1-x2) ** 2) * (-g)) + e

def svm_wraper(H, DTR):
    def svm_obj(alpha):
        # we now need to return the objective L = -J

        LD = 0.5 * numpy.dot(alpha.T, numpy.dot(H, alpha)) - numpy.dot(alpha.T, numpy.ones((DTR.shape[1], 1)))
        grad = numpy.reshape(numpy.dot(H, alpha) - numpy.ones((1, DTR.shape[1])), (DTR.shape[1],1))
        
        return (LD, grad)
    
    return svm_obj

def svm_poly_classifier_wrapper(K, C, d, c):
    def compute_svm_polykernel(DTR, LTR, DTE, PC):
        Z = LTR * 2 - 1
        
        Z = numpy.reshape(Z, (LTR.shape[0], 1))
        H = numpy.dot(Z, Z.T).astype(float)

        # will compute H in with for loops
        for i in range(0, DTR.shape[1]):
            for j in range(0, DTR.shape[1]):
                H[i][j] *= poly_kernel(DTR[:, i], DTR[:, j], c, d, K**2)

        BC = [(0, C) for i in range(0, DTR.shape[1])]
        [alpha, f, d2] = scipy.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), numpy.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
        
        S = numpy.ones((DTE.shape[1]))

        for i in range(0, DTE.shape[1]):
            result = 0
            for j in range(0, DTR.shape[1]):
                result += alpha[j]*Z[j]*poly_kernel(DTR[:, j], DTE[:, i], c, d, K**2)
            S[i] = result
        
        return S
    return compute_svm_polykernel

def svm_poly_weighted_classifier_wrapper(K, C, d, c):
    def compute_svm_polykernel(DTR, LTR, DTE, PT):
        Z = LTR * 2 - 1
        Zf = Z
        
        Z = numpy.reshape(Z, (LTR.shape[0], 1))
        H = numpy.dot(Z, Z.T).astype(float)

        # will compute H in with for loops
        for i in range(0, DTR.shape[1]):
            for j in range(0, DTR.shape[1]):
                H[i][j] *= poly_kernel(DTR[:, i], DTR[:, j], c, d, K**2)

        PEMP = LTR.sum()/LTR.shape[0]

        CT = C * (PT / PEMP)
        CF = C * ((1-PT)/(1-PEMP))
        
        BC = []
        for i in range(0, DTR.shape[1]):
            if Zf[i] == 1:
                BC.append((0, CT))
            else:
                BC.append((0, CF))
                
        [alpha, f, d2] = scipy.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), numpy.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
        
        S = numpy.ones((DTE.shape[1]))

        for i in range(0, DTE.shape[1]):
            result = 0
            for j in range(0, DTR.shape[1]):
                result += alpha[j]*Z[j]*poly_kernel(DTR[:, j], DTE[:, i], c, d, K**2)
            S[i] = result
        
        return S
    return compute_svm_polykernel

def svm_rbf_classifier_wrapper(K, C, g):
    def compute_svm_rbfkernel(DTR, LTR, DTE, PC):
        Z = LTR * 2 - 1
        
        Z = numpy.reshape(Z, (LTR.shape[0], 1))
        H = numpy.dot(Z, Z.T).astype(float)

        # will compute H in with for loops
        for i in range(0, DTR.shape[1]):
            for j in range(0, DTR.shape[1]):
                H[i][j] *= rbf_kernel(DTR[:, i], DTR[:, j], g, K**2)

        BC = [(0, C) for i in range(0, DTR.shape[1])]
        [alpha, f, d2] = scipy.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), numpy.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
        
        S = numpy.ones((DTE.shape[1]))

        for i in range(0, DTE.shape[1]):
            result = 0
            for j in range(0, DTR.shape[1]):
                result += alpha[j]*Z[j]*rbf_kernel(DTR[:, j], DTE[:, i], g, K**2)
            S[i] = result
        
        return S
    return compute_svm_rbfkernel

def svm_rbf_weighted_classifier_wrapper(K, C, g):
    def compute_svm_rbfkernel(DTR, LTR, DTE, PT):
        Z = LTR * 2 - 1
        Zf = Z

        Z = numpy.reshape(Z, (LTR.shape[0], 1))
        H = numpy.dot(Z, Z.T).astype(float)

        # will compute H in with for loops
        for i in range(0, DTR.shape[1]):
            for j in range(0, DTR.shape[1]):
                H[i][j] *= rbf_kernel(DTR[:, i], DTR[:, j], g, K**2)

        PEMP = LTR.sum()/LTR.shape[0]

        CT = C * (PT / PEMP)
        CF = C * ((1-PT)/(1-PEMP))

        BC = []
        for i in range(0, DTR.shape[1]):
            if Zf[i] == 1:
                BC.append((0, CT))
            else:
                BC.append((0, CF))

        [alpha, f, d2] = scipy.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), numpy.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
        
        S = numpy.ones((DTE.shape[1]))

        for i in range(0, DTE.shape[1]):
            result = 0
            for j in range(0, DTR.shape[1]):
                result += alpha[j]*Z[j]*rbf_kernel(DTR[:, j], DTE[:, i], g, K**2)
            S[i] = result
        
        return S
    return compute_svm_rbfkernel


def svm_classifier_wrapper(K, C):
    
    def compute_svm(DTR, LTR, DTE, PC=0):

        Z = LTR * 2 - 1
        DTRE = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1])) * K])
        D = numpy.multiply(DTRE, Z.T)
        H = numpy.dot(D.T, D)

        # define the array of constraints for the objective
        BC = [(0, C) for i in range(0, DTR.shape[1])]
        [alpha, LD, d] = scipy.optimize.fmin_l_bfgs_b(svm_wraper(H, DTRE), numpy.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
        
        
        # need to compute the primal solution from the dual solution
        w = numpy.multiply(alpha, numpy.multiply(DTRE, Z.T)).sum(axis=1)

        # need to compute the duality gap
        S = -numpy.dot(w.T, D) + 1
        JP = 0.5 * (numpy.linalg.norm(w) ** 2) + C * (S[S>0]).sum()
        
        # we now need to compute the scores and check the predicted lables with threshold
        DTEE = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1])) * K])
        return numpy.dot(w.T, DTEE)
    
    return compute_svm

if __name__ == '__main__':
    # # POLY KERNEL 2
    # print("POLY 2")
    # COST = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    # c_poly = 1
    # DP = compute_pca(D, 7)

    # muP = numpy.mean(DP, axis=1)
    # muP = muP.reshape((7,1))

    # DPC = DP - muP
    # CP = numpy.dot(DPC, DPC.T)
    # CP = CP / DP.shape[1]

    # DZN = z_norm(D)
    # DPZN = z_norm(DP)

    # DALL = whiten(DZN, C)
    # DPALL = whiten(DPZN, CP)

    # DALL = l2(DALL)
    # DPALL = l2(DPALL)

    # plt.figure()
    # plt.xlabel("C")
    # plt.xscale("log")
    # plt.title("SVM, Poly Kernel, d=2 c=1")
    # plt.ylim((0.25, 0.35))
    # plt.grid(visible=True)

    # dcfs = []
    # dcfsZ = []
    # dcfsA = []
    # dcfsP = []    
    # dcfsPZ = []
    # dcfsPA = []

    # for c in COST:
    #     print(c)
    #     result = k_fold(D, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("Niente d=2 c=0.1 --> ", result)
    #     dcfs.append(result)

    #     result = k_fold(DZN, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("ZNorm d=2 c=0.1 --> ", result)
    #     dcfsZ.append(result)

    #     result = k_fold(DALL, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("Znorm, W, L2 d=2 c=0.1 --> ", result)
    #     dcfsA.append(result)

    #     result = k_fold(DP, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, Niente d=2 c=0.1 --> ", result)
    #     dcfsP.append(result)

    #     result = k_fold(DPZN, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, ZNorm d=2 c=0.1 --> ", result)
    #     dcfsPZ.append(result)

    #     result = k_fold(DPALL, L, 10, svm_poly_classifier_wrapper(1, c, 2, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, Znorm, W, L2 d=2 c=0.1 --> ", result)
    #     dcfsPA.append(result)


    # plt.plot(COST, dcfs, label="Poly d=2, c=1, noPCA")
    # plt.plot(COST, dcfsZ, label="Poly d=2, c=1, noPCA, ZNorm")
    # plt.plot(COST, dcfsA, label="Poly d=2, c=1, noPCA, ZNorm+W+L2")
    # plt.plot(COST, dcfsP, label="Poly d=2, c=1, PCA=7")
    # plt.plot(COST, dcfsPZ, label="Poly d=2, c=1, PCA=7, ZNorm")
    # plt.plot(COST, dcfsPA, label="Poly d=2, c=1, PCA=7, ZNorm+W+L2")

    # plt.legend()
    # plt.savefig("../assets/svm/kernel-poly2")
    # plt.close()

    # # POLY KERNEL 3

    # print("POLY 3")
    # COST = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]
    # c_poly = 1
    # DP = compute_pca(D, 7)

    # muP = numpy.mean(DP, axis=1)
    # muP = muP.reshape((7,1))

    # DPC = DP - muP
    # CP = numpy.dot(DPC, DPC.T)
    # CP = CP / DP.shape[1]

    # DZN = z_norm(D)
    # DPZN = z_norm(DP)

    # DALL = whiten(DZN, C)
    # DPALL = whiten(DPZN, CP)

    # DALL = l2(DALL)
    # DPALL = l2(DPALL)

    # plt.figure()
    # plt.xlabel("C")
    # plt.xscale("log")
    # plt.title("SVM, Poly Kernel, d=2 c=1")
    # plt.ylim((0.25, 0.35))
    # plt.grid(visible=True)

    # dcfs = []
    # dcfsZ = []
    # dcfsA = []
    # dcfsP = []    
    # dcfsPZ = []
    # dcfsPA = []

    # for c in COST:
    #     print(c)
    #     result = k_fold(D, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("Niente d=3 c=0.1 --> ", result)
    #     dcfs.append(result)

    #     result = k_fold(DZN, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("ZNorm d=3 c=0.1 --> ", result)
    #     dcfsZ.append(result)

    #     result = k_fold(DALL, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("Znorm, W, L2 d=3 c=0.1 --> ", result)
    #     dcfsA.append(result)

    #     result = k_fold(DP, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, Niente d=3 c=0.1 --> ", result)
    #     dcfsP.append(result)

    #     result = k_fold(DPZN, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, ZNorm d=3 c=0.1 --> ", result)
    #     dcfsPZ.append(result)

    #     result = k_fold(DPALL, L, 10, svm_poly_classifier_wrapper(1, c, 3, c_poly), 0.5, 1, 10)
    #     print("PCA = 7, Znorm, W, L2 d=3 c=0.1 --> ", result)
    #     dcfsPA.append(result)


    # plt.plot(COST, dcfs, label="Poly d=3, c=1, noPCA")
    # plt.plot(COST, dcfsZ, label="Poly d=3, c=1, noPCA, ZNorm")
    # plt.plot(COST, dcfsA, label="Poly d=3, c=1, noPCA, ZNorm+W+L2")
    # plt.plot(COST, dcfsP, label="Poly d=3, c=1, PCA=7")
    # plt.plot(COST, dcfsPZ, label="Poly d=3, c=1, PCA=7, ZNorm")
    # plt.plot(COST, dcfsPA, label="Poly d=3, c=1, PCA=7, ZNorm+W+L2")

    # plt.legend()
    # plt.savefig("../assets/svm/kernel-poly3")
    # plt.close()
    print("Hello")