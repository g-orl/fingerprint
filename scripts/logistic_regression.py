import scipy
import numpy

def prior_logreg_wrapper(DTR, LTR, l, PC):

    def logreg_derivative_b(v):
        w, b = numpy.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = numpy.exp((-z) * (numpy.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z) / (1 + exp))
        return result / DTR.shape[1]

    def logreg_derivative_w(v):
        w, b = numpy.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = numpy.exp((-z) * (numpy.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z * DTR.T[i]) / (1 + exp))
        return result / DTR.shape[1] + l * w

    def logreg_obj(v):
        # parameter w and b are passed as a single vector
        w, b = numpy.array(v[0:-1]), v[-1]
        nT = LTR.sum()
        nF = DTR.shape[1] - nT

        # need to compute
        result = (l/2)*(numpy.linalg.norm(w)**2)
        sum = 0
        for i in range(0, DTR.shape[1]):
            # for each sample in the dataset
            # we need zi

            z = 2 * LTR[i] -1

            if(z == -1):
                sum += (PC / LTR.sum()) * numpy.logaddexp(0, (numpy.dot(w.T, DTR.T[i]) + b)*(-z))
            else:
                sum += ((1-PC) /  (LTR==0).sum()) * numpy.logaddexp(0, (numpy.dot(w.T, DTR.T[i]) + b)*(-z))


        return (result + sum, numpy.concatenate([logreg_derivative_w(v), [logreg_derivative_b(v)]]))

    return logreg_obj

def logreg_wrapper(DTR, LTR, l, PC):

    def logreg_derivative_b(v):
        w, b = numpy.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = numpy.exp((-z) * (numpy.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z) / (1 + exp))
        return result / DTR.shape[1]

    def logreg_derivative_w(v):
        w, b = numpy.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = numpy.exp((-z) * (numpy.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z * DTR.T[i]) / (1 + exp))
        return result / DTR.shape[1] + l * w

    def logreg_obj(v):
        # parameter w and b are passed as a single vector
        w, b = numpy.array(v[0:-1]), v[-1]
        nT = LTR.sum()
        nF = DTR.shape[1] - nT

        # need to compute
        result = (l/2)*(numpy.linalg.norm(w)**2)
        sum = 0
        for i in range(0, DTR.shape[1]):
            # for each sample in the dataset
            # we need zi

            z = 2 * LTR[i] -1
            sum += numpy.logaddexp(0, (numpy.dot(w.T, DTR.T[i]) + b)*(-z))

        return (result + sum, numpy.concatenate([logreg_derivative_w(v), [logreg_derivative_b(v)]]))

    return logreg_obj

def expand_feature_space(D):
    newD = numpy.ndarray((D.shape[0]**2 + D.shape[0], 1))
    for i in range(0, D.shape[1]):
        x = D[:, i]
        x = x.reshape((D.shape[0], 1))
        
        XXT = numpy.dot(x, x.T)
        phy = numpy.matrix.flatten(XXT)
        
        phy = numpy.concatenate([phy, numpy.matrix.flatten(x)])
        # this is now our x. We now need to put all out xs together
        newD = numpy.concatenate([newD, phy.reshape(D.shape[0]**2 + D.shape[0], 1)], axis=1)
    
    newD = newD[:, 1:]
    return newD

def logreg_classifier_wrapper(l):

    def logreg_classifier(DTR, LTR, DTE, PC=0):
        [x, f, d] = scipy.optimize.fmin_l_bfgs_b(logreg_wrapper(DTR, LTR, l, PC), numpy.zeros((DTR.shape[0]+1)))
        w, b = numpy.array(x[0:-1]), x[-1]

        S = numpy.dot(w, DTE) + b
        # need to subtract the empirical prior so that we get score not based on the prior
        return S - numpy.log((LTR.sum()/(LTR.shape[0] - LTR.sum())))
        # LP = [1 if i > 0 else 0 for i in S]
        # return LP

    return logreg_classifier

def logreg_prior_classifier_wrapper(l):

    def logreg_classifier(DTR, LTR, DTE, PC=0):
        [x, f, d] = scipy.optimize.fmin_l_bfgs_b(prior_logreg_wrapper(DTR, LTR, l, PC), numpy.zeros((DTR.shape[0]+1)))
        w, b = numpy.array(x[0:-1]), x[-1]

        S = numpy.dot(w, DTE) + b
        # need to subtract the empirical prior so that we get score not based on the prior
        return S - numpy.log((LTR.sum()/(LTR.shape[0] - LTR.sum())))
        # LP = [1 if i > 0 else 0 for i in S]
        # return LP

    return logreg_classifier


if __name__ == '__main__':

    # [D, L] = load_data("../data/Train.txt", 10)

    # quadratic logistic regression

    # print("QuadReg")

    # lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    # dims = [7, 8, 9]
    # EXPD = expand_feature_space(D)

    # plt.figure()
    # plt.xlabel("lambda")
    # plt.xscale("log")
    # plt.title("QuadLogReg")
    # plt.ylim((0.20, 0.40))
    # plt.grid(visible=True)

    # print("Quad log reg")
    # print("No PCA")
    # dcfs = []
    # for l in lambdas:
    #     result = k_fold(EXPD, L, 10, logreg_classifier_wrapper(l), 0.5, 1, 10)
    #     dcfs.append(result)
    #     print(l, "--->", result)
    # plt.plot(lambdas, dcfs, label="NO PCA")

    # for m in dims:
    #     dcfs.clear()
    #     print("PCA DIM ", m)
    #     DP = compute_pca(D, m)
    #     DP = expand_feature_space(DP)
    #     for l in lambdas:
    #         result = k_fold(DP, L, 10, logreg_classifier_wrapper(l), 0.5, 1, 10)
    #         dcfs.append(result)
    #         print(l, "--->", result)
    #     plt.plot(lambdas, dcfs, label="PCA="+m.__str__())
    # plt.legend()
    # plt.savefig("assets/lr-lambda/expanded")
    # plt.close()

    # print("QuadReg ALL")
    # lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    # DP = compute_pca(D, 7)
    # m = 7
    # # D = compute_pca(D, 7)
    # mu = numpy.mean(DP, axis=1)
    # mu = mu.reshape((m,1))

    # DCP = DP - mu
    # CP = numpy.dot(DCP, DCP.T)
    # CP = CP / DP.shape[1]

    # DPZN = z_norm(DP)
    # DPALL = whiten(DPZN, CP)
    # DPALL = l2(DPALL)

    # DP = expand_feature_space(DP)
    # DPZN = expand_feature_space(DPZN)
    # DPALL = expand_feature_space(DPALL)

    # plt.figure()
    # plt.xlabel("lambda")
    # plt.xscale("log")
    # plt.title("QuadLogReg")
    # plt.ylim((0.20, 0.40))
    # plt.grid(visible=True)

    # dcfsP = []
    # dcfsPZ = []
    # dcfsPA = []
    # for l in lambdas:
    #     print(l)
    #     result = k_fold(DP, L, 10, logreg_classifier_wrapper(l), 0.5, 1, 10)
    #     print("PCA = 7, QuadLogReg --> ", result)
    #     dcfsP.append(result)

    #     result = k_fold(DPZN, L, 10, logreg_classifier_wrapper(l), 0.5, 1, 10)
    #     print("PCA = 7, Znorm, QuadLogReg --> ", result)
    #     dcfsPZ.append(result)

    #     result = k_fold(DPALL, L, 10, logreg_classifier_wrapper(l), 0.5, 1, 10)
    #     print("PCA = 7, All, QuadLogReg --> ", result)
    #     dcfsPA.append(result)

    # plt.plot(lambdas, dcfsP, label="PCA=7")
    # plt.plot(lambdas, dcfsPZ, label="PCA=7, ZNorm")
    # plt.plot(lambdas, dcfsPA, label="PCA=7, ZNorm, W, L2")
    # plt.legend()
    # plt.savefig("../assets/lr-lambda/expanded-all")
    # plt.close()
    print("Hello")