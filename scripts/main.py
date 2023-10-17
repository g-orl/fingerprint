from load import *
from pca import *
from lda import *
from k_fold import *
from mvg import *
from logistic_regression import *
from explore_data import *
from gmm import *
from svm import *
import numpy
import matplotlib
import matplotlib.pyplot as plt
from evaluation import *
from bayes_cost import *

matplotlib.use('TKAgg')

if __name__ == '__main__':
    [DTR, LTR] = load_data("../data/Train.txt", 10)
    [DTE, LTE] = load_data("../data/Test.txt", 10)
    
    m = 7
    DP = compute_pca(DTR, 7)
    mu = numpy.mean(DP, axis=1)
    mu = mu.reshape((m,1))
    DPC = DP - mu
    CDP = numpy.dot(DPC, DPC.T)
    CDP = CDP / DPC.shape[1]

    DPZN = z_norm(DPC)
    DPW = whiten(DPZN, CDP)
    DPALL = l2(DPW)

    
    S, LS = k_fold(DPALL, LTR, 10, svm_poly_classifier_wrapper(1, 1e-1, 2, 1),0.09, 0, 0)
    minDCF = compute_minimum_dcf(S, LS.astype(int), 1/11, 1, 1)
    print(minDCF)
    minDCF = compute_minimum_dcf(S, LS.astype(int), 0.5, 1, 1)
    print(minDCF)
    minDCF = compute_minimum_dcf(S, LS.astype(int), 0.9, 1, 1)
    print(minDCF)
    
 
    
    # print("Train, K0 = 4(tied), K1 = 1 --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(3, 1, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 4(tied), K1 = 1 --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(3, 1, tied0=True), 0.5, 1, 10)[0])
    
    # print("Train, K0 = 8(tied), K1 = 1 --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(4, 1, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 8(tied), K1 = 1 --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(4, 1, tied0=True), 0.5, 1, 10)[0])
    
    # print("Train, K0 = 16(tied), K1 = 1 --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(5, 1, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 16(tied), K1 = 1 --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(5, 1, tied0=True), 0.5, 1, 10)[0])
    
    # print("Train, K0 = 4(tied), K1 = 4(diag) --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(3, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 4(tied), K1 = 4(diag) --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(3, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])
    
    # print("Train, K0 = 8(tied), K1 = 4(diag) --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(4, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 8(tied), K1 = 4(diag) --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(4, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])
    
    # print("Train, K0 = 16(tied), K1 = 4(diag) --> ", k_fold(DTR_P, LTR, 10, gmm_classifier_wrapper(5, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])
    # print("Test, K0 = 16(tied), K1 = 4(diag) --> ", evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(5, 3, diag1=True, tied0=True), 0.5, 1, 10)[0])

    # PREPROCESSING
    # muDTR = numpy.mean(DTR_P, axis=1)
    # muDTR = muDTR.reshape((m,1))
    
    # muDTE = numpy.mean(DTE_P, axis=1)
    # muDTE = muDTE.reshape((m,1))

    # DTRC = DTR_P - muDTR
    # CDTR = numpy.dot(DTRC, DTRC.T)
    # CDTR = CDTR / DTRC.shape[1]
    
    # DTEC = DTE_P - muDTE
    # CDTE = numpy.dot(DTEC, DTEC.T)
    # CDTE = CDTE / DTEC.shape[1]

    # DTR_ZN = z_norm(DTR_P)
    # DTR_W = whiten(DTR_ZN, CDTR)
    # DTR_ALL = l2(DTR_W)

    # DTE_ZN = z_norm(DTE_P)
    # DTE_W = whiten(DTE_ZN, CDTE)
    # DTE_ALL = l2(DTE_W)

    # COST = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    # dcfTR1 = [0.3602459016393443, 0.3493032786885246, 0.3217622950819672, 0.2886885245901639, 0.2655737704918033, 0.2786885245901639, 0.28743852459016395]
    # dcfTR2 = [0.3683196721311476, 0.3661475409836066, 0.3336475409836066, 0.28743852459016395, 0.2521311475409836, 0.2846311475409836, 0.28401639344262297]
    # dcfTR3 = [0.39461065573770493, 0.3721106557377049, 0.3533811475409836, 0.29584016393442625, 0.2589959016393443, 0.27901639344262297, 0.3152049180327869]
    
    # dcfTE1 = [0.32364441930618404, 0.31092571644042233, 0.28009238310708895, 0.2648001508295626, 0.2496530920060332, 0.260237556561086, 0.2725810708898945]
    # dcfTE2 = [0.3247473604826546, 0.3075603318250377, 0.26918552036199095, 0.25183069381598794, 0.2519136500754148, 0.2700810708898944, 0.27488310708898944]
    # dcfTE3 = [0.3150282805429864, 0.31218514328808444, 0.28224736048265464, 0.24771681749622926, 0.2555184766214178, 0.2700188536953243, 0.2859238310708899]

    # for C in COST:
    #     # C_POLY = [0.1, 1, 10]
    #     print("COST --> ", C)
    #     S_TR_pre, L_TR_pre = k_fold(DTR_ALL, LTR, 10, svm_poly_classifier_wrapper(1, C, 2, 0.1), 0.5, 1, 10)
    #     S_TR, L_TR = k_fold(vrow(S_TR_pre), L_TR_pre, 10, mvg_classifier, 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(DTR_ALL, LTR, DTE_ALL, LTE, svm_poly_classifier_wrapper(1, C, 2, 0.1), 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(vrow(S_TR_pre), L_TR_pre, S_TE, L_TE, mvg_classifier, 0.5, 1, 10)
        
    #     result = compute_minimum_dcf(S_TR, L_TR, 0.5, 1, 10)
    #     dcfTR1.append(result)
    #     print("train, c=0.1,", result)
        
    #     result = compute_minimum_dcf(S_TE, L_TE, 0.5, 1, 10)
    #     dcfTE1.append(result)
    #     print("test, c=0.1,", result)
        
    #     S_TR_pre, L_TR_pre = k_fold(DTR_ALL, LTR, 10, svm_poly_classifier_wrapper(1, C, 2, 1), 0.5, 1, 10)
    #     S_TR, L_TR = k_fold(vrow(S_TR_pre), L_TR_pre, 10, mvg_classifier, 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(DTR_ALL, LTR, DTE_ALL, LTE, svm_poly_classifier_wrapper(1, C, 2, 1), 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(vrow(S_TR_pre), L_TR_pre, S_TE, L_TE, mvg_classifier, 0.5, 1, 10)
    #     result = compute_minimum_dcf(S_TR, L_TR, 0.5, 1, 10)
    #     dcfTR2.append(result)
    #     print("train, c=1,", result)

    #     result = compute_minimum_dcf(S_TE, L_TE, 0.5, 1, 10)
    #     dcfTE2.append(result)
    #     print("test, c=1,", result)
        
    #     S_TR_pre, L_TR_pre = k_fold(DTR_ALL, LTR, 10, svm_poly_classifier_wrapper(1, C, 2, 10), 0.5, 1, 10)
    #     S_TR, L_TR = k_fold(vrow(S_TR_pre), L_TR_pre, 10, mvg_classifier, 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(DTR_ALL, LTR, DTE_ALL, LTE, svm_poly_classifier_wrapper(1, C, 2, 10), 0.5, 1, 10)
    #     S_TE, L_TE = evaluate(vrow(S_TR_pre), L_TR_pre, S_TE, L_TE, mvg_classifier, 0.5, 1, 10)
    #     result = compute_minimum_dcf(S_TR, L_TR, 0.5, 1, 10)
    #     dcfTR3.append(result)
    #     print("train, c=10,", result)
        
    #     result = compute_minimum_dcf(S_TE, L_TE, 0.5, 1, 10)
    #     dcfTE3.append(result)
    #     print("test, c=10,", result)
   
    
    # for K0 in non_target:
    #     S, L = evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(K0, 1), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Test, Non target:", 2**(K0 - 1), "target:1 --> ", result)
    #     dcfTE1.append(result)
    #     S, L = k_fold(DTR_P, LTR,10, gmm_classifier_wrapper(K0, 1), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Train, Non target:", 2**(K0 - 1), "target:1 --> ", result)
    #     dcfTR1.append(result)
        
    #     S, L = evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(K0, 2), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Test, Non target:", 2**(K0 - 1), "target:2 --> ", result)
    #     dcfTE2.append(result)
    #     S, L = k_fold(DTR_P, LTR,10, gmm_classifier_wrapper(K0, 2), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Train, Non target:", 2**(K0 - 1), "target:2 --> ", result)
    #     dcfTR2.append(result)
        
    #     S, L = evaluate(DTR_P, LTR, DTE_P, LTE, gmm_classifier_wrapper(K0, 3), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Test, Non target:", 2**(K0 - 1), "target:4 --> ", result)
    #     dcfTE3.append(result)
    #     S, L = k_fold(DTR_P, LTR,10, gmm_classifier_wrapper(K0, 1), 0.5, 1, 10)
    #     result = compute_minimum_dcf(S, L, 0.5, 1, 10)
    #     print("Train, Non target:", 2**(K0 - 1), "target:4 --> ", result)
    #     dcfTR3.append(result)

    # plt.figure()
    # plt.title("GMM")
    # plt.xlabel("K0")
    # plt.ylim((0.2, 0.5))
    # plt.ylabel("minDCF")
    # plt.grid(visible=True)

    # plt.legend()
    # plt.show()
    
    # print("Target = Full Covariance, Non Target = Full Covariance")
    # print(k_fold(D, L, 10, gmm_classifier_wrapper(3, 1), 0.5, 1, 10))



    

    
    
    