import numpy
import sys
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')

def compute_confusion_matrix(k, PL, L):
    CM = numpy.zeros((k,k), int)
    for idx, pl in enumerate(PL):
        CM[pl][L[idx]] += 1
    return CM

def compute_optimal_bayes_decision(LLR, pi_one, Cfn, Cfp):
    # need to compute each LLR element with the threshold
    t = -numpy.log((pi_one*Cfn)/((1-pi_one)*Cfp))

    return (LLR > t).astype(int)

def compute_FPR_FNR(CM):
    if(CM[0][1] == 0):
        FNR = 0
    else:
        FNR = CM[0][1]/(CM[0][1] + CM[1][1])
    
    if(CM[1][0] == 0):
        FPR = 0
    else:
        FPR = CM[1][0]/(CM[1][0] + CM[0][0])
    return (FPR, FNR)

def compute_dcf(LLR, L, pi_one, Cfn, Cfp):
    
    # need to compute the confusion matrix after having assing the class with the optimal cost
    LB = compute_optimal_bayes_decision(LLR, pi_one, Cfn, Cfp)
    CM = compute_confusion_matrix(2, LB, L)
    
    FPR, FNR = compute_FPR_FNR(CM)
    return pi_one*Cfn*FNR + (1-pi_one)*Cfp*FPR

def compute_normalized_dcf(LLR, L, pi_one, Cfn, Cfp):
    DCF = compute_dcf(LLR, L, pi_one, Cfn, Cfp)
    # we will now find the cost for a dummy system
    B_dummy = numpy.min([pi_one*Cfn, (1-pi_one)*Cfp])
    return DCF / B_dummy

def compute_minimum_dcf(LLR, L, pi_one, Cfn, Cfp):
    # need to compute all the scores
    min = 0
    S = LLR
    S = numpy.sort(S)
    S = numpy.insert(S, 0, sys.float_info.min)
    S = numpy.insert(S, S.shape[0], sys.float_info.max)
    B_dummy = numpy.min([pi_one*Cfn, (1-pi_one)*Cfp])
    for score in S:
        # need to compute the confusion matrix
        PL = (LLR > score).astype(int)
        CM = compute_confusion_matrix(2, PL, L)
        FPR, FNR = compute_FPR_FNR(CM)

        DCF = pi_one*Cfn*FNR + (1-pi_one)*Cfp*FPR
        # will now normalize the score
        DCF_norm = DCF / B_dummy
        if(min==0 or DCF_norm < min):
            min = DCF_norm
    return min

def plot_bayes_error(S, L, name):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    prior = 1/(numpy.exp(-effPriorLogOdds) + 1)
    dcf = numpy.zeros(prior.shape)
    dcf_min = numpy.zeros(prior.shape)
    for idx, pi in enumerate(prior):
        dcf[idx] = compute_normalized_dcf(S, L, pi, 1, 1)
        dcf_min[idx] = compute_minimum_dcf(S, L, pi, 1, 1)
    # plt.figure()
    plt.plot(effPriorLogOdds, dcf, label=name+", actDCF")
    plt.plot(effPriorLogOdds, dcf_min, label=name+", minDCF", linestyle="dashed")
    # plt.ylim([0, 1.1])
    # plt.xlim([-3, 3])
    # plt.show() 

def plot_ROC_curve(LLR, L, name):
    S = numpy.array(LLR)# + numpy.log((pi_one*Cfn)/((1-pi_one)*Cfp))
    S = numpy.insert(S, 0, sys.float_info.min)
    S = numpy.insert(S, 0, sys.float_info.max)
    S = numpy.sort(S)
    TPR = numpy.array(S)
    FPR = numpy.array(S)

    for idx, score in enumerate(S):
        # need to compute the confusion matrix
        PL = (LLR > score).astype(int)
        CM = compute_confusion_matrix(2, PL, L)
        fnr = CM[0][1]/(CM[0][1] + CM[1][1])
        fpr = CM[1][0]/(CM[1][0] + CM[0][0])
        
        FPR[idx] = fpr
        TPR[idx] = 1 - fnr

    plt.plot(FPR, TPR, label=name)
