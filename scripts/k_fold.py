import numpy
from bayes_cost import *

def k_fold(D, L, K, classifier, PC, CFN, CFP, seed = 0):

    finalL = []
    finalLLR = []
    numpy.random.seed(seed)
    fold_size = int(D.shape[1]/K) # last partition will be the bigger one because otherwise I have a fold with a small amount of validation
    i = 0
    j = fold_size
    done = False
    idx = numpy.random.permutation(D.shape[1])
    
    while(not done):
        # [i, j] will be used for evaluation
        if(i / fold_size == (K - 1)):
            # last partition
            done = True
            DTE = D[:, idx[i:]]
            LTE = L[idx[i:]]

            DTR = D[:, idx[:i]]
            LTR = L[idx[:i]]
        else:
            DTE = D[:, idx[i:j]]
            LTE = L[idx[i:j]]

            DTR0 = D[:, idx[0:i]]
            DTR1 = D[:, idx[j:]]
            DTR = numpy.concatenate([DTR0, DTR1], axis=1)

            LTR0 = L[idx[:i]]
            LTR1 = L[idx[j:]]
            LTR = numpy.concatenate([LTR0, LTR1])

        i+= fold_size
        j+= fold_size
        
        # predicted labels are returned
        # ADDED PRIOR
        PT = 1/11

        LLR = classifier(DTR, LTR, DTE, PC)

        # # I want to perform DCF so I need a way to save al the predicted labels
        
        finalL = numpy.concatenate([finalL, LTE])
        finalLLR = numpy.concatenate([finalLLR, LLR])
        
    # here we need to compute the DCF
    return (finalLLR, finalL.astype(int))
    normDCF = compute_normalized_dcf(finalLLR, finalL.astype(int), PC, CFN, CFP)
    minDCF = compute_minimum_dcf(finalLLR, finalL.astype(int), PC, CFN, CFP)
    # print("DCF is --> ", DCF)
    # print("normDCF is --> ", normDCF)
    # print("minDCF is --> ", minDCF)
    return (minDCF, normDCF)
