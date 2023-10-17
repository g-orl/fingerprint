import numpy
from bayes_cost import *

def evaluate(DTR, LTR, DTE, LTE, classifier, PT, CFN, CFP):
    # train the classifier with DTR and LTR and get scores for DTE
    S = classifier(DTR, LTR, DTE, PT)
    return S, LTE
    normDCF = compute_normalized_dcf(S, LTE, PT, CFN, CFP)
    minDCF = compute_minimum_dcf(S, LTE, PT, CFN, CFP)

    return (minDCF, normDCF)

