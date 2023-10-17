import numpy
import scipy
import sys

def vrow(v):
    return v.reshape(1, len(v))

def vcol(v):
    return v.reshape(len(v), 1)

def logpdf_GAU_ND(X, mu, C):
    M = C.shape[0] #number of features
    invC = numpy.linalg.inv(C) #C^-1
    logDetC = numpy.linalg.slogdet(C)[1] #log|C|
    

    XC = (X - mu).T # XC has shape (N, M)
    const = -0.5*M*numpy.log(2*numpy.pi)

    logN = const - 0.5*logDetC - 0.5*numpy.multiply(numpy.dot(XC, invC), XC).sum(1)
    return logN


def logpdf_GMM(X, gmm):

    S = numpy.empty(shape=(1, X.shape[1]))

    for x in gmm:
        [w, mu, C] = x
        Sg = logpdf_GAU_ND(X, mu, C)
        # add the weight(prior)
        Sg += numpy.log(w)
        S = numpy.vstack([S, Sg])

    S = S[1:, :]

    logdens = scipy.special.logsumexp(S, axis=0)

    return logdens

def E_step(X, gmm):
    S = numpy.empty(shape=(1, X.shape[1]))

    for x in gmm:
        [w, mu, C] = x
        Sg = logpdf_GAU_ND(X, mu, C)
        # add the weight(prior)
        Sg += numpy.log(w)
        S = numpy.vstack([S, Sg])

    S = S[1:, :]

    marginals = scipy.special.logsumexp(S, axis=0)
    post = S - marginals

    gammas = numpy.exp(post)
    ll = marginals.sum()/X.shape[1]

    return gammas, ll

def M_step(X, gammas, psi=0, diag=False, tied=False):
    triplets = []
    gmm = []
    Zsum = 0
    Csum = numpy.zeros((X.shape[0], X.shape[0]))
    for i in range(gammas.shape[0]):
        ZG, FG, SG = compute_new_params(X, gammas[i, :])
        triplets.append((ZG, FG, SG))
        Zsum += ZG

    for i in range(gammas.shape[0]):
        ZG, FG, SG = triplets[i]
        newMu = vcol(FG/ZG)
        newC = (SG/ZG) - numpy.dot(newMu, newMu.T)
        newW = ZG/Zsum
        Csum += (ZG * newC)

        gmm.append((newW, newMu, newC))

    for i in range(gammas.shape[0]):
        (w, mu, C) = gmm[i]

        if tied == True:
            C = (1/X.shape[1])*Csum

        if diag == True:
            C = numpy.eye(C.shape[0])
        
        U, s, _ = numpy.linalg.svd(C)
        s[s < psi] = psi
        C = numpy.dot(U, vcol(s)*U.T)

        gmm[i] = (w, mu, C)
    
    return gmm
        
def compute_new_params(X, gammas):
    ZG = gammas.sum()
    FG = numpy.multiply(X, gammas).sum(axis=1)
    SG = numpy.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        x = vcol(X[:, i])
        SG += gammas[i] * numpy.dot(x, x.T)
    return (ZG, FG, SG)

def EM(D, gmm, thr=1e-6, psi=0, diag=False, tied=False):
    ll = sys.float_info.min
    stop = False
    while not stop:
        gammas, lln = E_step(D, gmm)
        if(numpy.abs(ll - lln) < thr):
            stop = True
        if(not stop):
            ll = lln
            gmm = M_step(D, gammas, psi, diag, tied)
    return gmm, ll

def splitGMM(gmm, alpha=0.1):
    newGMM = []
    for i in range(len(gmm)):
        (w, mu, C) = gmm[i]

        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        newGMM.append((w/2, mu + d, C))
        newGMM.append((w/2, mu - d, C))

    return newGMM

def LBG(D, gmm=None, thr=1e-6, psi=0, diag=False, tied=False, alpha=0.1):
    # find empirical data from D
    mu = vcol(D.mean(1))
    DC = D - mu
    C = (1/D.shape[1])*numpy.dot(DC, DC.T)

    # for the initial we also need to apply this stp
    if gmm == None:
        U, s, _ = numpy.linalg.svd(C)
        s[s<psi] = psi
        C = numpy.dot(U, vcol(s)*U.T)
        gmm = [(1.0, mu, C)]
    else:
        gmm = splitGMM(gmm, alpha)

    ll = sys.float_info.min
    stop = False
    while not stop:
        gammas, lln = E_step(D, gmm)
        if(numpy.abs(ll - lln) < thr):
            stop = True
        if(not stop):
            ll = lln
            gmm = M_step(D, gammas, psi, diag, tied)
    return gmm, ll

def LBG_wrap(D, gmm=None, n_iter=1, thr=1e-6, psi=0, diag=False, tied=False, alpha=0.1):
    for i in range(n_iter):
        [gmm, ll] = LBG(D, gmm, thr, psi, diag, tied, alpha)

    return gmm, ll

def gmm_classifier_wrapper(Nk0, Nk1, tied0=False, tied1=False, diag0=False, diag1=False):

    def gmm_classifier(DTR, LTR, DTE, PT):
        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]

        gmm0, _ = LBG_wrap(DTR0, n_iter=Nk0, tied=tied0, diag=diag0)
        gmm1, _ = LBG_wrap(DTR1, n_iter=Nk1, tied=tied1, diag=diag1)

        S0 = logpdf_GMM(DTE, gmm0)
        S1 = logpdf_GMM(DTE, gmm1)

        S = S1-S0

        return S
    return gmm_classifier