# This function takes a NxN coefficient matrix and returns a NxN adjacency
# matrix by choosing only the K strongest connections in the similarity graph
# CMat: NxN coefficient matrix
# K: number of strongest edges to keep; if K=0 use all the coefficients
# CKSym: NxN symmetric adjacency matrix


import numpy as np


def BuildAdjacency(CMat, K):
    CMat = CMat.astype(np.float16)
    CKSym = None
    N, _ = CMat.shape
    CAbs = np.absolute(CMat, dtype=np.float16)
    for i in range(0, N):
        c = CAbs[:, i]
        PInd = np.flip(np.argsort(c), 0)
        CAbs[:, i] = CAbs[:, i] / np.absolute(c[PInd[0]]).astype(np.float16)
    CSym = np.add(CAbs, CAbs.T).astype(np.float16)
    if K != 0:
        Ind = np.flip(np.argsort(CSym, axis=0), 0)
        CK = np.zeros([N, N]).astype(np.float16)
        for i in range(0, N):
            for j in range(0, K):
                CK[Ind[j, i], i] = CSym[Ind[j, i], i] / np.absolute(CSym[Ind[0, i], i]).astype(np.float16)
        CKSym = np.add(CK, CK.T)
    else:
        CKSym = CSym
    return CKSym


if __name__ == "__main__":
    pass
