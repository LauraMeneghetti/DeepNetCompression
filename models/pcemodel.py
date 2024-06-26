'''
Module focused on the implementation of the Polunomial Chaos
Expansion Layer (PCE).
'''

import torch
import torch.nn as nn
import scipy.misc
import numpy as np

from torcheval.metrics import R2Score
import torch
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize).double()

    def forward(self, x):
        out = self.linear(x)
        return out

class PCEModel(nn.Module):
    '''
    Class that handles the implementation of the PCE layer as given
    in step 2 of Algorithm 3.1 in the paper:
    - Chunfeng Cui, Kaiqi Zhang, Talgat Daulbaev, Julia Gusak,
    Ivan Oseledets, and Zheng Zhang. "Active Subspace of Neural
    Networks: Structural Analysis and Universal Attacks".
    accepted by SIAM Journal on Mathematics of Data Science (SIMODS)

    :param torch.tensor mean: tensor containing the mean value of all
        elements in the input tensor (output AS layer)
    :param torch.tensor var: tensor containing the variance of all
       	elements in the input tensor (output AS layer)
    :param int d: If is not specified, its value is set to the default
        one, that is 50.
    :param int p: If is not specified, its value is set to the default
        value 2.
    :param torch.device device: object representing the device on
        which a torch.Tensor is or will be allocated. If None, the device
        in use is the cpu.
    '''
    def __init__(self, mean, var, d=50, p=2, device=None):
        super(PCEModel, self).__init__()
        self.d = d
        self.p = p
        self.mean = mean
        self.var = var
        # scipy.special.comb(N,k):The number of combinations of N
        # things taken k at a time.
        self.nbasis = scipy.special.comb(d + p, p).astype(int)
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.oneDbasis = self.NormalBasis().to(self.device)
        self.idxset = indexset(d, 0)
        for i in range(1, p + 1):
            self.idxset = torch.cat((self.idxset, indexset(d, i)), dim=0)

        self.mean = nn.Parameter(self.mean, requires_grad=False)
        self.var = nn.Parameter(self.var, requires_grad=False)
        self.oneDbasis = nn.Parameter(self.oneDbasis.to(self.device), requires_grad=False)
        self.idxset = nn.Parameter(self.idxset, requires_grad=False)

    def NormalBasis(self):
        '''
        Basis Functions for normal distribution
        :return: matrix containing the basis functions for normal
            distribution.
        :rtype: torch.tensor
        '''
        B = torch.zeros([self.p + 1, self.p + 1])
        B[0, 0] = 1  # 0nd order
        if self.p >= 1:
            B[1, 1] = 2  # 1st order
        for i in range(1, self.p):  # i-th order
            B[i + 1, 1:i + 2] = 2 * B[i, :i + 1]
            B[i + 1, :i] -= 2 * i * B[i - 1, :i]
        return B

    def PolyVal(self, x):
        '''
        Functions that handles the evaluation of the basis function at the
        input vector
        :param torch.tensor x: input tensor where we evaluate the functions at
        :return: tensor containing the basis functions evaluated at x
        :rtype: torch.tensor
        '''
        [n, m] = x.shape
        x = x.to(self.device)
        x_pows = torch.zeros((n, m, self.p + 1), dtype=torch.float32, device=self.device)
        for i in range(self.p + 1):
            x_pows[:, :, i] = x**i

        polyval = torch.zeros((n, m, self.p + 1), dtype=torch.float32, device=self.device)
        for ip in range(self.p + 1):
            for i in range(ip + 1):
                if self.oneDbasis[ip, i] != 0:
                    polyval[:, :, ip] += self.oneDbasis[ip, i] * x_pows[:, :, i]
        return polyval.to(self.device)

    def forward(self, x):
        '''
        Function that handles the evaluation of the basis functions
        at input x scaled with mean and variance and creation of the
        related matrix
        :param torch.tensor x: tensor where we compute the basis functions
            (input of that layer, e.g. output reduction layer)
        :return: matrix containing the basis functions evaluated at x
        :rtype: torch.tensor
        '''
        k = len(self.mean)
        assert len(self.var) == k
        for i in range(k):
            x[:, i] = (x[:, i] - self.mean[i]) / self.var[i]

        oneDpolyval = self.PolyVal(x)

        Phi = torch.ones([x.shape[0], self.nbasis],
                         dtype=torch.float32).to(self.device)
        for j in range(k):
            Phi *= oneDpolyval[:, j, self.idxset[:, j]]
        return Phi

    def Training(self, x, y, label):
        '''
        Function that implements the training procedure of the PCEmodel
        :param torch.tensor x: tensor representing the output of the
             reduction layer
        :param torch.tensor y: tensor representing the total output of the
             net
        :param torch.tensor label: tensor representing the labels associated
            to each image in the train dataset
        :return: coefficients of the linear combination of the basis
            functions, coefficient of determination R^2 of the prediction,
            scores for each image
        :rtype: np.ndarray, float, float
        '''
        x = x.to(self.device)
        Phi = self.forward(x)
        Phi = Phi.to(torch.float64)

        device = 'cpu'
        LR = linearRegression(Phi.size()[1], y.size()[1]).to(device)

        # Return the coefficient of determination R^2 of the prediction (float)
        y_PCE = LR(Phi.to(device))
        coeff = LR.linear.weight.float().t()
        metric = R2Score()
        metric.update(y_PCE, y.to(device))
        score_approx = metric.compute()
        score_label = 0
        
        return coeff, score_approx, score_label


    def Inference(self, x, coeff):
        '''
        Inference function
        :param torch.tensor x: input tensor
        :param torch.tensor coeff: coefficient tensor
        :return: inference matrix
        :rtype: torch.tensor
        '''
        Phi = self.forward(x)
        if Phi.shape[1] == coeff.shape[0]:
            y = Phi @ coeff.to(self.device)
        else:
            y = Phi.t() @ coeff.t().to(self.device)
        return y


def indexset(d, p):
    '''
    :param int d
    :param int p
    :return: tensor IdxMat
    :rtype: torch.tensor
    '''
    if d == 1:
        IdxMat = p * torch.ones((1, 1), dtype=torch.int64)
    else:
        for i in range(p + 1):
            Idx_tmp = indexset(d - 1, p - i)
            sz = Idx_tmp.shape[0]
            Idx_tmp = torch.cat((i * torch.ones(
                (sz, 1), dtype=torch.int64), Idx_tmp),
                                dim=1)
            if i == 0:
                IdxMat = Idx_tmp
            else:
                IdxMat = torch.cat((IdxMat, Idx_tmp), dim=0)

    return IdxMat
