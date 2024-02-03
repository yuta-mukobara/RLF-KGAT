# Copyright (c) 2024 Yuta Mukobara

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import numpy as np
import torch
import torch.nn.functional as F


def make_tensor(lprob_n, gold, idx):
    lprob_idx = lprob_n[idx, :]  # b_idx x c
    temp_idx = torch.ones_like(gold[idx])  # b_idx
    if lprob_idx.dim() == 1:
        lprob_idx = torch.unsqueeze(lprob_idx, dim=0)
        temp_idx = torch.unsqueeze(temp_idx, dim=0)
    assert lprob_idx.dim() == 2, lprob_idx
    assert temp_idx.dim() == 1, temp_idx
    return lprob_idx, temp_idx


def compute_loss(prob, gold, comp=None, imb=False, beta=0.9999):
    """
    Args:
        prob: output probabilities: batch_size x num_classes (i.e. 3 in FEVER task)
        gold: gold labels: batch_size
        comp:
            - all: binary cross-entropy like objective
            - sr: s-r
            - srn: s-r and sr-n
        imb: boolean
            if true, class balanced loss
        beta: a hyperparameter for cbl
    Outputs:
        loss_p: loss for positive labels
        loss_n: loss for negative labels
    """

    # LABEL_MAP = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    NUM_SAMPLE = [80035, 29775, 35639]  # the number of samples in clsses
    num_class = prob.shape[1]  # the number of classes
    assert len(NUM_SAMPLE) == num_class, num_class

    # unweighted (uniform)
    weight = torch.ones(num_class, device=prob.device)  # c
    # weight computation
    if imb:
        for i in range(num_class):
            weight[i] = (1 - beta) / (1 - np.power(beta, NUM_SAMPLE[i]))

    # positive label loss: cross-entropy loss
    lprob = torch.log(prob)  # b x c
    loss_p = F.nll_loss(lprob, gold, weight=weight)  # 1

    # initialize loss tensor
    loss_n = torch.zeros_like(loss_p)  # 1
    # negative label loss
    if comp != None:
        # probabiilty for negative labels
        prob_n = 1 - prob  # b x c
        # to obtain finite loss value
        prob_n = torch.clamp(prob_n, 1e-7, 1.0)  # b x c
        lprob_n = torch.log(prob_n)  # b x c

        # initialize loss tensor
        loss_sup = torch.zeros_like(loss_p)  # 1
        # loss for SUPPORTS
        idx_sup = torch.nonzero(gold == 0).squeeze()  # b_sup << b
        if idx_sup.numel() > 0:
            lprob_sup, temp_g = make_tensor(lprob_n, gold, idx_sup)  # b_sup x c, b_sup
            weight_sup = weight[0] * torch.ones_like(weight)  # c
            if comp == "all":
                loss_sup = F.nll_loss(lprob_sup, 1 * temp_g, weight_sup) + F.nll_loss(
                    lprob_sup, 2 * temp_g, weight_sup
                )  # 1
            elif comp in ["sr", "srn"]:
                loss_sup = F.nll_loss(lprob_sup, 1 * temp_g, weight_sup)  # 1

        # initialize loss tensor
        loss_ref = torch.zeros_like(loss_p)  # 1
        # loss for REFUTES
        idx_ref = torch.nonzero(gold == 1).squeeze()  # b_ref << b
        if idx_ref.numel() > 0:
            lprob_ref, temp_g = make_tensor(lprob_n, gold, idx_ref)  # b_ref x c, b_ref
            weight_ref = weight[1] * torch.ones_like(weight)  # c
            if comp == "all":
                loss_ref = F.nll_loss(lprob_ref, 0 * temp_g, weight_ref) + F.nll_loss(
                    lprob_ref, 2 * temp_g, weight_ref
                )  # 1
            elif comp in ["sr", "srn"]:
                loss_sup = F.nll_loss(lprob_ref, 0 * temp_g, weight_ref)  # 1

        # initialize loss tensor
        loss_nei = torch.zeros_like(loss_p)  # 1
        # loss for NotEnoughInformation
        if comp in ["all", "srn"]:
            idx_nei = torch.nonzero(gold == 2).squeeze()  # b_nei << b
            if idx_nei.numel() > 0:
                lprob_nei, temp_g = make_tensor(
                    lprob_n, gold, idx_nei
                )  # b_nei x c, b_nei
                weight_nei = weight[2] * torch.ones_like(weight)  # c
                loss_nei = F.nll_loss(lprob_nei, 0 * temp_g, weight_nei) + F.nll_loss(
                    lprob_nei, 1 * temp_g, weight_nei
                )  # 1

        # gather losses
        loss_n = loss_sup + loss_ref + loss_nei  # 1

    return loss_p, loss_n

