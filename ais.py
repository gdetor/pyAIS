#!/usr/bin/env python
#
# AIS.py is a Python script that implements the Annealed Importance Sampling.
#
# Copyright (C) <2017>  UCI, Georgios Detorakis (gdetor@protonmail.com)
#
# The current Python script is based on the work made by Ruslan Salakhutdinov,
# and the original Matlab code can be found here:
# http://www.utstat.toronto.edu/~rsalakhu/rbm_ais.html
#
# Permission is granted for anyone to copy, use, modify, or distribute this
# program and accompanying programs and documents for any purpose, provided
# this copyright notice is retained and prominently displayed, along with
# a note saying that the original programs are available from our
# web page.
# The programs and documents are distributed without any warranty, express or
# implied.  As the programs were written for research purposes only, they have
# not been tested to the degree that would be advisable in any important
# application.  All use of these programs is entirely at the user's own risk.
import numpy as np
# import matplotlib.pylab as plt


rng = np.random
# rng.seed(10)


class ais(object):
    def __init__(self, W, H, V):
        """ Class constructor.

            Args:
                W (ndarray)     : Visible to Hidden weights of the RBM network
                H (ndarray)     : Hidden units' biases
                V (ndarray)     : Visible units' biases

            Returns:
        """
        self.W = W      # Visible to Hidden weights
        self.H = H      # Hidden biases
        self.V = V      # Visible biases
        self.num_dims, self.num_hids = W.shape  # Dimensions, Hidden units

    def logsum(self, x, dim=0):
        """ Returns the log of sum of logs of x.

            Args:
                x (ndarray)     : The input array
                dim (int)       : The dimension on which sums up

            Returns:
                The log of sum of logs of x.
        """
        x = np.array(x)
        a = x.max(axis=dim) - np.log(np.finfo('d').max) / 2
        return a + np.log(np.exp(x - np.expand_dims(a, 1)).sum(axis=dim))

    def logdiff(self, x, dim=0):
        """ Returns the log of diff of logs of x.

            Args:
                x (ndarray)     : The input array
                dim (int)       : The dimension on which differentiates

            Returns:
                The log of diff of logs of x.
        """
        x = np.array(x)
        a = x.max(axis=dim) - np.log(np.finfo('d').max) / 2
        return a + np.log(np.diff(np.exp(x - a), axis=dim))

    def base_rate(self, data, num_cases=1, num_dims=1, num_batches=1):
        """ Implements the base-rate model.

            Args:
                data (ndarray)      : Original data on which the RBM has been
                                      trained
                num_cases (int)     : Number of iterations
                num_dims (int)      : Dimensions
                num_batches (int)   : Number of batches

            Returns:
                log(p) - log(1 - p)
        """
        count = data.sum(axis=0)
        lp = 5
        prob = ((count + lp * num_batches) /
                (num_cases * num_batches + lp * num_batches))
        return np.log(prob) - np.log(1 - prob)

    def run(self, beta, num_cases=10):
        """ Runs the main AIS algorithm for given betas (distributions) and
            number of iterations (num_cases).

            Args:
                beta (ndarray)   : A partition of [0, 1] on which the
                                   intermediate distributions are centered on
                num_cases (int)  : Number of iterations

            Returns:
                logZ_est (double)   : The log of partition function estimate.
                The +-3 variance of the estimate.
        """
        visbiases_base = 0 * V
        d = (num_cases, 1)
        visbias_base = np.tile(visbiases_base, d)
        hidbias = np.tile(H, d)
        visbias = np.tile(V, d)

        logW = np.zeros((num_cases, 1))
        neg_data = np.tile(1 / (1 + np.exp(-visbiases_base)), d)
        neg_data = neg_data > rng.uniform(0, 1, (num_cases, self.num_dims))
        logW -= ((np.dot(neg_data, visbiases_base.T) +
                  self.num_hids * np.log(2)))

        Wh = np.dot(neg_data, W) + hidbias
        Bv_base = np.dot(neg_data, visbiases_base.T)
        Bv = np.dot(neg_data, V.T)

        tt = 1
        x, y = [], []
        for bb in beta[1:]:
            tt += 1
            expWh = np.exp(bb * Wh)
            logW += ((1 - bb) * Bv_base + bb * Bv +
                     np.expand_dims(np.log(1 + expWh).sum(axis=1), 1))

            pos_hid_probs = expWh / (1 + expWh)
            pos_hid_states = (pos_hid_probs >
                              rng.uniform(0, 1, (num_cases, self.num_hids)))

            neg_data = (1 / (1 + np.exp(-(1 - bb) * visbias_base - bb *
                        (np.dot(pos_hid_states, W.T) + visbias))))
            neg_data = neg_data > rng.uniform(0, 1, (num_cases, self.num_dims))

            x.append(tt / len(beta))
            y.append(np.var(logW))

            Wh = np.dot(neg_data, W) + H
            Bv_base = np.dot(neg_data, visbiases_base.T)
            Bv = np.dot(neg_data, V.T)

            expWh = np.exp(bb * Wh)
            logW -= ((1 - bb) * Bv_base + bb * Bv +
                     np.expand_dims(np.log(1 + expWh).sum(axis=1), 1))

        expWh = np.exp(Wh)
        logW += (np.dot(neg_data, V.T) +
                 np.expand_dims(np.log(1 + expWh).sum(axis=1), 1))

        r_ais = self.logsum(logW) - np.log(num_cases)
        mu = np.mean(logW)
        log_std_ais = (np.log(np.std(np.exp(logW - mu))) + mu -
                       np.log(num_cases) / 2)
        logZ_base = (np.log(1 + np.exp(visbiases_base)).sum() +
                     self.num_hids * np.log(2))
        logZ_est = r_ais + logZ_base
        logZ_est_up = (self.logsum(np.hstack([np.log(3)+log_std_ais, r_ais])) +
                       logZ_base)
        logZ_est_down = (self.logdiff(np.hstack([np.log(3)+log_std_ais, r_ais]))
                         + logZ_base)

        logZ_lat_comp_down = 1
        if np.isreal(logZ_est_down) is False:
            logZ_lat_comp_down = 0

        # plt.plot(np.array(x), np.array(y))

        return logZ_est, logZ_est_down, logZ_est_up, logZ_lat_comp_down


if __name__ == '__main__':
    # An extremely simple test case
    W = np.array([[1, 2, 3], [4, 5, 6]])
    H = np.expand_dims(np.array([7, 8, 9]), 0)
    V = np.expand_dims(np.array([4, 4]), 0)
    beta = np.array([i/10.0 for i in range(10)])
    # beta = np.hstack([np.linspace(0, 0.5, 1000),
    #                   np.linspace(0.5, 0.9, 4000),
    #                   np.linspace(0.9, 1.0, 5000)])

    ais_ = ais(W, H, V)
    p, q, r, s = ais_.run(beta, 1000)
    print(p, q, r, s)
