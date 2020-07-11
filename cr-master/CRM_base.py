from abc import ABCMeta, abstractmethod
import sys
import numpy as np
from scipy import sparse as sp

# import autograd.numpy as anp
# from autograd import grad
# from autograd import elementwise_grad

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# import statsmodels.tsa.stattools as st

# import networkx as nx

# from sklearn import  linear_model
# from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics.pairwise import euclidean_distances

# from sklearn.feature_selection import f_regression, mutual_info_regression

# from sklearn import tree
# from sklearn import ensemble
from sklearn.metrics import r2_score as r2

from scipy.optimize import minimize, fmin_slsqp, check_grad, approx_fprime

# from scipy import stats
# from scipy.stats import norm, skew #for some statistics


import seaborn as sns

from itertools import (
    zip_longest,
    product,
    permutations,
    combinations_with_replacement,
)


class CRMP(object):
    def old_init(
        self,
        tau_max=50,
        f_max=1,
        p_max=50,
        tr_dist=1000,
        liq_rate_data=None,
        inj_rate_data=None,
        press_data=None,
        dist_wells=None,
        time_delta=None,
    ):
        """
		cr

		"""

        self.dist_wells = dist_wells
        # self.Qt = liq_rate_data[:-1, :]
        # self.Qt_target = liq_rate_data[1:, :]
        # self.Inj = inj_rate_data[1:, :]
        # self.Inj_n0 = inj_rate_data[:-1, :]
        if (liq_rate_data and inj_rate_data) is not None:
            self.data = {
                "Qt_target": liq_rate_data[1:, :],
                "Qt": liq_rate_data[:-1, :],
                "Inj": inj_rate_data[1:, :],
                "Inj_n0": inj_rate_data[:-1, :],
            }
            self.num_p = self.data["Qt"].shape[1]
            self.num_i = self.data["Inj"].shape[1]
            self.num_t = self.data["Qt"].shape[0]
            self.data["well_on"] = np.ones_like(self.data["Qt_target"]) * (
                self.data["Qt_target"] > 0
            )
            self.start_train = 0
            self.end_train = self.num_t

        if time_delta is not None:
            self.data["dt"] = time_delta
            self.time_ind = True
        else:
            self.time_ind = False

        if press_data is not None:
            self.data["P"] = press_data[1:, :]
            self.data["P_n0"] = press_data[:-1, :]
            self.data["dP"] = self.data["P"] - self.data["P_n0"]
            self.press_ind = True
        else:
            # self.data['dP'] = np.zeros_like(self.data['Qt'])
            self.press_ind = False

        # self.dt = time_delta

        self.tau_max = tau_max
        self.f_max = f_max
        self.p_max = p_max
        self.tr_dist = tr_dist

    def __init__(
        self,
        tau_max=50,
        f_max=1,
        p_max=50,
        tr_dist=1000,
        dist_wells=None,
        use_bhp=False,
    ):
        """
		cr

		"""

        self.dist_wells = dist_wells
        self.tau_max = tau_max
        self.f_max = f_max
        self.p_max = p_max
        self.tr_dist = tr_dist
        self.use_bhp = use_bhp
        print("model init is ok")

    def init_params_sp(self):
        """
		CRMP model parameters initialization
		:param dist_tr: scalar, treshold distance
		"""
        self.A = np.random.random((self.num_p, 1)) * (100 - 20) + 20

        if self.dist_wells is not None:
            self.dist_wells[self.dist_wells == 0] = 0.1
            self.dist_mask = np.multiply(
                (self.dist_wells < self.tr_dist), (10 < self.dist_wells)
            )
            self.B1 = np.multiply(
                (1 / self.dist_wells)
                / (np.sum(1 / self.dist_wells, axis=0, keepdims=True)),
                self.dist_mask,
            )
        else:
            self.B1 = (
                np.random.random((self.num_p, self.num_i)) * (0.5 - 0.01)
                + 0.01
            )
        self.B1_sp = sp.csr_matrix(self.B1)
        self.b1_ind = self.B1_sp.indices
        self.b1_indp = self.B1_sp.indptr
        self.b1_nnz = self.B1_sp.nnz

        if self.use_bhp:
            self.B2 = np.random.random((self.num_p, 1)) * (50 - 1) + 1
            self.model_params = np.concatenate(
                (
                    self.A.reshape(1, -1),
                    self.B2.reshape(1, -1),
                    self.B1_sp.data.reshape(1, -1),
                ),
                axis=1,
            ).ravel()  # , self.b1_ind, self.b1_indp, self.b1_nnz
            return self.model_params

        else:
            self.model_params = np.concatenate(
                (self.A.reshape(1, -1), self.B1_sp.data.reshape(1, -1)), axis=1
            ).ravel()  # , self.b1_ind, self.b1_indp, self.b1_nnz
            return self.model_params

    def param_reshape_sp(self, params):
        """

		:rtype: object
		"""
        params = params.reshape(1, -1)
        A = params[:, : self.num_p].reshape(-1, 1)
        if self.use_bhp:
            B2 = params[:, self.num_p : self.num_p * 2].reshape(-1, 1)
            B1_sp = params[:, self.num_p * 2 :].reshape(-1)
            B1 = sp.csr_matrix((B1_sp, self.b1_ind, self.b1_indp)).toarray()
            return A, B1, B2
        else:
            B1_sp = params[:, self.num_p :].reshape(-1)
            B1 = sp.csr_matrix((B1_sp, self.b1_ind, self.b1_indp)).toarray()
            return A, B1

    def reconstr_sparse_B1(self, B1):
        """
		reconstruct sparse matrix from dense matrix and ind, indp, nnz
		:param B1: numpy array of size (num_p, num_i), B1 dense matrix
		:return: numpy array of size (1, nnz), sparse matrix B1
		"""
        B1_sp = np.zeros(self.nnz)
        for i in range(self.num_p):
            B1_sp[self.indp[i] : self.indp[i + 1]] = B1[
                i, self.ind[self.indp[i] : self.indp[i + 1]]
            ]
        return B1_sp

    def Qt_hat_dt_sp(self, params, X, well_on):
        """
		Calculates liquid rate at time step t+1 based on following formula:
		Ql(t+1) = A*Ql(t) + B1*Inj(t+1) + B2*dP(t+1)
		:param params: numpy array of size (1, num_params), parameters of CRMP model
		:return: numpy array of size (num_t, num_p), predicted liquid rate
		"""

        Qt = X[:, : self.num_p]
        if self.use_bhp:
            dP = X[:, self.num_p : self.num_p * 2]
            dt = X[:, self.num_p * 2 + 1]
            Inj = X[:, self.num_p * 2 + 1 :]

            A_v, B1, B2_v = self.param_reshape_sp(params)
            A = np.exp(-dt.T / (A_v + 1e-8))
            B2 = np.diagflat(-np.multiply(B2_v, A_v))
            B = np.concatenate((B1, B2), axis=1)
            Inj_cor = np.multiply(Inj, (1 + np.dot((1 - well_on), B1)))
            u = np.concatenate((Inj_cor.T, dP.T), axis=0)
        else:
            dt = X[:, self.num_p + 1]
            Inj = X[:, self.num_p + 1 :]

            A_v, B1 = self.param_reshape_sp(params)
            A = np.exp(-dt.T / (A_v + 1e-8))
            B = B1
            Inj_cor = np.multiply(Inj, (1 + np.dot((1 - well_on), B1)))
            u = Inj_cor.T

        if len(Qt.shape) < 2:
            Qt = np.expand_dims(Qt, axis=0)
            Inj = np.expand_dims(Inj, axis=0)
            if self.use_bhp:
                dP = np.expand_dims(dP, axis=0)
            dt = np.expand_dims(dt, axis=0)

        Qt_pred = np.multiply(
            (
                np.multiply(A, Qt.T)
                + np.multiply((np.ones_like(A) - A), np.dot(B, u))
            ).T,
            well_on,
        )
        # assert(dw.shape == w.shape)
        # mask = (start_prod_ind_CRM == j)
        # Qt_pred[start_prod_ind_CRM, mask] = Qt_target[j,mask]
        return Qt_pred

    def f_to_opt_sp(self, params, X, y):
        """
		target (loss) function for optimization (MSE)
		:param params: numpy array of size (1, num_params), parameters of CRMP model
		:return: scalar, MSE
		"""
        Qt_target = y
        well_on = np.ones_like(y) * (y > 0)
        Qt_pred = self.Qt_hat_dt_sp(params, X, well_on)
        # well_mask = [i for i in range(Qt_target.shape[1])]
        # Qt_pred[start_prod_ind_CRM, well_mask] = Qt_target[start_prod_ind_CRM, well_mask]
        # return mse(np.multiply(Qt_target, well_on), Qt_pred)
        return 1 / 2 * np.average((Qt_target - Qt_pred) ** 2)

    def d_f_to_opt_sp(self, params, start_train=0, end_train=1):
        """
		diff of target (loss) function for optimization (MSE)
		:param params: numpy array of size (1, num_params), parameters of CRMP model
		:return: numpy array of size (1, num_params)
		"""
        Qt_pred = self.Qt_hat_dt_sp(params, start_train=0, end_train=1)
        d_params = np.zeros_like(params)

        Qt = self.Qt[start_train:end_train, :]
        Qt_target = self.Qt_target[start_train:end_train, :]
        Inj = self.Inj[start_train:end_train, :]
        dt = self.dt[start_train:end_train, :]
        well_on = self.well_on[start_train:end_train, :]
        dP = self.well_on[start_train:end_train, :]
        num_t = end_train - start_train + 1
        # num_t, num_p = Qt.shape
        # num_i = Inj.shape[1]
        # B1_sp = params[2 * num_p:]
        A_v, B1, B2_v = self.param_reshape_sp(params)
        # B2_v = np.zeros_like(A_v)
        B2 = np.diagflat(-np.multiply(B2_v, A_v))
        A = np.exp(-dt.T / (A_v + 1e-8))
        B = np.concatenate((B1, B2), axis=1)
        Inj_cor = np.multiply(Inj, (1 + np.dot((1 - well_on), B1)))
        u = np.concatenate((Inj_cor.T, np.zeros_like(dP.T)), axis=0)
        A2 = np.ones_like(A) - A
        Qt_diff = Qt_target - Qt_pred

        # sp_temp = np.ones_like(params[:,2*num_p:]).reshape(-1)
        # B1_mask = sp.csr_matrix((sp_temp, ind, indp), shape = (num_p, num_i))

        d_params[: self.num_p] = (
            -1
            / (num_t * self.num_p)
            * np.sum(
                (
                    Qt_diff
                    * (
                        (A * (dt.T * (A_v + 1e-8) ** (-2)))
                        * (Qt.T - (np.dot(B, u)))
                    ).T
                ).T,
                axis=1,
            )
        )
        d_params[self.num_p : 2 * self.num_p] = 0
        d_B1 = (
            -1
            / (num_t * self.num_p)
            * (
                np.dot(Qt_diff.T * A2, Inj_cor)
                + np.dot(Qt_diff.T * A2 * (1 - well_on).T, Inj)
            )
        )

        # d_B1[B1_mask.todense() == 0] = 0
        # d_B1[B1_mask.todense() == 1] += 1E-8
        # d_B1 = B1_mask.todense()
        d_B1_sp = self.reconstr_sparse_B1(d_B1)

        d_params[2 * self.num_p :] = d_B1_sp.reshape(1, -1)
        return d_params

    def B_constr_ineq_dt_sp(self, params, X, y):
        if self.use_bhp:
            A_v, B1, _ = self.param_reshape_sp(params)
        else:
            A_v, B1 = self.param_reshape_sp(params)
        return (1 - np.sum(B1, axis=0)).ravel()

    def param_bounds_sp(self):
        bounds = []
        if self.use_bhp:
            for i in range(0, self.num_p * self.num_i + 2 * self.num_p):
                if i < self.num_p:
                    bounds.append((1e-8, self.tau_max))
                elif i >= self.num_p and i < 2 * self.num_p:
                    bounds.append((1e-8, self.p_max))
                else:
                    bounds.append((1e-8, self.f_max))
        else:
            for i in range(0, self.num_p * self.num_i + self.num_p):
                if i < self.num_p:
                    bounds.append((1e-8, self.tau_max))
                else:
                    bounds.append((1e-8, self.f_max))

        return bounds

    def fit(self, X=None, y=None, use_fprime=False):

        self.num_t = X.shape[0]
        self.num_p = y.shape[1]
        if self.use_bhp:
            self.num_i = X[:, 2 * self.num_p + 1 :].shape[1]
        else:
            self.num_i = X[:, self.num_p + 1 :].shape[1]

        self.model_params = self.init_params_sp()

        if use_fprime:
            self.model_params = fmin_slsqp(
                func=self.f_to_opt_sp,
                x0=self.model_params,
                args=(X, y),
                bounds=self.param_bounds_sp(),
                fprime=self.d_f_to_opt_sp,
                # f_eqcons = self.B_constr_eq,
                f_ieqcons=self.B_constr_ineq_dt_sp,
                iprint=2,
                iter=100,
            )
        else:
            self.model_params = fmin_slsqp(
                func=self.f_to_opt_sp,
                x0=self.model_params,
                args=(X, y),
                bounds=self.param_bounds_sp(),
                # fprime=self.d_f_to_opt_sp,
                # f_eqcons = self.B_constr_eq,
                f_ieqcons=self.B_constr_ineq_dt_sp,
                iprint=2,
                iter=100,
            )

        if self.use_bhp:
            self.A, self.B1, self.B2 = self.param_reshape_sp(self.model_params)
        else:
            self.A, self.B1 = self.param_reshape_sp(self.model_params)
        print("model fit is ok")
        return self.model_params

    def predict(self, X, well_on):
        Qt_train_pred = self.Qt_hat_dt_sp(
            params=self.model_params, X=X, well_on=well_on
        )

        return Qt_train_pred

    def test_func():
        print("ok")
