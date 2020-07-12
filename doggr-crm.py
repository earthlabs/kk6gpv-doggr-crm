import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import fmin_slsqp
import plotly.graph_objects as go
from pymongo import MongoClient
import os
import random


def get_prodinj(wells):
    client = MongoClient(os.environ["MONGODB_CLIENT"])
    db = client.petroleum
    docs = db.doggr.aggregate(
        [{"$unwind": "$prodinj"}, {"$match": {"api": {"$in": wells}}},]
    )
    df = pd.DataFrame()
    for x in docs:
        doc = dict(x)
        df_ = pd.DataFrame(doc["prodinj"])
        df_["api"] = doc["api"]
        df = df.append(df_)

    df.sort_values(by=["api", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.fillna(0, inplace=True)

    for col in [
        "date",
        "oil",
        "water",
        "gas",
        "oilgrav",
        "pcsg",
        "ptbg",
        "btu",
        "steam",
        "water_i",
        "cyclic",
        "gas_i",
        "air",
        "pinjsurf",
    ]:
        if col not in df:
            df[col] = 0
        if col not in ["date", "oilgrav", "pcsg", "ptbg", "btu", "pinjsurf"]:
            df[col] = df[col] / 30.45
    return df


def get_graph_oilgas(api):
    client = MongoClient(os.environ["MONGODB_CLIENT"])
    db = client.petroleum
    docs = db.doggr.find({"api": api})
    for x in docs:
        header = dict(x)
    return header


def get_offsets_oilgas(header, radius):
    client = MongoClient(os.environ["MONGODB_CLIENT"])
    db = client.petroleum
    docs = db.doggr.find(
        {"api": api}, {"api": 1, "latitude": 1, "longitude": 1}
    )
    for x in docs:
        header = dict(x)
    try:
        r = radius / 100
        lat = header["latitude"]
        lon = header["longitude"]
        df = pd.DataFrame(
            list(
                db.doggr.find(
                    {
                        "latitude": {"$gt": lat - r, "$lt": lat + r},
                        "longitude": {"$gt": lon - r, "$lt": lon + r},
                    },
                    {"api": 1, "latitude": 1, "longitude": 1},
                )
            )
        )
        df["dist"] = (
            np.arccos(
                np.sin(lat * np.pi / 180)
                * np.sin(df["latitude"] * np.pi / 180)
                + np.cos(lat * np.pi / 180)
                * np.cos(df["latitude"] * np.pi / 180)
                * np.cos((df["longitude"] * np.pi / 180) - (lon * np.pi / 180))
            )
            * 6371
        )
        df = df[df["dist"] <= radius]
        df.sort_values(by="dist", inplace=True)
        # df = df[:25]
        offsets = df["api"].tolist()
        dists = df["dist"].tolist()

        df_offsets = pd.DataFrame()
        for idx in range(len(df)):
            try:
                df_ = get_prodinj([df["api"].iloc[idx]])
                df_["api"] = df["api"].iloc[idx]
                df_["date"] = pd.to_datetime(df_["date"])
                df_offsets = df_offsets.append(df_)
            except Exception:
                pass

        # df_offsets['api'] = df_offsets['api'].apply(
        #     lambda x: str(np.round(dists[offsets.index(x)], 3))+' mi - '+x)
        df_offsets.sort_values(by="api", inplace=True)
    except Exception:
        pass

    return df, df_offsets


class CRMP(object):
    def __init__(
        self,
        tau_max=50,
        f_max=1,
        p_max=50,
        iter_max=200,
        tr_dist=1000,
        dist_wells=None,
        use_bhp=False,
    ):

        self.dist_wells = dist_wells
        self.tau_max = tau_max
        self.f_max = f_max
        self.p_max = p_max
        self.iter_max = iter_max
        self.tr_dist = tr_dist
        self.use_bhp = use_bhp
        self.fx = 9e9
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
        Qt_pred = self.Qt_hat_dt_sp(params, self.X, self.well_on)
        d_params = np.zeros_like(params)

        Qt = self.X.astype(int)[start_train:end_train, :]
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
            model_params, fx, iters, exit, _ = fmin_slsqp(
                func=self.f_to_opt_sp,
                x0=self.model_params,
                args=(X, y),
                bounds=self.param_bounds_sp(),
                fprime=self.d_f_to_opt_sp,
                # f_eqcons = self.B_constr_eq,
                f_ieqcons=self.B_constr_ineq_dt_sp,
                iprint=0,
                iter=self.iter_max,
                full_output=True,
            )
        else:
            model_params, fx, iters, exit, _ = fmin_slsqp(
                func=self.f_to_opt_sp,
                x0=self.model_params,
                args=(X, y),
                bounds=self.param_bounds_sp(),
                # fprime=self.d_f_to_opt_sp,
                # f_eqcons = self.B_constr_eq,
                f_ieqcons=self.B_constr_ineq_dt_sp,
                iprint=0,
                iter=self.iter_max,
                full_output=True,
            )

        if fx < self.fx:
            self.model_params = model_params
            self.fx = fx
        self.iters = iters
        self.exit = exit

        if self.use_bhp:
            self.A, self.B1, self.B2 = self.param_reshape_sp(self.model_params)
        else:
            self.A, self.B1 = self.param_reshape_sp(self.model_params)
        #         print('model fit is ok')
        return self.model_params

    def predict(self):
        Qt_train_pred = self.Qt_hat_dt_sp(
            params=self.model_params, X=self.X, well_on=self.well_on
        )
        return Qt_train_pred

    def plot_results(self):
        wells = self.wells
        X = self.X
        well_on = self.well_on

        fig_map = go.Figure()
        fig_hist = go.Figure()
        fig_cum = go.Figure()
        fig_mapb = go.Figure()

        x = []
        y = []
        sz = []
        names = []
        for idx, prd in enumerate(wells["prds"].keys()):
            wells["prds"][prd]["tau"] = self.A[idx][0]
            wells["prds"][prd]["fj_sum"] = self.B1[idx].sum()
            wells["prds"][prd]["fit"] = pd.Series(
                self.predict().T[idx],
                index=wells["prds"][prd]["prod"].index[:-1],
            )
            x.append(wells["prds"][prd]["x"])
            y.append(wells["prds"][prd]["y"])
            sz.append(wells["prds"][prd]["fj_sum"] * 20)
            names.append(prd)
            fig_hist.add_trace(
                go.Scatter(
                    x=wells["prds"][prd]["prod"].index,
                    y=wells["prds"][prd]["prod"].values,
                    mode="lines",
                    name=prd,
                )
            )
            fig_hist.add_trace(
                go.Scatter(
                    x=wells["prds"][prd]["fit"].index,
                    y=wells["prds"][prd]["fit"].values,
                    mode="lines",
                    name=prd + "_fit",
                )
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=wells["prds"][prd]["prod"].index,
                    y=wells["prds"][prd]["prod"].cumsum().values,
                    mode="lines",
                    name=prd,
                )
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=wells["prds"][prd]["fit"].index,
                    y=wells["prds"][prd]["fit"].cumsum().values,
                    mode="lines",
                    name=prd + "_fit",
                )
            )
        fig_map.add_trace(
            go.Scatter(
                x=x,
                y=y,
                text=names,
                mode="markers",
                name="producers",
                marker=dict(color="green", size=sz),
            )
        )

        fig_mapb.add_trace(
            go.Scattermapbox(
                lon=x,
                lat=y,
                mode="markers",
                name="producers",
                marker=dict(color="green", size=sz),
            )
        )

        x = []
        y = []
        sz = []
        names = []
        for idx, inj in enumerate(wells["injs"].keys()):
            wells["injs"][inj]["fi_sum"] = self.B1.T[idx].sum()
            for jdx, prd in enumerate(wells["prds"].keys()):
                if self.B1.T[idx][jdx] > 0.8:
                    clr = "#f542d7"
                elif self.B1.T[idx][jdx] > 0.7:
                    clr = "#f5428a"
                elif self.B1.T[idx][jdx] > 0.6:
                    clr = "#f55d42"
                else:
                    clr = "#f5c542"
                if self.B1.T[idx][jdx] > 0.5:
                    fig_map.add_shape(
                        go.layout.Shape(
                            type="line",
                            x0=wells["injs"][inj]["x"],
                            y0=wells["injs"][inj]["y"],
                            x1=wells["prds"][prd]["x"],
                            y1=wells["prds"][prd]["y"],
                            layer="below",
                            line=dict(
                                color=clr, width=self.B1.T[idx][jdx] * 10,
                            ),
                        )
                    )
                    fig_mapb.add_trace(
                        go.Scattermapbox(
                            lon=[
                                wells["injs"][inj]["x"],
                                wells["prds"][prd]["x"],
                            ],
                            lat=[
                                wells["injs"][inj]["y"],
                                wells["prds"][prd]["y"],
                            ],
                            mode="lines",
                            line=dict(
                                color=clr, width=self.B1.T[idx][jdx] * 10
                            ),
                        )
                    )
            fig_map.update_shapes(dict(xref="x", yref="y"))
            wells["injs"][inj]["cons"] = self.cons
            x.append(wells["injs"][inj]["x"])
            y.append(wells["injs"][inj]["y"])
            sz.append(wells["injs"][inj]["fi_sum"] * 20)
            names.append(inj)
        fig_map.add_trace(
            go.Scatter(
                x=x,
                y=y,
                text=names,
                mode="markers",
                name="injectors",
                marker=dict(color="red", size=sz),
            )
        )
        fig_mapb.add_trace(
            go.Scattermapbox(
                lon=x,
                lat=y,
                text=names,
                mode="markers",
                name="injectors",
                marker=dict(color="red", size=sz),
            )
        )

        fig_mapb.update_layout(
            margin={"l": 0, "t": 0, "b": 0, "r": 0},
            mapbox={
                "style": "white-bg",
                "layers": [
                    {
                        "below": "traces",
                        "sourcetype": "raster",
                        "source": [
                            "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                        ],
                    }
                ],
                "center": {
                    "lon": header["longitude"],
                    "lat": header["latitude"],
                },
                "zoom": 17,
            },
        )

        fig_map.show()
        fig_hist.show()
        fig_cum.show()
        fig_mapb.show()

    def save_cons(self, api):
        wells = self.wells
        X = self.X
        well_on = self.well_on

        x = []
        y = []
        sz = []
        names = []
        self.taus = []
        for idx, prd in enumerate(wells["prds"].keys()):
            self.taus.append(
                dict(prd=prd, tau=self.A[idx][0], fj_sum=self.B1[idx].sum())
            )
            wells["prds"][prd]["tau"] = self.A[idx][0]
            wells["prds"][prd]["fj_sum"] = self.B1[idx].sum()
            wells["prds"][prd]["fit"] = pd.Series(
                self.predict().T[idx],
                index=wells["prds"][prd]["prod"].index[:-1],
            )
            x.append(wells["prds"][prd]["x"])
            y.append(wells["prds"][prd]["y"])
            sz.append(wells["prds"][prd]["fj_sum"] * 20)
            names.append(prd)

        x = []
        y = []
        sz = []
        names = []
        self.cons = []
        for idx, inj in enumerate(wells["injs"].keys()):
            if inj == api:
                wells["injs"][inj]["fi_sum"] = self.B1.T[idx].sum()
                for jdx, prd in enumerate(wells["prds"].keys()):
                    self.cons.append(
                        dict(
                            frm=inj,
                            to=prd,
                            gain=self.B1.T[idx][jdx],
                            x0=wells["injs"][inj]["x"],
                            y0=wells["injs"][inj]["y"],
                            xm=np.mean(
                                [
                                    wells["injs"][inj]["x"],
                                    wells["prds"][prd]["x"],
                                ]
                            ),
                            ym=np.mean(
                                [
                                    wells["injs"][inj]["y"],
                                    wells["prds"][prd]["y"],
                                ]
                            ),
                            x1=wells["prds"][prd]["x"],
                            y1=wells["prds"][prd]["y"],
                        )
                    )
            wells["injs"][inj]["cons"] = self.cons

    def write_db(self, api):
        client = MongoClient(os.environ["MONGODB_CLIENT"])
        db = client.petroleum

        message = db.doggr.find_one({"api": api})

        df_cons = pd.DataFrame(self.cons)
        df_cons.index = df_cons.index.astype("str")

        df_taus = pd.DataFrame(self.taus)
        df_taus.index = df_taus.index.astype("str")

        message["crm"] = {}
        message["crm"] = dict(cons=df_cons.to_dict(), taus=df_taus.to_dict())

        db.doggr.replace_one({"api": api}, message, upsert=True)

    def process_well(self, offsets):
        locs = df[["api", "latitude", "longitude"]]

        data_oil = pd.pivot_table(
            offsets,
            values="oil",
            index=["date"],
            columns=["api"],
            aggfunc=np.sum,
        )
        data_wtr = pd.pivot_table(
            offsets,
            values="water",
            index=["date"],
            columns=["api"],
            aggfunc=np.sum,
        )

        data_prod = data_oil + data_wtr
        data_prod = data_prod.replace(0, pd.np.nan)
        data_prod = data_prod.dropna(axis=1, how="all")
        data_prod = data_prod.fillna(0)

        data_prod.index = pd.to_datetime(data_prod.index)
        prds = {}
        for idx, well in enumerate(data_prod.columns):
            prds[well] = dict(
                type="prd",
                x=locs[locs["api"] == well[-8:]]["longitude"].values[0],
                y=locs[locs["api"] == well[-8:]]["latitude"].values[0],
                prod=data_prod[well],
            )

        days_of_work = np.array(
            (data_prod.index - data_prod.first_valid_index()).days
        )

        data_cyc = pd.pivot_table(
            offsets,
            values="cyclic",
            index=["date"],
            columns=["api"],
            aggfunc=np.sum,
        )
        data_stm = pd.pivot_table(
            offsets,
            values="steam",
            index=["date"],
            columns=["api"],
            aggfunc=np.sum,
        )
        data_wtri = pd.pivot_table(
            offsets,
            values="water_i",
            index=["date"],
            columns=["api"],
            aggfunc=np.sum,
        )

        data_inj = data_stm + data_wtri + data_cyc
        data_inj = data_inj.replace(0, pd.np.nan)
        data_inj = data_inj.dropna(axis=1, how="all")
        data_inj = data_inj.fillna(0)

        data_inj.index = pd.to_datetime(data_inj.index)
        injs = {}
        for idx, well in enumerate(data_inj.columns):
            injs[well] = dict(
                type="inj",
                x=locs[locs["api"] == well[-8:]]["longitude"].values[0],
                y=locs[locs["api"] == well[-8:]]["latitude"].values[0],
                inj=data_inj[well],
            )

        self.wells = dict(prds=prds, injs=injs)

        Qt_target = data_prod.values[1:, :]
        Qt = data_prod.values[:-1, :]
        inj = data_inj.values[1:, :]
        time_delta = (days_of_work[1:] - days_of_work[:-1]).reshape(-1, 1)

        self.X = np.concatenate((Qt, time_delta, inj), axis=1)
        self.well_on = np.ones_like(Qt_target) * (Qt_target > 0)
        attempts = 0
        while attempts <= 5:
            self.fit(self.X, Qt_target)
            print(attempts, self.iters, self.fx)
            attempts += 1

    def test_func(self):
        print("ok")


client = MongoClient(os.environ["MONGODB_CLIENT"])

db = client.petroleum

df_ = pd.DataFrame(list(db.doggr.find({"field": "Cymric"}, {"api": 1})))
apis = df_["api"].values

print(len(apis))

random.shuffle(apis)

for api in apis:
    print(str(api) + " started")
    try:
        header = get_graph_oilgas(str(api))
        df, offsets = get_offsets_oilgas(header, 0.1)
        crm = CRMP(tau_max=365 * 3, iter_max=100)
        crm.process_well(offsets)
        crm.save_cons(api)
        crm.write_db(api)
        print(str(api) + " completed")
    except Exception:
        print(str(api) + " failed")
