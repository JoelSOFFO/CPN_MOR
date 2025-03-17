import numpy as np

def l2_error(true, pred):
    return np.linalg.norm(true - pred, ord=2)

class utils(object):
    def __init__(self, S, Sref, tol_eps=1e-3, alpha=1., beta=1 / np.sqrt(2), **config):
        self.tol_eps = float(tol_eps)
        self.S = S
        self.Sref = Sref
        self.alpha = alpha
        self.beta = beta
        self.config = config
        assert self.tol_eps < 1

    def truncate_svd(self, U):

        """
        Truncation of the left singular vectors matrix, to keep the N singular vectors needed to reach target precision.

        Args:
            U: Matrix containing the left singular vectors.

        Returns:
            - The truncated singular vectors matrix (basis of X_N)
            - The projected coordinates on the space X_N

        """
        ## Start at the end
        Q = U.T @ (self.S - self.Sref)
        r = Q.shape[0] - 1
        tol_check = (np.linalg.norm(self.S, 'fro') * (self.tol_eps * self.beta)) ** 2
        Q_norm = np.linalg.norm(Q[r, :], ord=2) ** 2
        while Q_norm <= tol_check:
            r -= 1
            Q_norm += np.linalg.norm(Q[r, :], ord=2) ** 2
        return U[:, :r + 1], Q[:r + 1, :]

    def weights(self, index, indices_list, learnt_weights):
        X = 1 - np.sum(learnt_weights)
        w = (index ** (self.alpha) / np.sum(np.array(indices_list) ** (self.alpha))) * X
        return w

    def tol_eps_wise(self, index, indices_list, learnt_weights):

        return (self.tol_eps * np.sqrt(self.weights(index, indices_list,
                                                    learnt_weights)) * np.linalg.norm(self.S, 'fro')) * np.sqrt(
            1 - self.beta ** 2)

    @staticmethod
    def norm_i(b, index_n, gamma_list):

        if len(b.shape) == 1:
            return np.max(np.linalg.norm(b[index_n], ord=2), (1 / gamma_list) * np.abs(np.delete(b, index_n, axis=0)))
        elif len(b.shape) == 2:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0),
                                  np.max((1 / gamma_list)[:, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)),
                                         axis=0))
        elif len(b.shape) == 3:
            if len(gamma_list) == 0:
                return np.linalg.norm(b[index_n], ord=2, axis=0)
            else:
                return np.maximum(np.linalg.norm(b[index_n], ord=2, axis=0), np.max(
                    (1 / gamma_list)[:, np.newaxis, np.newaxis] * np.abs(np.delete(b, index_n, axis=0)), axis=0))

    def lip_norm_i(self, X, Y, index_n, gamma_list):

        diff_1 = X[index_n, :, np.newaxis] - X[index_n, np.newaxis, :]
        diff_2 = np.delete(X[:, :, np.newaxis], index_n, axis=0) - np.delete(X[:, np.newaxis, :], index_n, axis=0)
        diff = np.concatenate((diff_1, diff_2), axis=0)
        norm_diff = self.norm_i(diff, index_n, gamma_list)
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / self.norm_i(X - X_close, index_n, gamma_list)
        return np.max(ratio)

    @staticmethod
    def lip_norm_2(X, Y):

        diff = X[:, :, np.newaxis] - X[:, np.newaxis, :]
        norm_diff = np.linalg.norm(diff, ord=2, axis=0)  ####norme i
        norm_diff = np.where(norm_diff == 0, np.inf, norm_diff)
        closest_neighbor = np.argmin(norm_diff, axis=1)
        X_close = X[:, closest_neighbor]
        Y_close = Y[closest_neighbor]

        ratio = np.abs(Y - Y_close) / np.linalg.norm(X - X_close, axis=0)
        return np.max(ratio)

    def gamma_i(self, index, indices_list, learnt_weights, Gamma):
        gamma = np.sqrt(self.weights(index, indices_list, learnt_weights) * (Gamma ** 2 - 1))
        return gamma

    def n_min(self, U, Q, tol_min):
        n = 1
        S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        err = tol_min * np.linalg.norm(self.S, 'fro')
        while np.linalg.norm(self.S - S_approx, 'fro') > err:
            n += 1
            S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        return n

    def coeff_approximation(self, Q, func, index_r):
        Qbar = np.zeros((len(func.items()), Q.shape[1]))
        list_keys = list(func.keys())
        Q_total = np.zeros((len(index_r) + len(func.items()), Q.shape[1]))
        Q_total[index_r, :] = Q
        for i, coef_name in enumerate(list_keys):
            nb_deps = func[coef_name]['nb_deps']
            Qbar[i, :] = func[coef_name]['function'](Q_total.T[:, :nb_deps])
            Q_total[func[coef_name]['index']] = Qbar[i, :]
        return Qbar
