import numpy as np

def lp_error(true, pred, setting="mean_squared"):
    if setting == "mean_squared":
        return np.linalg.norm(true - pred, ord=2)
    else:
        return max(abs(true - pred))

class utils(object):
    def __init__(self, S, Sref, tol_eps, alpha=1., beta=1 / np.sqrt(2), setting="mean_squared", **config):
        self.tol_eps = float(tol_eps)
        self.S = S
        self.Sref = Sref
        self.alpha = alpha
        self.beta = beta
        self.setting = setting
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

    def greedy_snapshots(self, verbose=True):
        """
        Snapshot-only greedy basis selection.
        """

        D, m = self.S.shape
        self.e_infty_0 = max(np.linalg.norm(self.S, axis=0))
        tol_greedy = self.beta * self.tol_eps * self.e_infty_0
        S_centered = self.S - self.Sref

        # store basis vectors here
        V = np.zeros((D, 0))
        selected = []
        residuals = []

        def project(v, V):
            if V.shape[1] == 0:
                return np.zeros_like(v)
            return V @ (V.T @ v)

        for k in range(m):

            # Compute projection errors for all snapshots
            errors = np.zeros(m)
            for j in range(m):
                if j in selected:
                    errors[j] = -np.inf
                    continue
                r = S_centered[:, j] - project(S_centered[:, j], V)
                errors[j] = np.linalg.norm(r)

            # Get maximum projection error
            max_err = np.max(errors)
            residuals.append(max_err)

            if verbose:
                print(f"Iter {k}, max residual {max_err:.3e}")

            # stopping criterion
            if max_err < tol_greedy:
                if verbose:
                    print("Tolerance reached — stopping.")
                break

            candidates = np.where(errors >= max_err)[0]
            print("candidates:", candidates)

            # assert len(candidates) == 1
            j_star = candidates[0]

            selected.append(j_star)

            # residual vector to be added (Gram–Schmidt)
            v = S_centered[:, j_star] - project(S_centered[:, j_star], V)
            v = v / np.linalg.norm(v)

            V = np.hstack((V, v.reshape(-1, 1)))
        Q = V.T @ (self.S - self.Sref)
        return V, Q, selected, residuals

    def weights(self, index, indices_list, learnt_weights):
        X = 1 - np.sum(learnt_weights)
        w = (index ** (self.alpha) / np.sum(np.array(indices_list) ** (self.alpha))) * X
        return w

    def tol_eps_wise(self, index, indices_list, learnt_weights):

        if self.setting == "mean_squared":
            return (self.tol_eps * np.sqrt(self.weights(index, indices_list,
                                                    learnt_weights)) * np.linalg.norm(self.S, 'fro')) * np.sqrt(
            1 - self.beta ** 2)
        else:
            return self.weights(index, indices_list, learnt_weights) * (1 - self.beta) * self.tol_eps * self.e_infty_0

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

    def find_n_min(self, U, Q, tol_min):
        n = 1
        S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        err = tol_min * np.linalg.norm(self.S, 'fro')
        while np.linalg.norm(self.S - S_approx, 'fro') > err:
            n += 1
            S_approx = self.Sref + U[:, :n] @ Q[:n, :]
        return n
