import numpy as np


def basic_mlc_fast(cage: np.ndarray, vertex: np.ndarray):
    """
    Vectorized implementation of Basic Maximum Likelihood Coordinates.

    Args:
        cage: (n, d) vertices of the polytope
        vertex: (d,) point inside the polytope
    """
    n, d = cage.shape

    # Center the cage vertices (v_i = v_i - v)
    # shape(n, d)
    V = cage - vertex

    phi = np.zeros(d)
    eps = 1e-8
    rho = 0.55
    tau = 0.4
    K = 20

    F = -n * np.log(n)
    C = np.full(n, n, dtype=np.float64)

    while True:
        # g = sum( -v_i / c_i )
        gradient = -V.T @ (1.0 / C)
        if np.linalg.norm(gradient) < eps:
            break
        # H = sum( (v_i * v_i.T) / c_i^2 )
        weights = 1.0 / (C ** 2)
        H = (V.T * weights) @ V
        # gradient direction
        q = -np.linalg.solve(H, gradient)

        found_step = False
        for k in range(K + 1):
            step_size = rho ** k
            phi_new = phi + step_size * q

            # c_i = n + phi_new^T v_i
            C_new = n + V @ phi_new

            if np.any(C_new <= 0):
                continue

            F_new = -np.sum(np.log(C_new))

            # Armijo condition check
            descent_check = F + tau * step_size * np.dot(gradient, q)

            if F_new < descent_check:
                phi = phi_new
                F = F_new
                C = C_new  # Update C here so we don't have to recompute it at start of loop
                found_step = True
                break

        if not found_step:
            print("Line search failed to find descent.")
            break

    bmlc = 1.0 / C
    return bmlc, phi


def compute_bmlc(cage: np.ndarray, vertices: np.ndarray):
    """
        Calls a vectorized implementation of Basic Maximum Likelihood Coordinates.

        Args:
            cage: (n, d) vertices of the polytope
            vertices: (m, d) all points inside the polytope
        """
    W = np.zeros(shape=(len(cage), len(vertices)))
    for i in range(len(vertices)):
        weights_i, _ = basic_mlc_fast(cage, vertices[i])
        W[:, i] = weights_i
    return W


def mlc_fast(cage: np.ndarray, vertex: np.ndarray):
    """
    Vectorized implementation of Maximum Likelihood Coordinates.
    Args:
        cage: (n, d) vertices of the polytope
        vertex: (d,) point inside the polytope
    """

    n, d = cage.shape
    V = cage - vertex  # translate by -v
    R = np.linalg.norm(V, axis=1)
    # project
    if np.any(R == 0):
        w = np.zeros(n)
        w[R == 0] = 1.0
        return w

    V_projected = V / R[:, None]
    V_projected = np.vstack([V_projected, V_projected[0]])
    # averaging step 1
    S = V_projected[:-1] + V_projected[1:]
    S_R = np.linalg.norm(S, axis=1) + 1e-8 # small epsilon
    S_averaged = S / S_R[:, None]
    S_averaged = np.vstack([S_averaged[-1], S_averaged])

    # second averaging and projection
    T = S_averaged[:-1] + S_averaged[1:]
    T_R = np.linalg.norm(T, axis=1) + 1e-8 # small epsilon
    T_projected = T / T_R[:, None]
    T_R = np.concatenate([T_R, [T_R[0]]])

    lambda_hat, phi = basic_mlc_fast(T_projected, np.zeros_like(vertex))  # compute bmlc for origin
    lambda_hat = np.concatenate([[lambda_hat[-1]], lambda_hat, [lambda_hat[0]]])
    alpha = 1.0 / (S_R * T_R[:-1])
    beta = 1.0 / (S_R * T_R[1:])
    alpha = np.concatenate([[alpha[-1]], alpha])
    beta = np.concatenate([[beta[-1]], beta])

    wi_hat = alpha[:-1] * lambda_hat[:-2] + (alpha[1:] + beta[:-1]) * lambda_hat[1:-1] + beta[1:] * lambda_hat[2:]
    w = wi_hat / R
    mlc = w / np.sum(w)
    return mlc


def compute_mlc(cage: np.ndarray, vertices: np.ndarray):
    """
    Calls a vectorized implementation of Maximum Likelihood Coordinates.
    :param cage: (n, d) vertices of the polytope
    :param vertices: (m, d) all points inside the polytope
    :return: Barycentric Coordinates
    """
    W = np.zeros(shape=(len(cage), len(vertices)))
    for i in range(len(vertices)):
        weights_i = mlc_fast(cage, vertices[i])
        W[:, i] = weights_i
    return W
