import numpy as np


def basic_mlc_fast(cage: np.ndarray, vertex: np.ndarray):
    """
    Vectorized implementation of Basic Maximum Likelihood Coordinates.

    Args:
        cage: (n, d) vertices of the polytope
        vertex: (d,) point inside the polytope
    """
    n, d = cage.shape

    #Center the cage vertices (v_i := v_i - v)
    #shape(n, d)
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
    n, d = cage.shape
    V = cage - vertex # translate by -v
    R = np.linalg.norm(V, axis=1)

    eps = 1e-12
    # project
    if np.any(R < eps):
        w = np.zeros(n)
        close_vertices = R < eps
        w[close_vertices] = 1.0 / np.sum(close_vertices)
        return w

    V_projected = V / R[:, None]

    # First averaging
    S = np.zeros((n, d))
    S_R = np.zeros(n)

    for j in range(n):
        j_next = (j + 1) % n
        v_j = V_projected[j]
        v_j_next = V_projected[j_next]

        # Compute arc length α_j
        dot_product = np.clip(np.dot(v_j, v_j_next), -1.0, 1.0)
        alpha_j = np.arccos(dot_product)

        # Handle special cases
        if alpha_j < eps:
            # v̊_j ≈ v̊_{j+1}, use either one
            S[j] = v_j
            S_R[j] = 1.0
        elif alpha_j > np.pi - eps:
            # vectors pointing in exactly opposite directions
            if d == 2:
                S[j] = v_j
                S_R[j] = 1.0
            else:
                # In 3D+ use halfway point
                S[j] = v_j + v_j_next
                S_R[j] = np.linalg.norm(S[j])
                if S_R[j] < eps:
                    S[j] = v_j
                    S_R[j] = 1.0
        else:
            # Normal case: s_j = v̊_j + v̊_{j+1} normalized
            S[j] = v_j + v_j_next
            S_R[j] = np.linalg.norm(S[j])

    S_averaged = S / S_R[:, None]

    # Second averaging with same careful treatment
    T = np.zeros((n, d))
    T_R = np.zeros(n)

    for i in range(n):
        i_prev = (i - 1) % n
        s_i_prev = S_averaged[i_prev]
        s_i = S_averaged[i]

        dot_product = np.clip(np.dot(s_i_prev, s_i), -1.0, 1.0)
        alpha_i = np.arccos(dot_product)

        if alpha_i < eps:
            T[i] = s_i
            T_R[i] = 1.0
        elif alpha_i > np.pi - eps:
            if d == 2:
                T[i] = s_i
                T_R[i] = 1.0
            else:
                T[i] = s_i_prev + s_i
                T_R[i] = np.linalg.norm(T[i])
                if T_R[i] < eps:
                    T[i] = s_i
                    T_R[i] = 1.0
        else:
            T[i] = s_i_prev + s_i
            T_R[i] = np.linalg.norm(T[i])

    T_projected = T / T_R[:, None]

    # BMLC at origin
    lambda_hat, phi = basic_mlc_fast(T_projected, np.zeros(d))

    if np.any(np.isnan(lambda_hat)):
        return np.full(n, np.nan)

    # Compute final weights
    w_tilde = np.zeros(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n

        alpha_i = 1.0 / (S_R[i] * T_R[i])
        alpha_prev = 1.0 / (S_R[i_prev] * T_R[i])
        beta_i = 1.0 / (S_R[i] * T_R[i_next])
        beta_prev = 1.0 / (S_R[i_prev] * T_R[i])

        w_tilde[i] = (alpha_prev * lambda_hat[i_prev] +
                      (alpha_i + beta_prev) * lambda_hat[i] +
                      beta_i * lambda_hat[i_next])

    w = w_tilde / R
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

