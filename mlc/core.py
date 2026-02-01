import numpy as np
from tqdm import tqdm
from math_utils.biharmonic_distances import compute_biharmonic_distance_2d

def basic_mlc_fast(cage: np.ndarray, vertex: np.ndarray, debug=False):
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

    if debug:
        from scipy.spatial import Delaunay
        try:
            delaunay = Delaunay(cage)
            is_inside = delaunay.find_simplex(vertex) >= 0
            print(f"Vertex inside convex hull: {is_inside}")
        except:
            pass

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


def mlc_fast_2d(cage: np.ndarray, vertex: np.ndarray, smlc = False):
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

    distances = None
    if smlc:
        distances = compute_biharmonic_distance_2d(
            cage,
            vertex
        )
        T_projected /= distances[:, None]

    lambda_hat, phi = basic_mlc_fast(T_projected, np.zeros_like(vertex))  # compute bmlc for origin
    if smlc:
        lambda_hat /= distances
    lambda_hat = np.concatenate([[lambda_hat[-1]], lambda_hat, [lambda_hat[0]]])
    alpha = 1.0 / (S_R * T_R[:-1])
    beta = 1.0 / (S_R * T_R[1:])
    alpha = np.concatenate([[alpha[-1]], alpha])
    beta = np.concatenate([[beta[-1]], beta])

    wi_hat = alpha[:-1] * lambda_hat[:-2] + (alpha[1:] + beta[:-1]) * lambda_hat[1:-1] + beta[1:] * lambda_hat[2:]
    w = wi_hat / R
    mlc = w / np.sum(w)
    return mlc

def compute_gradients_bmlc(cage_vertices: np.ndarray, vertex: np.ndarray):
    """
    Implements Algorithm 2: Gradient of basic maximum likelihood coordinates.
    Output: gradients (n, 2) where row i is gradient of lambda_i w.r.t v.
    """
    n, d = cage_vertices.shape
    G_v = -np.eye(d)
    V = cage_vertices - vertex
    lambda_hat, phi = compute_bmlc(cage_vertices, vertex)
    V_i = lambda_hat[:, None] * V
    G_phi = V_i.T @ V_i
    G_v += phi[:, None] @ np.sum(V_i * lambda_hat[:, None], axis=0)[None, :]
    G_phi_inv = np.linalg.inv(G_phi)
    G = np.linalg.matmul(G_v, G_phi_inv)
    GV = (G @ V_i.T).T  # (n, d)
    gradient = lambda_hat[:, None] * (
            lambda_hat[:, None] * phi[None, :] - GV
    )
    return gradient


def get_mlc_intermediates(cage: np.ndarray, vertex: np.ndarray, smlc=False, distances=None):
    """
    Runs the MLC forward pass but returns all intermediate variables
    required for the gradient computation.
    """
    n, d = cage.shape

    V_tilde = cage - vertex
    R = np.linalg.norm(V_tilde, axis=1)

    if np.any(R < 1e-10):
        #return dummy data to avoid crash, though gradient is technically undefined/infinite
        return None

    V_ring = V_tilde / R[:, None]
    V_ring_next = np.roll(V_ring, -1, axis=0)

    S = V_ring + V_ring_next
    S_bar = np.linalg.norm(S, axis=1) + 1e-10
    S_ring = S / S_bar[:, None]

    S_ring_prev = np.roll(S_ring, 1, axis=0)
    T = S_ring_prev + S_ring
    T_bar = np.linalg.norm(T, axis=1) + 1e-10

    V_hat = T / T_bar[:, None]
    V_hat_bmlc = V_hat.copy()
    if smlc and distances is not None:
        V_hat_bmlc /= distances[:, None]

    lambda_hat, phi = basic_mlc_fast(V_hat_bmlc, np.zeros_like(vertex))

    if smlc and distances is not None:
        lambda_hat /= distances

    Alpha = 1.0 / (S_bar * T_bar)
    T_bar_next = np.roll(T_bar, -1, axis=0)
    Beta = 1.0 / (S_bar * T_bar_next)

    L = lambda_hat
    A = Alpha
    B = Beta

    term1 = np.roll(A, 1) * np.roll(L, 1)
    term2 = (A + np.roll(B, 1)) * L
    term3 = B * np.roll(L, -1)

    W_ring = term1 + term2 + term3
    W_vec = W_ring / R
    W_sum = np.sum(W_vec)

    return {
        'n': n,
        'R': R,  # r
        'V_ring': V_ring,  # v_ring (normalized v-v_point)
        'S_bar': S_bar,  # s_bar
        'S_ring': S_ring,  # s_ring
        'T_bar': T_bar,  # t_bar
        'V_hat': V_hat,  # v_hat (input to BMLC)
        'lambda_hat': lambda_hat,  # lambda_hat
        'phi': phi,  # phi (Lagrange multiplier)
        'Alpha': Alpha,  # alpha
        'Beta': Beta,  # beta
        'W_ring': W_ring,  # w_ring (unnormalized weights)
        'W_vec': W_vec,  # w (weights pre-normalization)
        'W_sum': W_sum  # W (total weight)
    }


def gradient_mlc_2d(cage: np.ndarray, vertex: np.ndarray):
    """
    Implements Algorithm 4: Gradient of maximum likelihood coordinates.
    Output: gradients (n, 2) where row i is gradient of lambda_i w.r.t v.
    """

    # 2: Get intermediates
    data = get_mlc_intermediates(cage, vertex)
    if data is None:
        return np.zeros((cage.shape[0], 2))  # Singularity fallback

    n = data['n']
    R = data['R']
    V_ring = data['V_ring']  # (n, 2)
    S_bar = data['S_bar']
    S_ring = data['S_ring']  # (n, 2)
    T_bar = data['T_bar']
    V_hat = data['V_hat']  # (n, 2)
    lambda_hat = data['lambda_hat']
    phi = data['phi']  # (2,)
    Alpha = data['Alpha']
    Beta = data['Beta']
    W_ring = data['W_ring']
    W_vec = data['W_vec']
    W_sum = data['W_sum']

    I2 = np.eye(2)

    grad_v_ring = np.zeros((n, 2, 2))
    for i in range(n):
        term = np.outer(V_ring[i], V_ring[i]) - I2
        grad_v_ring[i] = term / R[i]

    grad_s_ring = np.zeros((n, 2, 2))
    X = np.zeros((n, 2, 2))
    for i in range(n):
        idx_next = (i + 1) % n

        Xi = grad_v_ring[i] + grad_v_ring[idx_next]
        X[i] = Xi
        S_matrix = (I2 - np.outer(S_ring[i], S_ring[i])) / S_bar[i]
        grad_s_ring[i] = Xi @ S_matrix

    grad_v_hat = np.zeros((n, 2, 2))
    Y = np.zeros((n, 2, 2))

    for i in range(n):
        idx_prev = (i - 1 + n) % n
        Yi = grad_s_ring[idx_prev] + grad_s_ring[i]
        Y[i] = Yi
        V_hat_matrix = (I2 - np.outer(V_hat[i], V_hat[i])) / T_bar[i]
        grad_v_hat[i] = Yi @ V_hat_matrix

    G_phi = np.zeros((2, 2))
    grad_phi = np.zeros((2, 2))

    for i in range(n):
        G_phi += (lambda_hat[i] ** 2) * np.outer(V_hat[i], V_hat[i])
        G_v_hat_i = lambda_hat[i] * I2 - (lambda_hat[i] ** 2) * np.outer(phi, V_hat[i])
        grad_phi += grad_v_hat[i] @ G_v_hat_i

    grad_phi = grad_phi @ np.linalg.inv(G_phi)

    grad_lambda_hat = np.zeros((n, 2))
    for i in range(n):
        term1 = grad_phi @ V_hat[i]
        term2 = grad_v_hat[i] @ phi
        grad_lambda_hat[i] = -(lambda_hat[i] ** 2) * (term1 + term2)

    x_aux = np.zeros((n, 2))
    y_aux = np.zeros((n, 2))

    for i in range(n):
        x_aux[i] = (X[i] @ S_ring[i]) / S_bar[i]
        y_aux[i] = (Y[i] @ V_hat[i]) / T_bar[i]

    a_aux = np.zeros((n, 2))
    b_aux = np.zeros((n, 2))

    for i in range(n):
        xi = x_aux[i]
        yi = y_aux[i]

        grad_alpha_i = -Alpha[i] * (xi + yi)
        a_aux[i] = Alpha[i] * grad_lambda_hat[i] + lambda_hat[i] * grad_alpha_i
        idx_prev = (i - 1 + n) % n
        grad_beta_prev = -Beta[idx_prev] * (x_aux[idx_prev] + yi)
        b_aux[i] = Beta[idx_prev] * grad_lambda_hat[i] + lambda_hat[i] * grad_beta_prev

    grad_w_ring = np.zeros((n, 2))
    for i in range(n):
        idx_prev = (i - 1 + n) % n
        idx_next = (i + 1) % n
        grad_w_ring[i] = a_aux[idx_prev] + a_aux[i] + b_aux[i] + b_aux[idx_next]

    grad_w = np.zeros((n, 2))
    grad_W = np.zeros(2)

    for i in range(n):
        numerator = grad_w_ring[i] * R[i] + W_ring[i] * V_ring[i]
        grad_w[i] = numerator / (R[i] ** 2)
        grad_W += grad_w[i]

    grad_final = np.zeros((n, 2))
    for i in range(n):
        numerator = grad_w[i] * W_sum - W_vec[i] * grad_W
        grad_final[i] = numerator / (W_sum ** 2)

    return grad_final


def compute_mlc(cage: np.ndarray, vertices: np.ndarray, smlc = False):
    """
    Calls a vectorized implementation of Maximum Likelihood Coordinates.
    :param cage: (n, d) vertices of the polytope
    :param vertices: (m, d) all points inside the polytope
    :return: Barycentric Coordinates
    """
    W = np.zeros(shape=(len(cage), len(vertices)))
    G = np.zeros(shape=(len(cage), len(vertices), 2))
    for i in tqdm(range(len(vertices))):
        weights_i = mlc_fast_2d(cage, vertices[i], smlc)
        grads = gradient_mlc_2d(cage, vertices[i])
        W[:, i] = weights_i
        G[:, i] = grads
    return W, G
