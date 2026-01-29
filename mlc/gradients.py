import numpy as np

from .core import compute_bmlc, compute_mlc

def compute_gradients_bmlc(cage_vertices: np.ndarray, vertex: np.ndarray):
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

def compute_gradients_mlc(cage_vertices: np.ndarray, vertex: np.ndarray):
    pass

