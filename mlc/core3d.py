import numpy as np

from math_utils.biharmonic_distances import compute_biharmonic_distance_3d
from .core import basic_mlc_fast


class MlcMesh3d:
    """
    This class represents a Mesh for 3D-MLC computation.
    """
    def __init__(self, cage_vertices: np.ndarray, cage_faces: np.ndarray):
        self.cage_vertices = cage_vertices.astype(np.float64)
        self.cage_faces = cage_faces
        self.num_vertices = cage_vertices.shape[0]
        self.num_faces = cage_faces.shape[0]

        self.vertex_to_faces = [[] for _ in range(self.num_vertices)]
        for f_idx, face in enumerate(self.cage_faces):
            for v_idx in face:
                self.vertex_to_faces[v_idx].append(f_idx)

        self.vertex_degrees = np.array([len(f) for f in self.vertex_to_faces], dtype=np.float64)


    def compute_weights_mlc_3d(self, query_point, eps=1e-8, smlc = False):
        """
        Vectorized implementation of Maximum Likelihood Coordinates.
        :param query_point: Vertex
        :param eps: Numerical Stability Parameter
        :param smlc: If true, scaling with biharmonic distances is applied
        :return: MLC
        """
        nV = self.cage_vertices.shape[0]
        nF = self.cage_faces.shape[0]

        V = self.cage_vertices - query_point  # (nV, 3)
        R = np.linalg.norm(V, axis=1)

        #handle vertex hit
        if np.any(R < 1e-8):
            w = np.zeros(nV)
            w[np.argmin(R)] = 1.0
            return w

        V0 = V / R[:, None]  # (nV, 3)

        f = self.cage_faces
        v1 = V0[f[:, 0]]
        v2 = V0[f[:, 1]]
        v3 = V0[f[:, 2]]

        n12 = np.cross(v1, v2)
        n23 = np.cross(v2, v3)
        n31 = np.cross(v3, v1)

        n12 /= np.linalg.norm(n12, axis=1, keepdims=True) + eps
        n23 /= np.linalg.norm(n23, axis=1, keepdims=True) + eps
        n31 /= np.linalg.norm(n31, axis=1, keepdims=True) + eps

        b12 = np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1, 1))
        b23 = np.arccos(np.clip(np.sum(v2 * v3, axis=1), -1, 1))
        b31 = np.arccos(np.clip(np.sum(v3 * v1, axis=1), -1, 1))
        # S
        sig0 = (b23 + b12 * np.sum(n12*n23, axis=1) + b31 * np.sum(n31*n23, axis=1)) * np.sin(b23)
        sig1 = (b31 + b23 * np.sum(n23*n31, axis=1) + b12 * np.sum(n12*n31, axis=1)) * np.sin(b31)
        sig2 = (b12 + b31 * np.sum(n31*n12, axis=1) + b23 * np.sum(n23*n12, axis=1)) * np.sin(b12)
        S = sig0[:, None] * v1 + sig1[:, None] * v2 + sig2[:, None] * v3

        S_norm = np.linalg.norm(S, axis=1) + eps
        S_hat = S / S_norm[:, None]

        T = np.zeros((nV, 3))
        face_count = np.zeros(nV)

        for j in range(nF):
            i0, i1, i2 = f[j]
            T[i0] += S_hat[j]
            T[i1] += S_hat[j]
            T[i2] += S_hat[j]
            face_count[[i0, i1, i2]] += 1.0

        T /= face_count[:, None]
        T_norm = np.linalg.norm(T, axis=1) + eps
        V_hat = T / T_norm[:, None]
        distances = None
        if smlc:
            distances = compute_biharmonic_distance_3d(
                self.cage_vertices,
                self.cage_faces,
                query_point
            )
            V_hat /= distances[:, None]
        lambda_hat, _ = basic_mlc_fast(V_hat, np.zeros(3))
        if smlc:
            lambda_hat /= distances
        x = lambda_hat / T_norm

        # A2
        y = np.zeros(nF)
        for j in range(nF):
            i0, i1, i2 = f[j]
            y[j] = x[i0] + x[i1] + x[i2]

        # P1
        y /= S_norm

        # A1
        w = np.zeros(nV)
        for j in range(nF):
            i0, i1, i2 = f[j]
            w[i0] += sig0[j] * y[j]
            w[i1] += sig1[j] * y[j]
            w[i2] += sig2[j] * y[j]

        # P0
        w /= R
        w_sum = np.sum(w)
        return w / w_sum

    def compute_all(self, points, smlc = False):
        """
        Calls a vectorized implementation of Maximum Likelihood Coordinates.
        :param points: (n, d) vertices of the polytope
        :param smlc: if True, scaling with biharmonic distances is applied
        :return: Barycentric Coordinates
        """
        res = []
        for p in points:
            res.append(self.compute_weights_mlc_3d(p, smlc))
        print('Max value: {}'.format(np.max(res)))
        print('Min value: {}'.format(np.min(res)))
        return np.array(res).T
