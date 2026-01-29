import numpy as np
from scipy.sparse import csr_matrix

def compute_biharmonic_distance_2d(cage_vertices, query_point):
    """
    Compute biharmonic distances for 2D following the paper:
    "Biharmonic Distance" by Lipman, Rustamov, Funkhouser (2010)

    Uses the discrete Green's function g_d of the Bi-Laplacian.
    Distance: d_B(i,j)² = g_d(i,i) + g_d(j,j) - 2*g_d(i,j)

    Args:
        cage_vertices: (n, 2) array of polygon vertices (ordered)
        query_point: (2,) query point location

    Returns:
        distances: (n,) array of biharmonic distances from query point to each vertex
    """
    from scipy.linalg import pinv

    n = cage_vertices.shape[0]

    euclidean_dists = np.linalg.norm(cage_vertices - query_point, axis=1)
    nearest_idx = np.argmin(euclidean_dists)

    try:
        row_ind = []
        col_ind = []
        data = []

        areas = np.zeros(n)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            edge_prev = np.linalg.norm(cage_vertices[i] - cage_vertices[prev_i])
            edge_next = np.linalg.norm(cage_vertices[i] - cage_vertices[next_i])

            w_prev = 1.0 / (edge_prev + 1e-10)
            w_next = 1.0 / (edge_next + 1e-10)

            row_ind.extend([i, i])
            col_ind.extend([prev_i, next_i])
            data.extend([-w_prev, -w_next])

            row_ind.append(i)
            col_ind.append(i)
            data.append(w_prev + w_next)

            areas[i] = (edge_prev + edge_next) / 2.0

        L_c = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
        A_inv = csr_matrix(np.diag(1.0 / areas))

        #compute L_c * A^(-1) * L_c
        L2 = L_c @ A_inv @ L_c
        L2_dense = L2.toarray()
        g_d = pinv(L2_dense)
        distances = np.zeros(n)

        for j in range(n):
            d_squared = g_d[nearest_idx, nearest_idx] + g_d[j, j] - 2 * g_d[nearest_idx, j]
            distances[j] = np.sqrt(np.abs(d_squared))

        distances = distances + euclidean_dists[nearest_idx]

    except Exception as e:
        print(f"Biharmonic distance computation failed: {e}")
        return np.ones(n)

    distances = np.maximum(distances, 1e-8)

    return distances


def compute_biharmonic_distance_3d(cage_vertices, cage_faces, query_point):
    """
    Compute proper biharmonic distances for 3D mesh following the paper:
    "Biharmonic Distance" by Lipman, Rustamov, Funkhouser (2010)

    Uses the discrete Green's function g_d of the Bi-Laplacian.
    Distance: d_B(i,j)² = g_d(i,i) + g_d(j,j) - 2*g_d(i,j)

    Args:
        cage_vertices: (n, 3) array of mesh vertices
        cage_faces: (m, 3) array of triangle faces
        query_point: (3,) query point location

    Returns:
        distances: (n,) array of biharmonic distances from query point to each vertex
    """
    from scipy.linalg import pinv

    n = cage_vertices.shape[0]

    euclidean_dists = np.linalg.norm(cage_vertices - query_point, axis=1)
    nearest_idx = np.argmin(euclidean_dists)

    try:
        L_c, A = build_cotangent_laplacian_3d(cage_vertices, cage_faces)

        A_diag = A.diagonal()
        A_inv_diag = 1.0 / (A_diag + 1e-10)
        A_inv = csr_matrix(np.diag(A_inv_diag))

        L2 = L_c @ A_inv @ L_c
        L2_dense = L2.toarray()
        g_d = pinv(L2_dense)
        distances = np.zeros(n)
        for j in range(n):
            d_squared = g_d[nearest_idx, nearest_idx] + g_d[j, j] - 2 * g_d[nearest_idx, j]
            distances[j] = np.sqrt(np.abs(d_squared))

        distances = distances + euclidean_dists[nearest_idx]

    except Exception as e:
        print(f"Biharmonic distance computation failed: {e}")
        return np.ones(n)

    distances = np.maximum(distances, 1e-8)
    return distances


def build_cotangent_laplacian_3d(vertices, faces):
    """
    Build cotangent Laplacian matrix L_c and mass matrix A for a 3D triangle mesh.

    Uses the standard cotangent formula (Eq. 7 from the paper):
    (Δu)_i = (1/A_i) * Σ_{j∈N(i)} (cot α_ij + cot β_ij) * (u_i - u_j)

    Args:
        vertices: (n, 3) array of vertex positions
        faces: (m, 3) array of triangle faces

    Returns:
        L_c: (n, n) sparse cotangent Laplacian (without area normalization)
        A: (n, n) sparse diagonal mass matrix (Voronoi areas)
    """
    from scipy.sparse import csr_matrix

    n = vertices.shape[0]

    I = []
    J = []
    S = []

    voronoi_areas = np.zeros(n)
    for face in faces:
        i0, i1, i2 = face

        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]


        e0 = v1 - v2
        e1 = v2 - v0
        e2 = v0 - v1

        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross)

        cot0 = np.dot(-e2, -e1) / (4.0 * area + 1e-10)
        cot1 = np.dot(-e0, -e2) / (4.0 * area + 1e-10)
        cot2 = np.dot(-e1, -e0) / (4.0 * area + 1e-10)

        I.extend([i0, i1, i0, i1])
        J.extend([i1, i0, i0, i1])
        S.extend([cot2, cot2, -cot2, -cot2])

        I.extend([i1, i2, i1, i2])
        J.extend([i2, i1, i1, i2])
        S.extend([cot0, cot0, -cot0, -cot0])

        I.extend([i2, i0, i2, i0])
        J.extend([i0, i2, i2, i0])
        S.extend([cot1, cot1, -cot1, -cot1])

        voronoi_areas[i0] += area / 3.0
        voronoi_areas[i1] += area / 3.0
        voronoi_areas[i2] += area / 3.0

    L_c = csr_matrix((S, (I, J)), shape=(n, n))
    A = csr_matrix(np.diag(voronoi_areas))
    return L_c, A