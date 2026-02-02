import polyscope as ps
import numpy as np
from mlc.core import compute_bmlc, compute_mlc
from mlc.core3d import MlcMesh3d
import trimesh


def load_obj(filename):
    """Simple OBJ loader for vertices and faces"""
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                # Handle faces (convert to 0-indexed)
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces, dtype=int)


ps.init()

# Load armadillo mesh and cage
V_mesh, F_mesh = load_obj('data/horse/horse.obj')
V_cage, F_cage = load_obj('data/horse/bounding_meshb.obj')

# centroid = np.mean(V_cage, axis=0)
#
# V_cage = (V_cage - centroid) * 1.1 + centroid

print(f"Mesh vertices: {V_mesh.shape}, Faces: {F_mesh.shape}")
print(f"Cage vertices: {V_cage.shape}")

V_cage_deformed, F_cage_deformed = load_obj("data/horse/deformed_bounding_mesh3b.obj")

print("Computing MLC weights...")
mlc_mesh = MlcMesh3d(V_cage, F_cage)
W_mlc = mlc_mesh.compute_all(V_mesh, False)

print(f"MLC shape: {W_mlc.shape}, NaN: {np.isnan(W_mlc).sum()}")

V_mesh_deformed_mlc = W_mlc.T @ V_cage_deformed

mesh_original = ps.register_surface_mesh("Original Mesh", V_mesh, F_mesh)
mesh_mlc = ps.register_surface_mesh("Deformed MLC", V_mesh_deformed_mlc, F_mesh)

edges_original = np.vstack([F_cage[:, [0, 1]], F_cage[:, [1, 2]], F_cage[:, [2, 0]]])
edges_deformed = np.vstack([F_cage_deformed[:, [0, 1]], F_cage_deformed[:, [1, 2]], F_cage_deformed[:, [2, 0]]])

cage_original = ps.register_curve_network("Original Cage", V_cage, edges_original)
cage_deformed = ps.register_curve_network("Deformed Cage", V_cage_deformed, edges_deformed)

bbox_size = V_mesh.max(axis=0) - V_mesh.min(axis=0)
spacing = bbox_size[0] * 1.5

mesh_original.set_transform(np.array([
    [1, 0, 0, -spacing],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

mesh_mlc.set_transform(np.array([
    [1, 0, 0, spacing],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

cage_original.set_transform(np.array([
    [1, 0, 0, -spacing],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

cage_deformed.set_transform(np.array([
    [1, 0, 0, spacing],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

ps.show()