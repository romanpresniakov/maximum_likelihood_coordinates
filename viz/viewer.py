import polyscope as ps
import triangle
from mlc.core import compute_bmlc, compute_mlc
from data.shapes import *

ps.init()

_, _, texture = get_checkers_board_texture()
vertices, edges = get_snake_shape()

vertices_deformed = vertices.copy()
vertices_deformed[3] += np.array([-0.4, 0.9])
vertices_deformed[4] += np.array([-0.4, 0.6])
vertices_deformed[5] += np.array([-0.2, 0.2])

A = dict(vertices=vertices, segments=edges)
tri = triangle.triangulate(A, 'pq20a0.005')
V2_original = np.array(tri['vertices'])
V_original = np.column_stack([tri['vertices'], np.zeros(len(tri['vertices']))])
F = np.array(tri['triangles'], dtype=int)

W_bmlc = compute_bmlc(np.array(vertices), V2_original)
W_mlc = compute_mlc(np.array(vertices), V2_original, True)

print(f"BMLC shape: {W_bmlc.shape}, NaN: {np.isnan(W_bmlc).sum()}")
print(f"MLC shape: {W_mlc.shape}, NaN: {np.isnan(W_mlc).sum()}")

# V_deformed = W^T @ vertices_deformed
V2_deformed_bmlc = W_bmlc.T @ vertices_deformed
V2_deformed_mlc = W_mlc.T @ vertices_deformed

V_deformed_bmlc = np.column_stack([V2_deformed_bmlc, np.zeros(len(V2_deformed_bmlc))])
V_deformed_mlc = np.column_stack([V2_deformed_mlc, np.zeros(len(V2_deformed_mlc))])

min_xy = V2_original.min(axis=0)
max_xy = V2_original.max(axis=0)
UV = (V2_original - min_xy) / (max_xy - min_xy)

mesh_original = ps.register_surface_mesh("original", V_original, F)
mesh_original.add_parameterization_quantity("uv", UV, defined_on="vertices", enabled=True)
mesh_original.add_color_quantity("texture", texture, defined_on='texture',
                                 param_name='uv', enabled=True)

mesh_mlc = ps.register_surface_mesh("deformed_MLC", V_deformed_mlc, F)
mesh_mlc.add_parameterization_quantity("uv", UV, defined_on="vertices", enabled=True)
mesh_mlc.add_color_quantity("texture", texture, defined_on='texture',
                           param_name='uv', enabled=True)

mesh_original.set_transform(np.array([
    [1, 0, 0, -4.0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

mesh_mlc.set_transform(np.array([
    [1, 0, 0, 4.0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]))

ps.show()