import polyscope as ps
import triangle
from mlc.core import compute_bmlc, compute_mlc
from data.shapes import *

ps.init()

_, _, texture = get_checkers_board_texture()
vertices, edges = get_snake_shape()

A = dict(vertices=vertices, segments=edges)
# Options: p = PSLG, q = 20 deg minimum angle, a = max triangle area 0.01
tri = triangle.triangulate(A, 'pq20a0.005')
V2 = np.array(tri['vertices'])
V = np.column_stack([tri['vertices'], np.zeros(len(tri['vertices']))])
F = np.array(tri['triangles'], dtype=int)

min_xy = V2.min(axis=0)
max_xy = V2.max(axis=0)

UV = (V2 - min_xy) / (max_xy - min_xy)

W = compute_mlc(np.array(vertices), V2) # basic mlc
print(W.shape)
print(np.isnan(W).sum())
mesh = ps.register_surface_mesh("image", V, F)

# Color each vertex by which cage vertex has the most influence
dominant_indices = np.argmax(W, axis=0)

# Define 12 distinct colors manually
distinct_colors = np.array([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 1.0, 0.0],  # Green
    [0.0, 0.0, 1.0],  # Blue
    [1.0, 1.0, 0.0],  # Yellow
    [1.0, 0.0, 1.0],  # Magenta
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.5, 0.0],  # Orange
    [0.5, 0.0, 1.0],  # Purple
    [0.0, 0.5, 0.0],  # Dark Green
    [0.5, 0.5, 0.0],  # Olive
    [1.0, 0.0, 0.5],  # Pink
    [0.0, 0.5, 0.5],  # Teal
])

colors = distinct_colors[dominant_indices]

mesh.add_color_quantity("dominant_cage_vertex", colors,
                       defined_on="vertices", enabled=True)

mesh.add_parameterization_quantity("uv_coords", UV, defined_on="vertices", enabled=True)
mesh.add_color_quantity("texture", texture, defined_on='texture', param_name='uv_coords', enabled=True)
ps.show()
