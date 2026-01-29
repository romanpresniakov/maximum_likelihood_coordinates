import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

from mlc.core import compute_mlc, compute_bmlc
from data.shapes import get_snake_shape

vertices, edges = get_snake_shape()
vertices = np.asarray(vertices)

poly_path = Path(vertices)
res = 600

xmin, ymin = vertices.min(axis=0)
xmax, ymax = vertices.max(axis=0)

xs = np.linspace(xmin, xmax, res)
ys = np.linspace(ymin, ymax, res)

X, Y = np.meshgrid(xs, ys)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

inside = poly_path.contains_points(grid_points)
grid_inside = grid_points[inside]

W = compute_mlc(vertices, grid_inside)
i = 3  # index of cage vertex
lambda_i = W[i]

field = np.full(len(grid_points), np.nan)
field[inside] = lambda_i
field = field.reshape(res, res)

field /= np.nanmax(field)
plt.figure(figsize=(5, 5))

plt.imshow(
    field,
    extent=(xmin, xmax, ymin, ymax),
    origin="lower",
    cmap="viridis",
    interpolation="nearest"
)

plt.plot(
    np.r_[vertices[:, 0], vertices[0, 0]],
    np.r_[vertices[:, 1], vertices[0, 1]],
    "w-", linewidth=1.5
)

plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

