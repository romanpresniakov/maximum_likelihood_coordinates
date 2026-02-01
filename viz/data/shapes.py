from PIL import Image
import numpy as np

# utility class with some test shapes

def get_star_shape(outer_radius=5.0, inner_radius=2.0, num_points=5):
    vertices = []
    edges = []

    total_vertices = num_points * 2
    for i in range(total_vertices):
        angle = (i * np.pi) / num_points
        r = outer_radius if i % 2 == 0 else inner_radius
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        vertices.append([round(x, 4), round(y, 4)])
        edges.append([i, (i + 1) % total_vertices])

    return vertices, edges

def get_star_shape_deformation(star_vertices):
    deformed_vertices = star_vertices.copy()
    deformed_vertices[0] += [-1, -1]#
    deformed_vertices[2] += [-1, -1]
    return deformed_vertices


def get_snake_shape():
    vertices = [
        [0, 0],
        [5, 0],
        [5, 3],
        [2, 3],
        [2, 2],
        [4, 2],
        [4, 1],
        [1, 1],
        [1, 4],
        [5, 4],
        [5, 5],
        [0, 5]
    ]

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 0]
    ]

    return vertices, edges

def get_simple_pentagon_shape():
    vertices = [
        [0, 0],
        [4, 0],
        [4, 4],
        [1, 4],
        [-1, 2]
    ]

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0]
    ]

    return vertices, edges

def get_checkers_board_texture():
    img = Image.open("data/default.jpg").convert("RGB")
    W, H = img.size
    texture = np.array(img)
    return W, H, texture