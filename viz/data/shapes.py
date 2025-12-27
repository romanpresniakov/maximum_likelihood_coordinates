from PIL import Image
import numpy as np


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