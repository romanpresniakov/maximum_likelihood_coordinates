import numpy as np

def build_jacobian(deformed_vertices: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Build a jacobian matrix from deformed vertices and gradients
    :param deformed_vertices: deformed cage vertices
    :param gradients: pre-computed gradients
    :return:
    """
    jacobian = np.einsum('nj, nmk -> mjk', deformed_vertices, gradients)
    return jacobian

def calculate_area_distortion(deformed_vertices: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Calculate area distortion
    :param deformed_vertices: deformed cage vertices
    :param gradients: pre-computed gradients
    :return: area distortion computed as a determinant of Jacobian
    """
    jacobian = build_jacobian(deformed_vertices, gradients)
    return np.linalg.det(jacobian)

def calculate_angle_distortion(deformed_vertices: np.ndarray, gradients: np.ndarray) -> np.ndarray:
    """
    Calculate angle distortion
    :param deformed_vertices: deformed cage vertices
    :param gradients: pre-computed gradients
    :return: angle distortion computed as MIPS-Energy
    """
    jacobian = build_jacobian(deformed_vertices, gradients)
    frobenius_sq = np.sum(jacobian ** 2, axis=(1, 2))
    det_safe = np.maximum(np.linalg.det(jacobian), 1e-6)
    mips_energies = frobenius_sq / det_safe
    return mips_energies