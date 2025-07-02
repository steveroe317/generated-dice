#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum
from stl import mesh, Mode


class ConcaveSphereOctant:
    """Generates a sphere octant mesh."""

    def __init__(self, radius=1.0, resolution=10):
        self.radius = radius
        self.resolution = resolution
        self.vertices = []
        self.faces = []
        self.generate_sphere_octant()

    def generate_sphere_octant(self):
        phi = np.linspace(0, np.pi / 2, self.resolution)
        theta = np.linspace(0, np.pi / 2, self.resolution)
        phi, theta = np.meshgrid(phi, theta)

        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)

        self.vertices = np.stack((x, y, z), axis=2)

        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                self.faces.append(
                    [
                        self.vertices[i][j],
                        self.vertices[i][j + 1],
                        self.vertices[i + 1][j + 1],
                    ]
                )
                self.faces.append(
                    [
                        self.vertices[i][j],
                        self.vertices[i + 1][j + 1],
                        self.vertices[i + 1][j],
                    ]
                )

        self.faces = np.array(self.faces)

    def local_xz_vertices(self):
        """Returns the vertices in the xz plane."""
        vertices = []
        for j in range(self.resolution):
            vertices.append(self.vertices[0][j])
        return vertices

    def local_yz_vertices(self):
        vertices = []
        for j in range(self.resolution):
            vertices.append(self.vertices[-1][j])
        return vertices

    def local_xy_vertices(self):
        """Returns the vertices in the xy plane."""
        verticies = []
        for i in range(self.resolution):
            verticies.append(self.vertices[i][0])
        return verticies

    def rotate(self, rotation_matrix):
        self.vertices = np.matmul(self.vertices, rotation_matrix)
        self.faces = np.matmul(self.faces, rotation_matrix)

    def translate(self, vector):
        self.vertices += vector
        self.faces += vector


class Rotations(Enum):
    UP_0 = 0
    UP_90 = 1
    UP_180 = 2
    UP_270 = 3
    DOWN_0 = 4
    DOWN_90 = 5
    DOWN_180 = 6
    DOWN_270 = 7


octant_rotation_matricies = {
    Rotations.UP_0: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    Rotations.UP_90: [
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ],
    Rotations.UP_270: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    Rotations.UP_180: [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    Rotations.DOWN_0: [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ],
    Rotations.DOWN_90: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
    ],
    Rotations.DOWN_270: [
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1],
    ],
    Rotations.DOWN_180: [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
}

octant_shift = np.array([1, 1, 1])

sphere_octants = []
for rotation_matrix in octant_rotation_matricies.values():
    octant = ConcaveSphereOctant(radius=1.0, resolution=10)
    octant.translate(octant_shift)
    octant.rotate(rotation_matrix)
    sphere_octants.append(octant)

# Combine the octant's faces
faces = []
for octant in sphere_octants:
    faces.extend(octant.faces)

# Create the mesh
octant_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    octant_mesh.vectors[i] = f

# Write the mesh to a file
octant_mesh.save("sphere.stl", mode=Mode.ASCII)
