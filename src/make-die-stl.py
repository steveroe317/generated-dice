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
        vertices = []
        for i in range(self.resolution):
            vertices.append(self.vertices[i][-1])
        return vertices

    def rotate(self, rotation_matrix):
        self.vertices = np.matmul(self.vertices, rotation_matrix)
        self.faces = np.matmul(self.faces, rotation_matrix)

    def translate(self, vector):
        self.vertices += vector
        self.faces += vector


class Rotation(Enum):
    UP_0 = 0
    UP_90 = 1
    UP_180 = 2
    UP_270 = 3
    DOWN_0 = 4
    DOWN_90 = 5
    DOWN_180 = 6
    DOWN_270 = 7


octant_rotation_matrices = {
    Rotation.UP_0: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    Rotation.UP_90: [
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ],
    Rotation.UP_270: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    Rotation.UP_180: [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    Rotation.DOWN_0: [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ],
    Rotation.DOWN_90: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
    ],
    Rotation.DOWN_270: [
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1],
    ],
    Rotation.DOWN_180: [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
}

octant_shift = np.array([1, 1, 1])


def bridge_arcs(arc0, arc1, flip_normal=False):
    faces = []
    for i in range(len(arc0) - 1):
        if flip_normal:
            faces.append([arc0[i], arc1[i], arc0[i + 1]])
            faces.append([arc0[i + 1], arc1[i], arc1[i + 1]])
        else:
            faces.append([arc0[i], arc0[i + 1], arc1[i]])
            faces.append([arc0[i + 1], arc1[i + 1], arc1[i]])
    return faces


# Generate the die's rounded corners.
sphere_octants = {}
for rotation in octant_rotation_matrices:
    octant = ConcaveSphereOctant(radius=1.0, resolution=10)
    octant.translate(octant_shift)
    octant.rotate(octant_rotation_matrices[rotation])
    sphere_octants[rotation] = octant

# Combine the octant's faces
faces = []
for octant in sphere_octants.values():
    faces.extend(octant.faces)

# Bridge the die's upper corners.
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_0].local_yz_vertices(),
        sphere_octants[Rotation.UP_90].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_90].local_yz_vertices(),
        sphere_octants[Rotation.UP_180].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_180].local_yz_vertices(),
        sphere_octants[Rotation.UP_270].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_270].local_yz_vertices(),
        sphere_octants[Rotation.UP_0].local_xz_vertices(),
    )
)

# Bridge the die's lower corners
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.DOWN_0].local_yz_vertices(),
        sphere_octants[Rotation.DOWN_90].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.DOWN_90].local_yz_vertices(),
        sphere_octants[Rotation.DOWN_180].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.DOWN_180].local_yz_vertices(),
        sphere_octants[Rotation.DOWN_270].local_xz_vertices(),
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.DOWN_270].local_yz_vertices(),
        sphere_octants[Rotation.DOWN_0].local_xz_vertices(),
    )
)

# Bridge the die's upper and lower corners
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_0].local_xy_vertices(),
        sphere_octants[Rotation.DOWN_90].local_xy_vertices()[::-1],
        flip_normal=True,
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_90].local_xy_vertices(),
        sphere_octants[Rotation.DOWN_0].local_xy_vertices()[::-1],
        flip_normal=True,
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_180].local_xy_vertices(),
        sphere_octants[Rotation.DOWN_270].local_xy_vertices()[::-1],
        flip_normal=True,
    )
)
faces.extend(
    bridge_arcs(
        sphere_octants[Rotation.UP_270].local_xy_vertices(),
        sphere_octants[Rotation.DOWN_180].local_xy_vertices()[::-1],
        flip_normal=True,
    )
)

# Create the mesh
octant_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    octant_mesh.vectors[i] = f

# Write the mesh to a file
octant_mesh.save("sphere.stl", mode=Mode.ASCII)
