#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum
from stl import mesh, Mode


# Dice size parameters in millimeters
DICE_SIZE = 16.0
CORNER_RADIUS = 1.0
DOT_RADIUS = 1.6


class SphereOctant:
    """Generates a sphere octant mesh."""

    def __init__(self, radius=1.0, resolution=10, flip_normals=False):
        self.radius = radius
        self.resolution = resolution
        self.vertices = []
        self.faces = []
        self.generate_sphere_octant(flip_normals)

    def generate_sphere_octant(self, flip_normals):
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

        if flip_normals:
            self.faces = [[face[0], face[2], face[1]] for face in self.faces]
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


class CornerRotation(Enum):
    UP_0 = 0
    UP_90 = 1
    UP_180 = 2
    UP_270 = 3
    DOWN_0 = 4
    DOWN_90 = 5
    DOWN_180 = 6
    DOWN_270 = 7


corner_rotation_matrices = {
    CornerRotation.UP_0: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    CornerRotation.UP_90: [
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ],
    CornerRotation.UP_270: [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ],
    CornerRotation.UP_180: [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    CornerRotation.DOWN_0: [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ],
    CornerRotation.DOWN_90: [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1],
    ],
    CornerRotation.DOWN_270: [
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1],
    ],
    CornerRotation.DOWN_180: [
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ],
}


class FaceRotation(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    FRONT = 4
    BACK = 5


face_rotation_matrices = {
    FaceRotation.TOP: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    FaceRotation.BOTTOM: [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ],
    FaceRotation.FRONT: [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ],
    FaceRotation.BACK: [
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ],
    FaceRotation.LEFT: [
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ],
    FaceRotation.RIGHT: [
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0],
    ],
}


def bridge_arcs(arc0, arc1, flip_normals=False):
    faces = []
    for i in range(len(arc0) - 1):
        faces.append([arc0[i], arc0[i + 1], arc1[i]])
        faces.append([arc0[i + 1], arc1[i + 1], arc1[i]])

    if flip_normals:
        faces = [[face[0], face[2], face[1]] for face in faces]

    return faces


def bridge_fan(base, arc, flip_normals=False):
    faces = []
    for i in range(len(arc) - 1):
        faces.append([base, arc[i], arc[i + 1]])

    if flip_normals:
        faces = [[face[0], face[2], face[1]] for face in faces]

    return faces


def GenerateCornerOctants(dice_size, corner_radius):
    # Each ConcaveSphereOctant represents a rounded corner of the die (a cube),
    # and the different rotations correspond to the eight corners of the cube,
    # ensuring all corners are covered with the correct orientation.

    sphere_octants = {}
    for rotation in corner_rotation_matrices:
        octant = SphereOctant(radius=corner_radius, resolution=10)
        octant.translate(np.full([3], dice_size / 2 - corner_radius))
        octant.rotate(corner_rotation_matrices[rotation])
        sphere_octants[rotation] = octant

    return sphere_octants


def bridge_upper_corners(corners):
    faces = []

    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_0].local_yz_vertices(),
            corners[CornerRotation.UP_90].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_90].local_yz_vertices(),
            corners[CornerRotation.UP_180].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_180].local_yz_vertices(),
            corners[CornerRotation.UP_270].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_270].local_yz_vertices(),
            corners[CornerRotation.UP_0].local_xz_vertices(),
        )
    )
    return faces


def bridge_lower_corners(corners):
    faces = []

    faces.extend(
        bridge_arcs(
            corners[CornerRotation.DOWN_0].local_yz_vertices(),
            corners[CornerRotation.DOWN_90].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.DOWN_90].local_yz_vertices(),
            corners[CornerRotation.DOWN_180].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.DOWN_180].local_yz_vertices(),
            corners[CornerRotation.DOWN_270].local_xz_vertices(),
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.DOWN_270].local_yz_vertices(),
            corners[CornerRotation.DOWN_0].local_xz_vertices(),
        )
    )

    return faces


def bridge_upper_to_lower_corners(corners):
    faces = []

    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_0].local_xy_vertices(),
            corners[CornerRotation.DOWN_90].local_xy_vertices()[::-1],
            flip_normals=True,
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_90].local_xy_vertices(),
            corners[CornerRotation.DOWN_0].local_xy_vertices()[::-1],
            flip_normals=True,
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_180].local_xy_vertices(),
            corners[CornerRotation.DOWN_270].local_xy_vertices()[::-1],
            flip_normals=True,
        )
    )
    faces.extend(
        bridge_arcs(
            corners[CornerRotation.UP_270].local_xy_vertices(),
            corners[CornerRotation.DOWN_180].local_xy_vertices()[::-1],
            flip_normals=True,
        )
    )

    return faces


def bridge_corners(corners):

    faces = []

    faces.extend(bridge_upper_corners(corners))
    faces.extend(bridge_lower_corners(corners))
    faces.extend(bridge_upper_to_lower_corners(corners))

    return faces


def generate_blank_die_face(rotation, die_size, corner_radius):
    edge_length = die_size / 2 - corner_radius
    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]
    mesh_faces = np.array(
        [
            [p0, p2, p1],
            [p0, p3, p2],
        ]
    )
    return np.matmul(mesh_faces, face_rotation_matrices[rotation])


def generate_dot_panel(p0, p1, p2, p3, dot_center, die_size):
    dot_rotations = [
        CornerRotation.DOWN_0,
        CornerRotation.DOWN_90,
        CornerRotation.DOWN_180,
        CornerRotation.DOWN_270,
    ]
    dot_octants = {}
    for dot_rotation in dot_rotations:
        octant = SphereOctant(radius=DOT_RADIUS, resolution=10, flip_normals=True)
        octant.rotate(corner_rotation_matrices[dot_rotation])
        octant.translate(np.array(dot_center))
        octant.translate(np.array([0, 0, die_size / 2]))
        dot_octants[dot_rotation] = octant

    mesh_faces = []
    for octant in dot_octants.values():
        mesh_faces.extend(octant.faces)

    mesh_faces.extend(
        bridge_fan(p0, dot_octants[CornerRotation.DOWN_90].local_xy_vertices())
    )
    mesh_faces.extend(
        bridge_fan(p1, dot_octants[CornerRotation.DOWN_180].local_xy_vertices())
    )
    mesh_faces.extend(
        bridge_fan(p2, dot_octants[CornerRotation.DOWN_270].local_xy_vertices())
    )
    mesh_faces.extend(
        bridge_fan(p3, dot_octants[CornerRotation.DOWN_0].local_xy_vertices())
    )

    mesh_faces.append(
        [p0, dot_octants[CornerRotation.DOWN_180].local_xy_vertices()[0], p1]
    )
    mesh_faces.append(
        [p1, dot_octants[CornerRotation.DOWN_270].local_xy_vertices()[0], p2]
    )
    mesh_faces.append(
        [p2, dot_octants[CornerRotation.DOWN_0].local_xy_vertices()[0], p3]
    )
    mesh_faces.append(
        [p3, dot_octants[CornerRotation.DOWN_90].local_xy_vertices()[0], p0]
    )

    return mesh_faces


def generate_1_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    dot_center = [0, 0, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, p1, p2, p3, dot_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_2_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [0, edge_length, die_size / 2]
    q1 = [0, -edge_length, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread, edge_length * dot_spread, 0]
    dot1_center = [-edge_length * dot_spread, -edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, p1, q1, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, q1, p2, p3, dot1_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_3_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [edge_length / 3, edge_length, die_size / 2]
    q1 = [edge_length / 3, -edge_length, die_size / 2]

    r0 = [-edge_length / 3, edge_length, die_size / 2]
    r1 = [-edge_length / 3, -edge_length, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread, edge_length * dot_spread, 0]
    dot1_center = [0, 0, 0]
    dot2_center = [-edge_length * dot_spread, -edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, p1, q1, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, q1, r1, r0, dot1_center, die_size))
    mesh_faces.extend(generate_dot_panel(r0, r1, p2, p3, dot2_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_4_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [0, edge_length, die_size / 2]
    q1 = [0, -edge_length, die_size / 2]

    r0 = [edge_length, 0, die_size / 2]
    r1 = [-edge_length, 0, die_size / 2]

    s0 = [0, 0, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread, edge_length * dot_spread, 0]
    dot1_center = [edge_length * dot_spread, -edge_length * dot_spread, 0]
    dot2_center = [-edge_length * dot_spread, -edge_length * dot_spread, 0]
    dot3_center = [-edge_length * dot_spread, edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, r0, s0, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(r0, p1, q1, s0, dot1_center, die_size))
    mesh_faces.extend(generate_dot_panel(s0, q1, p2, r1, dot2_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, s0, r1, p3, dot3_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_5_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [edge_length / 3, edge_length, die_size / 2]
    q1 = [edge_length / 3, -edge_length, die_size / 2]

    r0 = [-edge_length / 3, edge_length, die_size / 2]
    r1 = [-edge_length / 3, -edge_length, die_size / 2]

    s0 = [edge_length, 0, die_size / 2]
    s1 = [edge_length / 3, 0, die_size / 2]
    s2 = [-edge_length / 3, 0, die_size / 2]
    s3 = [-edge_length, 0, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread, edge_length * dot_spread, 0]
    dot1_center = [edge_length * dot_spread, -edge_length * dot_spread, 0]
    dot2_center = [0, 0, 0]
    dot3_center = [-edge_length * dot_spread, -edge_length * dot_spread, 0]
    dot4_center = [-edge_length * dot_spread, edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, s0, s1, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(s0, p1, q1, s1, dot1_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, q1, r1, r0, dot2_center, die_size))
    mesh_faces.extend(generate_dot_panel(s2, r1, p2, s3, dot3_center, die_size))
    mesh_faces.extend(generate_dot_panel(r0, s2, s3, p3, dot4_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_6_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [edge_length / 3, edge_length, die_size / 2]
    q1 = [edge_length / 3, -edge_length, die_size / 2]

    r0 = [-edge_length / 3, edge_length, die_size / 2]
    r1 = [-edge_length / 3, -edge_length, die_size / 2]

    s0 = [edge_length, 0, die_size / 2]
    s1 = [edge_length / 3, 0, die_size / 2]
    s2 = [-edge_length / 3, 0, die_size / 2]
    s3 = [-edge_length, 0, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread, edge_length * dot_spread, 0]
    dot1_center = [edge_length * dot_spread, -edge_length * dot_spread, 0]
    dot2_center = [0, edge_length * dot_spread, 0]
    dot3_center = [0, -edge_length * dot_spread, 0]
    dot4_center = [-edge_length * dot_spread, edge_length * dot_spread, 0]
    dot5_center = [-edge_length * dot_spread, -edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, s0, s1, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(s0, p1, q1, s1, dot1_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, s1, s2, r0, dot2_center, die_size))
    mesh_faces.extend(generate_dot_panel(s1, q1, r1, s2, dot3_center, die_size))
    mesh_faces.extend(generate_dot_panel(r0, s2, s3, p3, dot4_center, die_size))
    mesh_faces.extend(generate_dot_panel(s2, r1, p2, s3, dot5_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_7_dot_die_face(face_rotation, die_size, corner_radius):

    edge_length = die_size / 2 - corner_radius

    p0 = [edge_length, edge_length, die_size / 2]
    p1 = [edge_length, -edge_length, die_size / 2]
    p2 = [-edge_length, -edge_length, die_size / 2]
    p3 = [-edge_length, edge_length, die_size / 2]

    q0 = [0, edge_length, die_size / 2]
    q1 = [0, -edge_length, die_size / 2]

    r0 = [edge_length, edge_length / 3, die_size / 2]
    r1 = [edge_length / 3, edge_length / 3, die_size / 2]
    r2 = [0, edge_length / 3, die_size / 2]
    r3 = [-edge_length / 3, edge_length / 3, die_size / 2]
    r4 = [-edge_length, edge_length / 3, die_size / 2]

    s0 = [edge_length, -edge_length / 3, die_size / 2]
    s1 = [edge_length / 3, -edge_length / 3, die_size / 2]
    s2 = [0, -edge_length / 3, die_size / 2]
    s3 = [-edge_length / 3, -edge_length / 3, die_size / 2]
    s4 = [-edge_length, -edge_length / 3, die_size / 2]

    dot_spread = 0.6

    dot0_center = [edge_length * dot_spread / 2, edge_length * dot_spread, 0]
    dot1_center = [-edge_length * dot_spread / 2, edge_length * dot_spread, 0]
    dot2_center = [edge_length * dot_spread, 0, 0]
    dot3_center = [0, 0, 0]
    dot4_center = [-edge_length * dot_spread, 0, 0]
    dot5_center = [edge_length * dot_spread / 2, -edge_length * dot_spread, 0]
    dot6_center = [-edge_length * dot_spread / 2, -edge_length * dot_spread, 0]

    mesh_faces = []
    mesh_faces.extend(generate_dot_panel(p0, r0, r2, q0, dot0_center, die_size))
    mesh_faces.extend(generate_dot_panel(q0, r2, r4, p3, dot1_center, die_size))
    mesh_faces.extend(generate_dot_panel(r0, s0, s1, r1, dot2_center, die_size))
    mesh_faces.extend(generate_dot_panel(r1, s1, s3, r3, dot3_center, die_size))
    mesh_faces.extend(generate_dot_panel(r3, s3, s4, r4, dot4_center, die_size))
    mesh_faces.extend(generate_dot_panel(s0, p1, q1, s2, dot5_center, die_size))
    mesh_faces.extend(generate_dot_panel(s2, q1, p2, s4, dot6_center, die_size))

    return np.matmul(mesh_faces, face_rotation_matrices[face_rotation])


def generate_die_face(rotation, dot_count, die_size, corner_radius):
    match dot_count:
        case 0:
            return generate_blank_die_face(rotation, die_size, corner_radius)
        case 1:
            return generate_1_dot_die_face(rotation, die_size, corner_radius)
        case 2:
            return generate_2_dot_die_face(rotation, die_size, corner_radius)
        case 3:
            return generate_3_dot_die_face(rotation, die_size, corner_radius)
        case 4:
            return generate_4_dot_die_face(rotation, die_size, corner_radius)
        case 5:
            return generate_5_dot_die_face(rotation, die_size, corner_radius)
        case 6:
            return generate_6_dot_die_face(rotation, die_size, corner_radius)
        case 7:
            return generate_7_dot_die_face(rotation, die_size, corner_radius)
        case _:
            raise ValueError(f"faces with {dot_count} dots not supported")


def generate_die(dice_size, corner_radius, die_spec):
    corners = GenerateCornerOctants(DICE_SIZE, CORNER_RADIUS)

    faces = []

    for octant in corners.values():
        faces.extend(octant.faces)
    faces.extend(bridge_corners(corners))
    for rotation, dot_count in die_spec:
        faces.extend(generate_die_face(rotation, dot_count, dice_size, corner_radius))

    return faces


def generate_dice_set():

    standard_die = generate_die(
        DICE_SIZE,
        CORNER_RADIUS,
        (
            (FaceRotation.TOP, 1),
            (FaceRotation.BOTTOM, 6),
            (FaceRotation.LEFT, 2),
            (FaceRotation.RIGHT, 5),
            (FaceRotation.FRONT, 3),
            (FaceRotation.BACK, 4),
        ),
    )

    red_die = generate_die(
        DICE_SIZE,
        CORNER_RADIUS,
        (
            (FaceRotation.TOP, 1),
            (FaceRotation.BOTTOM, 5),
            (FaceRotation.LEFT, 1),
            (FaceRotation.RIGHT, 5),
            (FaceRotation.FRONT, 5),
            (FaceRotation.BACK, 5),
        ),
    )
    red_die = np.add(red_die, [3 * DICE_SIZE, 0, 0])

    white_die = generate_die(
        DICE_SIZE,
        CORNER_RADIUS,
        (
            (FaceRotation.TOP, 6),
            (FaceRotation.BOTTOM, 2),
            (FaceRotation.LEFT, 6),
            (FaceRotation.RIGHT, 2),
            (FaceRotation.FRONT, 6),
            (FaceRotation.BACK, 2),
        ),
    )
    white_die = np.add(white_die, [0, 3 * DICE_SIZE, 0])

    blue_die = generate_die(
        DICE_SIZE,
        CORNER_RADIUS,
        (
            (FaceRotation.TOP, 7),
            (FaceRotation.BOTTOM, 3),
            (FaceRotation.LEFT, 7),
            (FaceRotation.RIGHT, 3),
            (FaceRotation.FRONT, 3),
            (FaceRotation.BACK, 3),
        ),
    )
    blue_die = np.add(blue_die, [-3 * DICE_SIZE, 0, 0])

    black_die = generate_die(
        DICE_SIZE,
        CORNER_RADIUS,
        (
            (FaceRotation.TOP, 4),
            (FaceRotation.BOTTOM, 4),
            (FaceRotation.LEFT, 4),
            (FaceRotation.RIGHT, 4),
            (FaceRotation.FRONT, 4),
            (FaceRotation.BACK, 4),
        ),
    )
    black_die = np.add(black_die, [0, -3 * DICE_SIZE, 0])

    faces = []
    faces.extend(standard_die)
    faces.extend(red_die)
    faces.extend(white_die)
    faces.extend(blue_die)
    faces.extend(black_die)

    return faces


def main():

    # Create dice wireframe.
    faces = generate_dice_set()

    # Create the STL mesh from the wireframe.
    octant_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        octant_mesh.vectors[i] = f

    # Write the mesh to a file.
    octant_mesh.save("dice.stl")


if __name__ == "__main__":
    main()
