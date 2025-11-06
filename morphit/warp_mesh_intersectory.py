
import trimesh
import warp
import torch
import numpy as np
import warp as wp
from typing import Union


class WarpMeshIntersector:
    def __init__(self, mesh: trimesh.Trimesh, device: torch.device):
        wp.config.quiet = True
        wp.init()

        if device.index is None:
            device = torch.device("cuda", 0)
        self.device = device

        verts = mesh.vertices
        faces = mesh.faces
        self._wp_device = wp.device_from_torch(device)
        self._wp_verts = wp.array(verts, dtype=wp.vec3, device=self._wp_device)
        self._wp_faces = wp.array(np.ravel(faces), dtype=wp.int32, device=self._wp_device)

        bounds = mesh.bounds # bounds[0] contains min, bounds[1] contains max

        # Calculate max distance as the diagonal of the bounding box
        box_dimensions = bounds[1] - bounds[0]  # [width, height, depth]
        self._max_distance = float(np.linalg.norm(box_dimensions)) + 0.01

        self._wp_mesh = wp.Mesh(points=self._wp_verts, indices=self._wp_faces)
        self._out_mask = torch.zeros((1,), dtype=torch.bool, device=self.device)
        self._batch_size = 1

    def update_batch_size(self, batch_size: int):
        if batch_size != self._batch_size:
            self._batch_size = batch_size
            self._out_mask = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)


    def query(self, points: torch.Tensor) -> torch.Tensor:

        # launch warp kernel:
        if len(points.shape) != 2:
            raise ValueError("Points must be a 2D tensor")
        num_points = points.shape[0]
        self.update_batch_size(num_points)
        wp_device = wp.device_from_torch(points.device)
        if wp_device != self._wp_device:
            raise ValueError("Points must be on the same device as the mesh")

        wp_stream = None if not points.is_cuda else wp.stream_from_torch(points.device)

        wp.launch(
            kernel=query_mesh_outside,
            dim=num_points,
            inputs=[
                self._wp_mesh.id,
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(self._out_mask, dtype=wp.bool),
                self._max_distance,
            ],
            stream=wp_stream,
            device=self._wp_device,
        )

        return self._out_mask




@wp.kernel
def query_mesh_outside(mesh: wp.uint64,
                       query_points: wp.array(dtype=wp.vec3),
                       outside_mask: wp.array(dtype=wp.bool),
                       max_distance: wp.float32):
    tid = wp.tid()

    point = query_points[tid]

    result = wp.mesh_query_point(mesh, point, max_distance)
    inside = False
    if result.result:
        inside = result.sign < 0 # true if point is inside the mesh
    outside = not inside
    outside_mask[tid] = outside
