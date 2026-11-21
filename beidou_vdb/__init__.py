"""Voxel occupancy fill for building glTF meshes."""

from beidou_vdb.gltf_load import load_gltf_triangles
from beidou_vdb.occupy import fill_occupancy

__all__ = ["fill_occupancy", "load_gltf_triangles"]
