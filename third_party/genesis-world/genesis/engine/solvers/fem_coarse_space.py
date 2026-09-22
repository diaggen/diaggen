"""Generic graph-based coarse-space helpers for volumetric FEM meshes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


@dataclass(frozen=True, slots=True)
class MaterialConnectedPartition:
    partition_index: int
    material_index: int
    material_key: tuple[float, ...]
    component_index: int
    component_tet_count: int
    component_vertex_count: int
    partition_index_in_component: int
    partition_count_in_component: int
    seed_tet_index: int
    support_vertex_count: int = 0


@dataclass(frozen=True, slots=True)
class RigidMotionCandidate:
    column_index: int
    partition_index: int
    mode: str
    center: tuple[float, float, float]
    raw_norm: float
    partition: MaterialConnectedPartition | None


def _tet_adjacency(tetrahedra: np.ndarray, n_vertices: int) -> sp.csr_matrix:
    n_tets = tetrahedra.shape[0]
    incidence = sp.csr_matrix(
        (
            np.ones(n_tets * 4, dtype=np.float64),
            (np.repeat(np.arange(n_tets, dtype=np.int64), 4), tetrahedra.reshape(-1)),
        ),
        shape=(n_tets, n_vertices),
    )
    adjacency = (incidence @ incidence.T).tocsr()
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)
    return adjacency


def _vertex_adjacency(tetrahedra: np.ndarray, component_vertices: np.ndarray) -> sp.csr_matrix:
    global_to_local = np.full(int(component_vertices[-1]) + 1, -1, dtype=np.int64)
    global_to_local[component_vertices] = np.arange(component_vertices.size, dtype=np.int64)
    local_tetrahedra = global_to_local[tetrahedra]
    local_pairs = np.asarray([(i, j) for i in range(4) for j in range(4) if i != j], dtype=np.int64)
    rows = local_tetrahedra[:, local_pairs[:, 0]].reshape(-1)
    cols = local_tetrahedra[:, local_pairs[:, 1]].reshape(-1)
    adjacency = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, cols)),
        shape=(component_vertices.size, component_vertices.size),
    )
    adjacency.data.fill(1.0)
    return adjacency


def _graph_farthest_point_seeds(
    adjacency: sp.csr_matrix,
    global_tet_indices: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_local_indices = [int(np.argmin(global_tet_indices))]
    distance_rows = []
    minimum_distance = np.full(global_tet_indices.size, np.inf, dtype=np.float64)
    selected = np.zeros(global_tet_indices.size, dtype=np.bool_)

    for i_seed in range(count):
        seed_local = seed_local_indices[i_seed]
        selected[seed_local] = True
        distance = np.asarray(
            csgraph.shortest_path(adjacency, directed=False, unweighted=True, indices=seed_local),
            dtype=np.float64,
        )
        distance_rows.append(distance)
        minimum_distance = np.minimum(minimum_distance, distance)
        if i_seed + 1 < count:
            farthest_distance = np.max(minimum_distance[~selected])
            candidates = np.flatnonzero((minimum_distance == farthest_distance) & ~selected)
            seed_local_indices.append(int(candidates[np.argmin(global_tet_indices[candidates])]))

    distances = np.stack(distance_rows, axis=0)
    return global_tet_indices[np.asarray(seed_local_indices, dtype=np.int64)], distances


def build_material_connected_partition_of_unity(
    tetrahedra,
    material_keys,
    n_vertices: int,
    *,
    max_partitions_per_component: int = 8,
    target_tets_per_partition: int = 128,
    smoothing_steps: int = 3,
):
    """Build deterministic overlapping PoU weights over material-connected tet graphs.

    Tets are adjacent when they share a global vertex.  Each same-material
    connected component receives ``min(8, ceil(n_tets / 128))`` graph-distance
    farthest-point seeds by default.  Seed Voronoi indicators are transferred
    to the component's existing global vertices, smoothed three times with
    ``0.5 I + 0.5 D^-1 Adj``, then normalized across every partition so the
    weights sum to one at each FEM vertex.  Vertices are never duplicated.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    material_keys = np.asarray(material_keys)
    if material_keys.ndim == 1:
        material_keys = material_keys[:, None]
    numeric_material_keys = np.asarray(material_keys, dtype=np.float64)
    unique_materials, material_inverse = np.unique(numeric_material_keys, axis=0, return_inverse=True)

    partition_weights = []
    metadata = []
    component_index = 0
    for material_index, material_key in enumerate(unique_materials):
        material_tets = np.flatnonzero(material_inverse == material_index)
        material_adjacency = _tet_adjacency(tetrahedra[material_tets], n_vertices)
        _, labels = csgraph.connected_components(material_adjacency, directed=False)
        components = [material_tets[labels == label] for label in np.unique(labels)]
        components.sort(key=lambda indices: int(indices.min()))

        for component_tets in components:
            component_tetrahedra = tetrahedra[component_tets]
            component_vertices = np.unique(component_tetrahedra)
            component_adjacency = _tet_adjacency(component_tetrahedra, n_vertices)
            partition_count = min(
                max_partitions_per_component,
                max(1, (component_tets.size + target_tets_per_partition - 1) // target_tets_per_partition),
            )
            seed_tets, seed_distances = _graph_farthest_point_seeds(
                component_adjacency,
                component_tets,
                partition_count,
            )
            tet_assignment = np.argmin(seed_distances, axis=0)

            component_weights = np.zeros((partition_count, component_vertices.size), dtype=np.float64)
            global_to_local = np.full(int(component_vertices[-1]) + 1, -1, dtype=np.int64)
            global_to_local[component_vertices] = np.arange(component_vertices.size, dtype=np.int64)
            local_tetrahedra = global_to_local[component_tetrahedra]
            for partition_index_in_component in range(partition_count):
                assigned_vertices = np.unique(local_tetrahedra[tet_assignment == partition_index_in_component])
                component_weights[partition_index_in_component, assigned_vertices] = 1.0
            component_weights /= component_weights.sum(axis=0, keepdims=True)

            vertex_adjacency = _vertex_adjacency(component_tetrahedra, component_vertices)
            vertex_degree = np.asarray(vertex_adjacency.sum(axis=1), dtype=np.float64).reshape(-1)
            for _ in range(smoothing_steps):
                neighbor_average = vertex_adjacency @ component_weights.T
                neighbor_average /= vertex_degree[:, None]
                component_weights = 0.5 * component_weights + 0.5 * neighbor_average.T

            for partition_index_in_component in range(partition_count):
                global_weights = np.zeros(n_vertices, dtype=np.float64)
                global_weights[component_vertices] = component_weights[partition_index_in_component]
                partition_index = len(partition_weights)
                partition_weights.append(global_weights)
                metadata.append(
                    MaterialConnectedPartition(
                        partition_index=partition_index,
                        material_index=material_index,
                        material_key=tuple(float(value) for value in material_key),
                        component_index=component_index,
                        component_tet_count=int(component_tets.size),
                        component_vertex_count=int(component_vertices.size),
                        partition_index_in_component=partition_index_in_component,
                        partition_count_in_component=partition_count,
                        seed_tet_index=int(seed_tets[partition_index_in_component]),
                    )
                )
            component_index += 1

    weights = np.asarray(partition_weights, dtype=np.float64)
    weights /= weights.sum(axis=0, keepdims=True)
    metadata = tuple(
        replace(item, support_vertex_count=int(np.count_nonzero(weights[item.partition_index] > 0.0)))
        for item in metadata
    )
    return weights, metadata


def build_weighted_rigid_motion_candidates(
    positions,
    weights,
    *,
    vertex_masses=None,
    partition_metadata=None,
):
    """Build partition-weighted translation and current-position rotation columns."""
    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if vertex_masses is None:
        vertex_masses = np.ones(positions.shape[0], dtype=np.float64)
    else:
        vertex_masses = np.asarray(vertex_masses, dtype=np.float64)
    if partition_metadata is None:
        partition_metadata = (None,) * weights.shape[0]

    axes = np.eye(3, dtype=np.float64)
    mode_names = ("tx", "ty", "tz", "rx", "ry", "rz")
    columns = []
    candidate_metadata = []
    for partition_index, partition_weights in enumerate(weights):
        center_weights = partition_weights * vertex_masses
        center = np.sum(positions * center_weights[:, None], axis=0) / center_weights.sum()
        relative = positions - center
        partition_columns = [partition_weights[:, None] * axis for axis in axes]
        partition_columns.extend(
            partition_weights[:, None] * np.cross(axis, relative)
            for axis in axes
        )
        for mode, column in zip(mode_names, partition_columns):
            flat_column = np.asarray(column, dtype=np.float64).reshape(-1)
            column_index = len(columns)
            columns.append(flat_column)
            candidate_metadata.append(
                RigidMotionCandidate(
                    column_index=column_index,
                    partition_index=partition_index,
                    mode=mode,
                    center=tuple(float(value) for value in center),
                    raw_norm=float(np.linalg.norm(flat_column)),
                    partition=partition_metadata[partition_index],
                )
            )
    return np.column_stack(columns), tuple(candidate_metadata)
