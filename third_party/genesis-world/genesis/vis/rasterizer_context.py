import math
import time

import numpy as np
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
import genesis.utils.particle as pu
from genesis.ext import pyrender
from genesis.ext.pyrender.jit_render import JITRenderer
from genesis.utils.misc import tensor_to_array, qd_to_numpy

PART_SEGMENTATION_CONTEXT_EDGE_RADIUS = 0.004
PART_SHADED_METALLIC = 0.0
PART_SHADED_ROUGHNESS = 0.65
PART_SHADED_CREASE_DEGREES = 50.0


def _crease_surface_corner_normals(vertices, triangles):
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    lengths = np.linalg.norm(face_normals, axis=1)
    unit_face_normals = np.zeros_like(face_normals, dtype=np.float32)
    unit_face_normals[lengths > 0] = face_normals[lengths > 0] / lengths[lengths > 0, None]
    incident_faces = [[] for _ in range(len(vertices))]
    for face_index, face in enumerate(triangles):
        for vertex_index in face:
            incident_faces[int(vertex_index)].append(face_index)
    crease_cosine = math.cos(math.radians(PART_SHADED_CREASE_DEGREES))
    corner_normals = np.empty((len(triangles), 3, 3), dtype=np.float32)
    for face_index, face in enumerate(triangles):
        reference = unit_face_normals[face_index]
        for corner_index, vertex_index in enumerate(face):
            candidates = np.asarray(incident_faces[int(vertex_index)], dtype=np.int64)
            smooth_faces = candidates[(unit_face_normals[candidates] @ reference) >= crease_cosine]
            normal = np.sum(face_normals[smooth_faces], axis=0)
            normal_length = float(np.linalg.norm(normal))
            corner_normals[face_index, corner_index] = normal / normal_length if normal_length > 0 else reference
    return corner_normals


def _part_segmentation_wireframe_box_mesh(bounds):
    bounds = np.asarray(bounds, dtype=np.float32)
    center = (bounds[0] + bounds[1]) * 0.5
    size = bounds[1] - bounds[0]
    edge_meshes = []
    for axis in range(3):
        transverse_axes = [index for index in range(3) if index != axis]
        for side_0 in (0, 1):
            for side_1 in (0, 1):
                edge_center = center.copy()
                edge_center[transverse_axes[0]] = bounds[side_0, transverse_axes[0]]
                edge_center[transverse_axes[1]] = bounds[side_1, transverse_axes[1]]
                edge_extents = np.full(3, 2.0 * PART_SEGMENTATION_CONTEXT_EDGE_RADIUS, dtype=np.float32)
                edge_extents[axis] = max(float(size[axis]), 2.0 * PART_SEGMENTATION_CONTEXT_EDGE_RADIUS)
                edge = trimesh.creation.box(extents=edge_extents)
                edge.apply_translation(edge_center)
                edge_meshes.append(edge)
    return trimesh.util.concatenate(edge_meshes)


class SegmentationColorMap:
    def __init__(self, seed: int = 0, to_torch: bool = False):
        self.seed = seed
        self.to_torch = to_torch
        self.idxc_map = {0: -1}
        self.key_map = {-1: 0}
        self._next_idxc = 1
        self.idxc_to_color = None

    def seg_key_to_idxc(self, seg_key):
        seg_idxc = self.key_map.get(seg_key)
        if seg_idxc is None:
            seg_idxc = self._next_idxc
            self._next_idxc += 1
            self.key_map[seg_key] = seg_idxc
        self.idxc_map[seg_idxc] = seg_key
        return seg_idxc

    def remove_seg_key(self, seg_key):
        seg_idxc = self.key_map.pop(seg_key, None)
        if seg_idxc is not None:
            self.idxc_map.pop(seg_idxc, None)
        return seg_idxc

    def seg_idxc_to_key(self, seg_idxc):
        return self.idxc_map[seg_idxc]

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self.idxc_to_color[seg_idxc_arr]

    def generate_seg_colors(self):
        # seg_key: same as entity/link/geom's idx
        # seg_idxc: segmentation index of objects
        # seg_idxc_rgb: colorized seg_idxc internally used by renderer

        # Evenly spaced hues
        num_keys = max(self.idxc_map, default=0) + 1
        hues = np.linspace(0.0, 1.0, num_keys, endpoint=False)
        rng = np.random.default_rng(seed=self.seed)
        rng.shuffle(hues)

        # Fixed saturation/value
        s, v = 0.8, 0.95

        # HSV to RGB conversion
        rgb = np.zeros((num_keys, 3), dtype=np.float32)
        i = (hues * 6).astype(np.int32)
        f = hues * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        for k in range(1, num_keys):  # Skip first color to enforce black background
            match i[k] % 6:
                case 0:
                    rgb[k] = (v, t[k], p)
                case 1:
                    rgb[k] = (q[k], v, p)
                case 2:
                    rgb[k] = (p, v, t[k])
                case 3:
                    rgb[k] = (p, q[k], v)
                case 4:
                    rgb[k] = (t[k], p, v)
                case 5:
                    rgb[k] = (v, p, q[k])
        rgb = mu.color_f32_to_u8(rgb)

        # Store the generated map
        if self.to_torch:
            self.idxc_to_color = torch.from_numpy(rgb).to(device=gs.device)
        else:
            self.idxc_to_color = rgb


class RasterizerContext:
    def __init__(self, options):
        self.show_world_frame = options.show_world_frame
        self.world_frame_size = options.world_frame_size
        self.show_link_frame = options.show_link_frame
        self.link_frame_size = options.link_frame_size
        self.show_cameras = options.show_cameras
        self.shadow = options.shadow
        self.plane_reflection = options.plane_reflection
        self.ambient_light = options.ambient_light
        self.background_color = options.background_color
        self.segmentation_level = options.segmentation_level
        self.lights = options.lights
        self.visualize_mpm_boundary = options.visualize_mpm_boundary
        self.visualize_sph_boundary = options.visualize_sph_boundary
        self.visualize_pbd_boundary = options.visualize_pbd_boundary
        self.particle_size_scale = options.particle_size_scale
        self.contact_force_scale = options.contact_force_scale
        self.render_particle_as = options.render_particle_as
        self.rendered_envs_idx = list(options.rendered_envs_idx) if options.rendered_envs_idx is not None else None
        self.env_separate_rigid = options.env_separate_rigid

        # nodes
        self.world_frame_node = None
        self.link_frame_node = None
        self.link_frame_nodes = dict()
        self.frustum_nodes = dict()  # nodes camera frustums
        self.rigid_nodes = dict()
        # (env_idx, geom.uid) -> per-env pyrender node for kinematic entities currently driven by set_vverts
        self.vverts_nodes = dict()
        self._per_env_vverts_entity_uids: set = set()
        self.static_nodes = dict()  # used across all frames
        self._fem_surface_vertex_indices = dict()
        self._fem_surface_triangles = dict()
        self._part_segmentation_vertex_indices = dict()
        self._part_shaded_corner_indices = dict()
        self.part_segmentation_nodes = dict()
        self.part_shaded_overlay_nodes = set()
        self.part_segmentation_context_nodes = dict()
        self._part_segmentation_context_vertex_counts = dict()
        self._part_segmentation_context_colors = dict()
        self._part_segmentation_context_seg_keys = dict()
        self.last_part_segmentation_context_update = {}
        self.part_segmentation_indexed_counts = dict()
        self.segmentation_only_nodes = set()
        self.rgb_only_nodes = set()
        self.part_segmentation_palette_by_idxc = {}
        self.last_part_segmentation_update = {}
        self.last_render_update_stats = {}
        self.dynamic_nodes = dict()  # nodes that live within single frame
        self.external_nodes = dict()  # nodes added by external user
        self.seg_node_map = dict()
        self.seg_color_map = SegmentationColorMap()

        self.init_meshes()

    def init_meshes(self):
        self.world_frame_shown = False
        self.link_frame_shown = False
        self.camera_frustum_shown = False

        self.world_frame_mesh = mu.create_frame(
            origin_radius=0.012,
            axis_radius=0.005,
            axis_length=self.world_frame_size,
            head_radius=0.01,
            head_length=0.03,
        )

        self.link_frame_mesh = trimesh.creation.axis(origin_size=0.03, axis_radius=0.025, axis_length=1.0)
        self.link_frame_mesh.visual.face_colors[:, :3] = (0.7 * self.link_frame_mesh.visual.face_colors[:, :3]).astype(
            int
        )
        self.link_frame_mesh.vertices *= self.link_frame_size

    def build(self, scene):
        self.scene = scene
        self.sim = scene.sim
        self.visualizer = scene.visualizer

        # Update visuals at this point avoids nasty visual artifacts during Scene build
        self.visualizer.update_visual_states()

        if self.rendered_envs_idx is None:
            self.rendered_envs_idx = list(range(self.sim._B))

        # pyrender scene
        self._scene = pyrender.Scene(
            ambient_light=self.ambient_light,
            bg_color=self.background_color,
            n_envs=len(self.rendered_envs_idx),
        )

        self.jit = JITRenderer(self._scene, [], [])

        self.on_lights()

        if self.show_world_frame:
            self.on_world_frame()
        if self.show_link_frame:
            self.on_link_frame()
        if self.show_cameras:
            self.on_camera_frustum()

        self.on_tool()
        self.on_rigid()
        self.on_mpm()
        self.on_sph()
        self.on_pbd()
        self.on_fem()

        # segmentation mapping
        self.seg_color_map.generate_seg_colors()

    def destroy(self):
        self.clear_dynamic_nodes(only_outdated=False)

        for node_registry in (
            self.link_frame_nodes,
            self.frustum_nodes,
            self.rigid_nodes,
            self.vverts_nodes,
            self.static_nodes,
            self.external_nodes,
            self.part_segmentation_nodes,
            self.part_segmentation_context_nodes,
        ):
            for external_node in node_registry.values():
                self.remove_node(external_node)
            node_registry.clear()
        self._per_env_vverts_entity_uids.clear()
        self._fem_surface_vertex_indices.clear()
        self._fem_surface_triangles.clear()
        self._part_segmentation_vertex_indices.clear()
        self._part_shaded_corner_indices.clear()
        self.part_segmentation_nodes.clear()
        self.part_shaded_overlay_nodes.clear()
        self.part_segmentation_context_nodes.clear()
        self._part_segmentation_context_vertex_counts.clear()
        self._part_segmentation_context_colors.clear()
        self._part_segmentation_context_seg_keys.clear()
        self.last_part_segmentation_context_update.clear()
        self.part_segmentation_indexed_counts.clear()
        self.segmentation_only_nodes.clear()
        self.rgb_only_nodes.clear()
        self.part_segmentation_palette_by_idxc.clear()

    def reset(self):
        self._t = -1

    def add_node(self, obj, **kwargs):
        with self.scene._visualizer.viewer_lock:
            return self._scene.add(obj, **kwargs)

    def remove_node(self, node):
        if self.scene._visualizer is None:
            self._scene.remove_node(node)
        else:
            with self.scene._visualizer.viewer_lock:
                self._scene.remove_node(node)

    def _get_geom_active_envs_idx(self, geom, rendered_envs_idx):
        """Get the intersection of geom.active_envs_idx (for heterogeneous sim) and rendered_envs_idx.

        For heterogeneous simulation, each geom is only active in certain environments.
        This returns the environments where the geom should actually be rendered.
        """
        geom_active_envs_idx = geom.active_envs_idx
        if geom_active_envs_idx is not None:
            return np.intersect1d(geom_active_envs_idx, rendered_envs_idx)
        return rendered_envs_idx

    def add_rigid_node(self, geom, obj, **kwargs):
        rigid_node = self.add_node(obj, **kwargs)
        self.rigid_nodes[geom.uid] = rigid_node

        # create segemtation id
        if self.segmentation_level == "geom":
            seg_key = (geom.entity.idx, geom.link.idx, geom.idx)
        elif self.segmentation_level == "link":
            seg_key = (geom.entity.idx, geom.link.idx)
        elif self.segmentation_level == "entity":
            seg_key = geom.entity.idx
        else:
            gs.raise_exception(f"Unsupported segmentation level: {self.segmentation_level}")
        self.create_node_seg(seg_key, rigid_node)

    def add_static_node(self, entity, obj, i_b, **kwargs):
        static_node = self.add_node(obj, **kwargs)
        self.static_nodes[(i_b, entity.uid)] = static_node
        self.create_node_seg(entity.idx, static_node)

    def add_dynamic_node(self, entity, obj, **kwargs):
        if obj:
            dynamic_node = self.add_node(obj, **kwargs)
            self.dynamic_nodes.setdefault(self.scene._t, []).append(dynamic_node)
        else:
            dynamic_node = None
        if entity:
            self.create_node_seg(entity.idx, dynamic_node)

    def add_external_node(self, obj, **kwargs):
        # Check if the node has a valid name
        if not hasattr(obj, "name") or not obj.name:
            gs.raise_exception("Node must have a valid 'name' attribute.")

        # Check if the name is already in use
        if obj.name in self.external_nodes:
            gs.raise_exception(f"A node with the name '{obj.name}' already exists.")

        self.external_nodes[obj.name] = self.add_node(obj, **kwargs)

    def mark_part_shaded_overlay(self, obj):
        self.part_shaded_overlay_nodes.add(self.external_nodes[obj.name])

    def clear_dynamic_nodes(self, only_outdated: bool = True):
        for t in tuple(self.dynamic_nodes.keys()):
            if not only_outdated or t < self.scene._t:
                for dynamic_node in self.dynamic_nodes.pop(t):
                    self.remove_node_seg(dynamic_node)
                    self.remove_node(dynamic_node)

    def clear_external_node(self, node):
        if node.name in self.external_nodes:
            scene_node = self.external_nodes[node.name]
            self.part_shaded_overlay_nodes.discard(scene_node)
            self.remove_node(scene_node)
            del self.external_nodes[node.name]

    def clear_external_nodes(self):
        for external_node in self.external_nodes.values():
            self.remove_node(external_node)
        self.external_nodes.clear()
        self.part_shaded_overlay_nodes.clear()

    def set_node_pose(self, node, pose):
        self._scene.set_pose(node, pose)

    def update_camera_frustum(self, camera):
        if self.camera_frustum_shown:
            self.set_node_pose(self.frustum_nodes[camera.uid], camera.transform)

    def on_camera_frustum(self):
        if not self.camera_frustum_shown:
            for camera in self.cameras:
                self.frustum_nodes[camera.uid] = self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_camera_frustum(camera, color=(1.0, 1.0, 1.0, 0.3)),
                        smooth=False,
                    )
                )
            self.camera_frustum_shown = True

    def off_camera_frustum(self):
        if self.camera_frustum_shown:
            for camera in self.cameras:
                self.remove_node(self.frustum_nodes[camera.uid])
            self.frustum_nodes.clear()
            self.camera_frustum_shown = False

    def on_world_frame(self):
        if not self.world_frame_shown:
            mesh = pyrender.Mesh.from_trimesh(self.world_frame_mesh, smooth=True, is_marker=True)
            self.world_frame_node = self.add_node(mesh)
            self.world_frame_shown = True

    def off_world_frame(self):
        if self.world_frame_shown:
            self.remove_node(self.world_frame_node)
            self.world_frame_node = None
            self.world_frame_shown = False

    def _link_frame_T(self, solver):
        """World-space 4x4 transforms for every (env, link) of solver, flattened in env-major, link-minor order."""
        pos = qd_to_numpy(solver.links_state.pos, self.rendered_envs_idx, transpose=True, copy=True)
        quat = qd_to_numpy(solver.links_state.quat, self.rendered_envs_idx, transpose=True)
        pos += self.scene.envs_offset[self.rendered_envs_idx, None]
        return gu.trans_quat_to_T(pos.reshape(-1, 3), quat.reshape(-1, 4))

    def on_link_frame(self):
        if not self.link_frame_shown:
            if self.env_separate_rigid:
                for solver in self._rigid_solvers():
                    all_T = self._link_frame_T(solver)
                    n_links = len(solver.links)
                    for i, link in enumerate(solver.links):
                        mesh = pyrender.Mesh.from_trimesh(
                            mesh=self.link_frame_mesh,
                            poses=all_T[i::n_links],
                            env_shared=False,
                            is_marker=True,
                        )
                        self.link_frame_nodes[link.uid] = self.add_node(mesh)
            else:
                all_T_parts = [self._link_frame_T(solver) for solver in self._rigid_solvers()]
                if all_T_parts:
                    mesh = pyrender.Mesh.from_trimesh(
                        mesh=self.link_frame_mesh,
                        poses=np.concatenate(all_T_parts, axis=0),
                        is_marker=True,
                    )
                    self.link_frame_node = self.add_node(mesh)
            self.link_frame_shown = True

    def off_link_frame(self):
        if self.link_frame_shown:
            if self.env_separate_rigid:
                for node in self.link_frame_nodes.values():
                    self.remove_node(node)
                self.link_frame_nodes.clear()
            elif self.link_frame_node is not None:
                self.remove_node(self.link_frame_node)
                self.link_frame_node = None
            self.link_frame_shown = False

    def update_link_frame(self):
        if self.link_frame_shown:
            if self.env_separate_rigid:
                for solver in self._rigid_solvers():
                    all_T = self._link_frame_T(solver)
                    n_links = len(solver.links)
                    for i, link in enumerate(solver.links):
                        link_T = all_T[i::n_links]
                        node = self.link_frame_nodes[link.uid]
                        node.mesh.primitives[0].poses = link_T
                        buf_id = self._scene.get_buffer_id(node, "model")
                        if buf_id >= 0:
                            self.jit.update_buffer(buf_id, link_T.transpose((0, 2, 1)))
            elif self.link_frame_node is not None:
                all_T = np.concatenate([self._link_frame_T(solver) for solver in self._rigid_solvers()], axis=0)
                self.link_frame_node.mesh.primitives[0].poses = all_T
                buf_id = self._scene.get_buffer_id(self.link_frame_node, "model")
                if buf_id >= 0:
                    self.jit.update_buffer(buf_id, all_T.transpose((0, 2, 1)))

    def on_tool(self):
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                if tool_entity.mesh is not None:
                    for idx in self.rendered_envs_idx:
                        mesh = trimesh.Trimesh(
                            tool_entity.mesh.raw_vertices,
                            tool_entity.mesh.faces_np.reshape([-1, 3]),
                            tool_entity.mesh.raw_vertex_normals,
                            process=False,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(tool_entity.surface, n_verts=len(mesh.vertices))

                        pose = gu.trans_quat_to_T(tool_entity.init_pos, tool_entity.init_quat)
                        double_sided = tool_entity.surface.double_sided
                        self.add_static_node(
                            tool_entity, pyrender.Mesh.from_trimesh(mesh, double_sided=double_sided), i_b=idx, pose=pose
                        )

    def update_tool(self):
        if self.sim.tool_solver.is_active:
            for tool_entity in self.sim.tool_solver.entities:
                poss = qd_to_numpy(tool_entity.pos)[self.sim.cur_substep_local] + self.scene.envs_offset
                quats = qd_to_numpy(tool_entity.quat)[self.sim.cur_substep_local]
                for idx in self.rendered_envs_idx:
                    pose = gu.trans_quat_to_T(poss[idx], quats[idx])
                    self.set_node_pose(self.static_nodes[(idx, tool_entity.uid)], pose=pose)

    def set_reflection_mat(self, geom_T):
        height = geom_T[0, 2, 3]
        self.jit.reflection_mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, height * 2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _rigid_solvers(self):
        """Yield active solvers that manage KinematicEntity-based entities (rigid + kinematic)."""
        if self.sim.rigid_solver.is_active:
            yield self.sim.rigid_solver
        if self.sim.kinematic_solver.is_active:
            yield self.sim.kinematic_solver

    def on_rigid(self):
        # Reuse a single pyrender material per genesis surface so that geoms sharing one surface (e.g. the many
        # textured submeshes of a GLB, which are kept separate to preserve baked convex decompositions) expose the
        # same Texture instances. The renderer then keeps a single host copy and GPU upload instead of one per geom.
        vis_materials = {}
        for solver in self._rigid_solvers():
            # TODO: support dynamic switching in GUI later
            for entity in solver.entities:
                if entity.surface.vis_mode == "visual":
                    geoms = entity.vgeoms
                    geoms_T = solver._vgeoms_render_T
                else:
                    geoms = entity.geoms
                    geoms_T = solver._geoms_render_T

                for geom in geoms:
                    # For heterogeneous simulation, filter envs based on geom's assigned environments
                    geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                    if len(geom_envs_idx) == 0:
                        continue

                    if "sdf" in entity.surface.vis_mode:
                        mesh = geom.get_sdf_trimesh()
                    else:
                        mesh = geom.get_trimesh()
                    # A heterogeneous variant is present in only a subset of environments. Render its full per-env pose
                    # set with a per-env visibility mask, so the per-env draw places the variant in its own environment
                    # instead of collapsing the subset of poses onto the wrong environments.
                    active_envs = None
                    if len(geom_envs_idx) < len(self.rendered_envs_idx):
                        geom_T = geoms_T[geom.idx][self.rendered_envs_idx]
                        active_envs = np.isin(self.rendered_envs_idx, geom_envs_idx)
                    else:
                        geom_T = geoms_T[geom.idx][geom_envs_idx]

                    # For z-axis normal planes, render a single instance shared across all envs to avoid z-fighting,
                    # unless they do not overlap. Env-masked variants always take the per-env path.
                    env_shared = active_envs is None and not self.env_separate_rigid
                    if not env_shared and active_envs is None and isinstance(entity.main_morph, gs.morphs.Plane):
                        plane_normal, plane_size = entity.main_morph.normal, entity.main_morph.plane_size
                        if (
                            abs(plane_normal[0]) < gs.EPS
                            and abs(plane_normal[1]) < gs.EPS
                            and self.scene.env_spacing[0] < plane_size[0]
                            and self.scene.env_spacing[1] < plane_size[1]
                        ):
                            geom_T = geom_T[:1]
                            env_shared = True

                    surface_key = id(geom.surface)
                    mesh_node = pyrender.Mesh.from_trimesh(
                        mesh=mesh,
                        poses=geom_T,
                        smooth=geom.surface.smooth if "collision" not in entity.surface.vis_mode else False,
                        double_sided=(
                            geom.surface.double_sided if "collision" not in entity.surface.vis_mode else False
                        ),
                        is_floor=isinstance(entity._morph, gs.morphs.Plane),
                        env_shared=env_shared,
                        active_envs=active_envs,
                        material=vis_materials.get(surface_key),
                    )
                    vis_materials.setdefault(surface_key, mesh_node.primitives[0].material)
                    self.add_rigid_node(geom, mesh_node)
                    if isinstance(entity._morph, gs.morphs.Plane):
                        self.set_reflection_mat(geom_T)

    def update_rigid(self):
        for solver in self._rigid_solvers():
            for entity in solver.entities:
                if entity.surface.vis_mode == "visual":
                    geoms = entity.vgeoms
                    geoms_T = solver._vgeoms_render_T
                    if entity._morph.enable_custom_vverts:
                        if entity.uid not in self._per_env_vverts_entity_uids:
                            # The per-env node's GL buffers are only allocated during the upcoming render pass, so any
                            # queued update_buffer call resolving its id right now would hit -1 and be silently
                            # dropped. Seed primitive.positions with the current world-space vverts directly so the
                            # first frame is already correct.
                            vverts = qd_to_numpy(solver.vverts_state.pos, self.rendered_envs_idx, transpose=True)
                            envs_offset = self.scene.envs_offset
                            custom_offset = entity._custom_vvert_start - entity._vvert_start
                            n_rendered_envs = len(self.rendered_envs_idx)
                            identity_poses = np.tile(np.eye(4, dtype=np.float32), (n_rendered_envs, 1, 1))
                            for geom in entity.vgeoms:
                                old_node = self.rigid_nodes.pop(geom.uid, None)
                                if old_node is not None:
                                    self.remove_node_seg(old_node)
                                    self.remove_node(old_node)

                                geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                                if len(geom_envs_idx) == 0:
                                    continue

                                mesh = geom.get_trimesh()
                                v_start = geom.vvert_start + custom_offset
                                v_end = geom.vvert_end + custom_offset
                                for i_b in geom_envs_idx:
                                    env_i = self.rendered_envs_idx.index(i_b)
                                    active_envs = np.zeros(n_rendered_envs, dtype=bool)
                                    active_envs[env_i] = True
                                    node = self.add_node(
                                        pyrender.Mesh.from_trimesh(
                                            mesh=mesh,
                                            poses=identity_poses,
                                            smooth=geom.surface.smooth,
                                            double_sided=geom.surface.double_sided,
                                            env_shared=False,
                                            active_envs=active_envs,
                                        ),
                                    )
                                    geom_vverts = vverts[env_i, v_start:v_end, :] + envs_offset[i_b]
                                    node.mesh.primitives[0].positions = self._scene.reorder_vertices(
                                        node, geom_vverts.astype(np.float32)
                                    )
                                    self.vverts_nodes[(i_b, geom.uid)] = node
                                    if self.segmentation_level == "geom":
                                        seg_key = (geom.entity.idx, geom.link.idx, geom.idx)
                                    elif self.segmentation_level == "link":
                                        seg_key = (geom.entity.idx, geom.link.idx)
                                    elif self.segmentation_level == "entity":
                                        seg_key = geom.entity.idx
                                    else:
                                        gs.raise_exception(f"Unsupported segmentation level: {self.segmentation_level}")
                                    self.create_node_seg(seg_key, node)
                            self._per_env_vverts_entity_uids.add(entity.uid)

                        vverts = qd_to_numpy(solver.vverts_state.pos, self.rendered_envs_idx, transpose=True)
                        envs_offset = self.scene.envs_offset
                        custom_offset = entity._custom_vvert_start - entity._vvert_start
                        for geom in entity.vgeoms:
                            geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                            if len(geom_envs_idx) == 0:
                                continue
                            v_start = geom.vvert_start + custom_offset
                            v_end = geom.vvert_end + custom_offset
                            for env_i, i_b in enumerate(self.rendered_envs_idx):
                                if i_b not in geom_envs_idx:
                                    continue
                                node = self.vverts_nodes[(i_b, geom.uid)]
                                geom_vverts = vverts[env_i, v_start:v_end, :] + envs_offset[i_b]
                                update_data = self._scene.reorder_vertices(node, geom_vverts.astype(np.float32))
                                self.jit.update_buffer(self._scene.get_buffer_id(node, "pos"), update_data)
                                normal_data = self.jit.update_normal(node, update_data)
                                if normal_data is not None:
                                    self.jit.update_buffer(self._scene.get_buffer_id(node, "normal"), normal_data)
                        continue
                else:
                    geoms = entity.geoms
                    geoms_T = solver._geoms_render_T

                for geom in geoms:
                    # Skip geoms that weren't added - in heterogeneous simulation, some geoms
                    # may not be rendered in any of the requested environments
                    if geom.uid not in self.rigid_nodes:
                        continue

                    # For heterogeneous simulation, filter envs based on geom's assigned environments
                    geom_envs_idx = self._get_geom_active_envs_idx(geom, self.rendered_envs_idx)
                    if len(geom_envs_idx) == 0:
                        continue

                    # Mirror on_rigid: full per-env poses for env-masked variants, compacted otherwise.
                    if len(geom_envs_idx) < len(self.rendered_envs_idx):
                        geom_T = geoms_T[geom.idx][self.rendered_envs_idx]
                    else:
                        geom_T = geoms_T[geom.idx][geom_envs_idx]

                    # Keep single-instance for z-axis normal planes (see on_rigid)
                    if isinstance(entity.main_morph, gs.morphs.Plane):
                        plane_normal, plane_size = entity.main_morph.normal, entity.main_morph.plane_size
                        if (
                            abs(plane_normal[0]) < gs.EPS
                            and abs(plane_normal[1]) < gs.EPS
                            and self.scene.env_spacing[0] < plane_size[0]
                            and self.scene.env_spacing[1] < plane_size[1]
                        ):
                            geom_T = geom_T[:1]

                    node = self.rigid_nodes[geom.uid]
                    node.mesh._bounds = None
                    node.mesh.primitives[0].poses = geom_T
                    self.jit.update_buffer(self._scene.get_buffer_id(node, "model"), geom_T.transpose((0, 2, 1)))
                    if isinstance(entity._morph, gs.morphs.Plane):
                        self.set_reflection_mat(geom_T)

    def update_contact(self):
        if self.sim.rigid_solver.is_active and any(link.visualize_contact for link in self.sim.rigid_solver.links):
            # Extract all contact information at once
            contacts_info_all = self.sim.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
            geoms_aabb = qd_to_numpy(self.sim.rigid_solver.geoms_init_AABB)

            for env_i, batch_idx in enumerate(self.rendered_envs_idx):
                if self.sim.rigid_solver.n_envs > 0:
                    contacts_info = {key: value[batch_idx] for key, value in contacts_info_all.items()}
                else:
                    contacts_info = contacts_info_all

                n_contacts = len(contacts_info["geom_a"])
                if n_contacts == 0:
                    continue

                ga_aabb = geoms_aabb[contacts_info["geom_a"]]
                gb_aabb = geoms_aabb[contacts_info["geom_b"]]
                ga_aabb_size = np.linalg.norm(ga_aabb[:, -1] - ga_aabb[:, 0], axis=1)
                gb_aabb_size = np.linalg.norm(gb_aabb[:, -1] - gb_aabb[:, 0], axis=1)
                arrow_scale = np.minimum(ga_aabb_size, gb_aabb_size)
                radius = np.minimum(arrow_scale * 0.04, 0.005)
                contact_pos = contacts_info["position"] + self.scene.envs_offset[batch_idx]
                contact_normal_scaled = contacts_info["normal"] * arrow_scale[:, None]
                contact_force = contacts_info["force"]

                for i_c in range(n_contacts):
                    for link_idx, sign in ((contacts_info["link_a"][i_c], -1), (contacts_info["link_b"][i_c], 1)):
                        if self.sim.rigid_solver.links[link_idx].visualize_contact:
                            self.draw_contact_arrow(
                                pos=contact_pos[i_c],
                                radius=radius[i_c],
                                force=sign * contact_force[i_c],
                                env_idx=env_i,
                            )
                            self.draw_debug_arrow(
                                pos=contact_pos[i_c],
                                radius=radius[i_c],
                                vec=-sign * contact_normal_scaled[i_c],
                                color=(0.9, 0.0, 0.8, 1.0),
                                persistent=False,
                                env_idx=env_i,
                            )

    def on_mpm(self):
        if self.sim.mpm_solver.is_active:
            for mpm_entity in self.sim.mpm_solver.entities:
                if mpm_entity.surface.vis_mode in ("recon", "visual"):
                    self.add_dynamic_node(mpm_entity, None)
                elif mpm_entity.surface.vis_mode == "particle":
                    for idx in self.rendered_envs_idx:
                        mesh = mu.create_sphere(
                            self.sim.mpm_solver.particle_radius * self.particle_size_scale, subdivisions=1
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))

                        tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = mpm_entity.init_particles
                        self.add_static_node(
                            mpm_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs), i_b=idx
                        )

            # boundary
            if self.visualize_mpm_boundary:
                for idx in self.rendered_envs_idx:
                    offset = self.scene.envs_offset[idx]
                    lower = self.sim.mpm_solver.boundary.lower + offset
                    upper = self.sim.mpm_solver.boundary.upper + offset
                    self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mu.create_box(
                                bounds=np.array([lower, upper], dtype=np.float32),
                                wireframe=True,
                                color=(1.0, 1.0, 0.0, 1.0),
                            ),
                            smooth=True,
                        )
                    )

    def update_mpm(self):
        if self.sim.mpm_solver.is_active:
            particles_all = qd_to_numpy(self.sim.mpm_solver.particles_render.pos) + self.scene.envs_offset
            active_all = qd_to_numpy(self.sim.mpm_solver.particles_render.active).astype(dtype=np.bool_, copy=False)
            vverts_all = qd_to_numpy(self.sim.mpm_solver.vverts_render.pos) + self.scene.envs_offset
            for mpm_entity in self.sim.mpm_solver.entities:
                for idx in self.rendered_envs_idx:
                    if mpm_entity.surface.vis_mode == "recon":
                        mesh = pu.particles_to_mesh(
                            positions=particles_all[mpm_entity.particle_start : mpm_entity.particle_end, idx][
                                active_all[mpm_entity.particle_start : mpm_entity.particle_end, idx]
                            ],
                            radius=self.sim.mpm_solver.particle_radius,
                            backend=mpm_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(mpm_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(mpm_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True))
                    elif mpm_entity.surface.vis_mode == "particle":
                        tfs = np.tile(np.eye(4), (mpm_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = particles_all[mpm_entity.particle_start : mpm_entity.particle_end, idx]

                        buf_id = self._scene.get_buffer_id(self.static_nodes[(idx, mpm_entity.uid)], "model")
                        self.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

                    elif mpm_entity.surface.vis_mode == "visual":
                        mpm_entity._vmesh.trimesh.vertices = vverts_all[
                            mpm_entity.vvert_start : mpm_entity.vvert_end, idx
                        ]
                        self.add_dynamic_node(
                            mpm_entity,
                            pyrender.Mesh.from_trimesh(mpm_entity.vmesh.trimesh, smooth=mpm_entity.surface.smooth),
                        )

    def on_sph(self):
        if self.sim.sph_solver.is_active:
            for sph_entity in self.sim.sph_solver.entities:
                if sph_entity.surface.vis_mode == "recon":
                    self.add_dynamic_node(sph_entity, None)
                elif sph_entity.surface.vis_mode == "particle":
                    for idx in self.rendered_envs_idx:
                        mesh = mu.create_sphere(
                            self.sim.sph_solver.particle_radius * self.particle_size_scale, subdivisions=1
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))

                        tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = sph_entity.init_particles
                        self.add_static_node(
                            sph_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs), i_b=idx
                        )

            # boundary
            if self.visualize_sph_boundary:
                for idx in self.rendered_envs_idx:
                    offset = self.scene.envs_offset[idx]
                    lower = self.sim.sph_solver.boundary.lower + offset
                    upper = self.sim.sph_solver.boundary.upper + offset
                    self.add_node(
                        pyrender.Mesh.from_trimesh(
                            mu.create_box(
                                bounds=np.array([lower, upper], dtype=np.float32),
                                wireframe=True,
                                color=(0.0, 1.0, 1.0, 1.0),
                            ),
                            smooth=True,
                        )
                    )

    def update_sph(self):
        if self.sim.sph_solver.is_active:
            particles_all = qd_to_numpy(self.sim.sph_solver.particles_render.pos) + self.scene.envs_offset
            active_all = qd_to_numpy(self.sim.sph_solver.particles_render.active).astype(dtype=np.bool_, copy=False)

            for sph_entity in self.sim.sph_solver.entities:
                for idx in self.rendered_envs_idx:
                    if sph_entity.surface.vis_mode == "recon":
                        mesh = pu.particles_to_mesh(
                            positions=particles_all[sph_entity.particle_start : sph_entity.particle_end, idx][
                                active_all[sph_entity.particle_start : sph_entity.particle_end, idx]
                            ],
                            radius=self.sim.sph_solver.particle_radius,
                            backend=sph_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(sph_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(sph_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True))
                    elif sph_entity.surface.vis_mode == "particle":
                        tfs = np.tile(np.eye(4), (sph_entity.n_particles, 1, 1))
                        tfs[:, :3, 3] = particles_all[sph_entity.particle_start : sph_entity.particle_end, idx]

                        buf_id = self._scene.get_buffer_id(self.static_nodes[(idx, sph_entity.uid)], "model")
                        self.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

    def on_pbd(self):
        if self.sim.pbd_solver.is_active:
            for pbd_entity in self.sim.pbd_solver.entities:
                if pbd_entity.surface.vis_mode == "visual":
                    # Apply surface visual with UVs to the trimesh
                    pbd_entity.vmesh.trimesh.visual = mu.surface_uvs_to_trimesh_visual(
                        pbd_entity.surface, uvs=pbd_entity.vmesh.uvs, n_verts=len(pbd_entity.vmesh.trimesh.vertices)
                    )
                for idx in self.rendered_envs_idx:
                    if pbd_entity.surface.vis_mode == "recon":
                        self.add_dynamic_node(pbd_entity, None)
                    elif pbd_entity.surface.vis_mode == "particle":
                        if self.render_particle_as == "sphere":
                            mesh = mu.create_sphere(
                                self.sim.pbd_solver.particle_radius * self.particle_size_scale, subdivisions=1
                            )
                            mesh.visual = mu.surface_uvs_to_trimesh_visual(
                                pbd_entity.surface, n_verts=len(mesh.vertices)
                            )
                            tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                            tfs[:, :3, 3] = pbd_entity.init_particles
                            self.add_static_node(
                                pbd_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True, poses=tfs), i_b=idx
                            )
                        elif self.render_particle_as == "tet":
                            mesh = mu.create_tets_mesh(
                                pbd_entity.n_particles, self.sim.pbd_solver.particle_radius * self.particle_size_scale
                            )
                            mesh.visual = mu.surface_uvs_to_trimesh_visual(
                                pbd_entity.surface, n_verts=len(mesh.vertices)
                            )
                            pbd_entity._tets_mesh = mesh
                            self.add_static_node(pbd_entity, pyrender.Mesh.from_trimesh(mesh, smooth=False), i_b=idx)
                    elif pbd_entity.surface.vis_mode == "visual":
                        self.add_static_node(
                            pbd_entity,
                            pyrender.Mesh.from_trimesh(
                                pbd_entity.vmesh.trimesh,
                                smooth=pbd_entity.surface.smooth,
                                double_sided=pbd_entity._surface.double_sided,
                            ),
                            i_b=idx,
                        )

            # boundary
            if self.visualize_pbd_boundary:
                self.add_node(
                    pyrender.Mesh.from_trimesh(
                        mu.create_box(
                            bounds=np.array(
                                [self.sim.pbd_solver.boundary.lower, self.sim.pbd_solver.boundary.upper],
                                dtype=np.float32,
                            ),
                            wireframe=True,
                            color=(0.0, 1.0, 1.0, 1.0),
                        ),
                        smooth=True,
                    )
                )

    def update_pbd(self):
        if self.sim.pbd_solver.is_active:
            particles_all = qd_to_numpy(self.sim.pbd_solver.particles_render.pos) + self.scene.envs_offset
            particles_vel_all = qd_to_numpy(self.sim.pbd_solver.particles_render.vel)
            active_all = qd_to_numpy(self.sim.pbd_solver.particles_render.active).astype(dtype=np.bool_, copy=False)
            vverts_all = qd_to_numpy(self.sim.pbd_solver.vverts_render.pos) + self.scene.envs_offset
            for pbd_entity in self.sim.pbd_solver.entities:
                for idx in self.rendered_envs_idx:
                    particles_env = particles_all[:, idx]
                    particles_vel_env = particles_vel_all[:, idx]
                    active_env = active_all[:, idx]
                    vverts_env = vverts_all[:, idx]

                    if pbd_entity.surface.vis_mode == "recon":
                        positions = particles_env[pbd_entity.particle_start : pbd_entity.particle_end][
                            active_env[pbd_entity.particle_start : pbd_entity.particle_end]
                        ]
                        mesh = pu.particles_to_mesh(
                            positions=positions,
                            radius=self.sim.mpm_solver.particle_radius,
                            backend=pbd_entity.surface.recon_backend,
                        )
                        mesh.visual = mu.surface_uvs_to_trimesh_visual(pbd_entity.surface, n_verts=len(mesh.vertices))
                        self.add_dynamic_node(pbd_entity, pyrender.Mesh.from_trimesh(mesh, smooth=True))

                    # TODO: need to support multi-env visulaization for tet mode (it's using static node)
                    elif pbd_entity.surface.vis_mode == "particle":
                        if self.render_particle_as == "sphere":
                            tfs = np.tile(np.eye(4), (pbd_entity.n_particles, 1, 1))
                            tfs[:, :3, 3] = particles_env[pbd_entity.particle_start : pbd_entity.particle_end]

                            buf_id = self._scene.get_buffer_id(self.static_nodes[(idx, pbd_entity.uid)], "model")
                            self.jit.update_buffer(buf_id, tfs.transpose((0, 2, 1)))

                        elif self.render_particle_as == "tet":
                            new_verts = mu.transform_tets_mesh_verts(
                                pbd_entity._tets_mesh.vertices,
                                positions=particles_env[pbd_entity.particle_start : pbd_entity.particle_end],
                                zs=particles_vel_env[pbd_entity.particle_start : pbd_entity.particle_end],
                            )
                            node = self.static_nodes[(idx, pbd_entity.uid)]
                            update_data = self._scene.reorder_vertices(node, new_verts.astype(np.float32))
                            self.jit.update_buffer(
                                self._scene.get_buffer_id(node, "pos"),
                                update_data,
                                node=node,
                                buffer_name="pos",
                            )
                            normal_data = self.jit.update_normal(node, update_data)
                            if normal_data is not None:
                                self.jit.update_buffer(self._scene.get_buffer_id(node, "normal"), normal_data)
                    elif pbd_entity.surface.vis_mode == "visual":
                        vverts = vverts_env[pbd_entity.vvert_start : pbd_entity.vvert_end]
                        node = self.static_nodes[(idx, pbd_entity.uid)]
                        update_data = self._scene.reorder_vertices(node, vverts.astype(np.float32))
                        self.jit.update_buffer(
                            self._scene.get_buffer_id(node, "pos"),
                            update_data,
                            node=node,
                            buffer_name="pos",
                        )
                        normal_data = self.jit.update_normal(node, update_data)
                        if normal_data is not None:
                            self.jit.update_buffer(
                                self._scene.get_buffer_id(node, "normal"),
                                normal_data,
                                node=node,
                                buffer_name="normal",
                            )

    def on_fem(self):
        if self.sim.fem_solver.is_active:
            vertices_qd, triangles_qd, uvs_qd = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vertices_all = qd_to_numpy(vertices_qd)
            triangles_all = qd_to_numpy(triangles_qd).reshape((-1, 3))
            uvs_all = qd_to_numpy(uvs_qd)

            for fem_entity in self.sim.fem_solver.entities:
                if fem_entity.surface.vis_mode == "visual":
                    triangles = (
                        triangles_all[fem_entity.s_start : (fem_entity.s_start + fem_entity.n_surfaces)]
                        - fem_entity.v_start
                    )
                    surf_idx, inv = np.unique(triangles.flat, return_inverse=True)
                    if surf_idx.shape[0] != fem_entity.n_surface_vertices:
                        gs.raise_exception(
                            f"FEM entity {fem_entity.uid} surface vertex map has {surf_idx.shape[0]} vertices, "
                            f"expected {fem_entity.n_surface_vertices}."
                        )
                    surf_idx = np.ascontiguousarray(surf_idx)
                    triangles_reindexed = inv.reshape(triangles.shape)
                    self._fem_surface_triangles[fem_entity.uid] = np.ascontiguousarray(triangles, dtype=np.int64)
                    for idx in self.rendered_envs_idx:
                        entity_vertices = vertices_all[
                            fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices, idx
                        ]
                        node_key = (idx, fem_entity.uid)
                        rgb_node = None
                        if not bool(getattr(fem_entity, "_rgb_visualization_disabled", False)):
                            self._fem_surface_vertex_indices[node_key] = surf_idx.copy()
                            vertices = entity_vertices[surf_idx]
                            uvs = uvs_all[fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices][surf_idx]
                            mesh = trimesh.Trimesh(vertices, triangles_reindexed, process=False)
                            mesh.visual = mu.surface_uvs_to_trimesh_visual(
                                fem_entity.surface, uvs=uvs, n_verts=fem_entity.n_surface_vertices
                            )
                            self.add_static_node(
                                fem_entity,
                                pyrender.Mesh.from_trimesh(
                                    mesh,
                                    smooth=fem_entity.surface.smooth,
                                    double_sided=fem_entity.surface.double_sided,
                                ),
                                i_b=idx,
                            )
                            rgb_node = self.static_nodes[node_key]

                        part_config = getattr(fem_entity, "_part_segmentation_config", None)
                        if part_config is not None:
                            if rgb_node is not None:
                                self.remove_node_seg(rgb_node)
                                self.rgb_only_nodes.add(rgb_node)
                            labels = np.asarray(fem_entity.surface_primitive_part_labels, dtype=np.int64)
                            if labels.shape != (triangles.shape[0],):
                                gs.raise_exception(
                                    f"FEM entity {fem_entity.uid} has {labels.shape} surface part labels, "
                                    f"expected {(triangles.shape[0],)}."
                                )
                            parts = {int(part["part_id"]): part for part in part_config["parts"]}
                            unknown = sorted(set(int(value) for value in np.unique(labels)) - set(parts))
                            if unknown:
                                gs.raise_exception(
                                    f"FEM entity {fem_entity.uid} surface labels reference unknown part ids {unknown}."
                                )
                            corner_normals = _crease_surface_corner_normals(entity_vertices, triangles)
                            for part_id in sorted(parts):
                                part_face_indices = np.flatnonzero(labels == part_id)
                                part_triangles = triangles[part_face_indices]
                                if part_triangles.shape[0] == 0:
                                    continue
                                part_vertices = np.ascontiguousarray(part_triangles.reshape(-1), dtype=np.int64)
                                part_faces = np.arange(len(part_vertices), dtype=np.int64).reshape((-1, 3))
                                part_corner_indices = np.ascontiguousarray(
                                    (part_face_indices[:, None] * 3 + np.arange(3)[None, :]).reshape(-1),
                                    dtype=np.int64,
                                )
                                mesh = trimesh.Trimesh(
                                    entity_vertices[part_vertices],
                                    part_faces,
                                    vertex_normals=corner_normals.reshape((-1, 3))[part_corner_indices],
                                    process=False,
                                )
                                color = np.asarray(parts[part_id]["part_color_rgb"], dtype=np.float32) / 255.0
                                material = pyrender.MetallicRoughnessMaterial(
                                    alphaMode="OPAQUE",
                                    baseColorFactor=(*color.tolist(), 1.0),
                                    metallicFactor=PART_SHADED_METALLIC,
                                    roughnessFactor=PART_SHADED_ROUGHNESS,
                                )
                                part_mesh = pyrender.Mesh.from_trimesh(
                                    mesh,
                                    smooth=True,
                                    double_sided=fem_entity.surface.double_sided,
                                    material=material,
                                )
                                primitive = part_mesh.primitives[0]
                                if primitive.indices is None or primitive.vertex_mapping is not None:
                                    gs.raise_exception(
                                        f"FEM entity {fem_entity.uid} part {part_id} segmentation mesh is not indexed."
                                    )
                                if primitive.positions.shape[0] != part_vertices.shape[0] or primitive.indices.shape != part_faces.shape:
                                    gs.raise_exception(
                                        f"FEM entity {fem_entity.uid} part {part_id} indexed mesh changed vertex/face counts."
                                    )
                                part_node = self.add_node(part_mesh)
                                part_key = (idx, fem_entity.uid, part_id)
                                self.part_segmentation_nodes[part_key] = part_node
                                self._part_segmentation_vertex_indices[part_key] = part_vertices
                                self._part_shaded_corner_indices[part_key] = part_corner_indices
                                self.segmentation_only_nodes.add(part_node)
                                self.part_segmentation_indexed_counts[part_key] = {
                                    "vertex_count": int(primitive.positions.shape[0]),
                                    "face_count": int(primitive.indices.shape[0]),
                                    "index_count": int(primitive.indices.size),
                                    "expanded_vertex_count": int(primitive.positions.shape[0]),
                                }
                                seg_key = ("hag4r_part", fem_entity.uid, part_id)
                                self.create_node_seg(seg_key, part_node)
                                seg_idxc = self.seg_color_map.key_map[seg_key]
                                self.part_segmentation_palette_by_idxc[seg_idxc] = np.asarray(
                                    parts[part_id]["part_color_rgb"], dtype=np.uint8
                                )

    def update_fem(self, render_pass="rgb"):
        if self.sim.fem_solver.is_active:
            vertices_all, _, _ = self.sim.fem_solver.get_state_render(self.sim.cur_substep_local)
            vertices_all = vertices_all.to_numpy(dtype=gs.np_float)

            if render_pass in {"part_segmentation", "part_shaded"}:
                uploaded_bytes = 0
                normal_upload_bytes = 0
                active_nodes = 0
                for fem_entity in self.sim.fem_solver.entities:
                    if fem_entity.surface.vis_mode != "visual":
                        continue
                    if getattr(fem_entity, "_part_segmentation_config", None) is None:
                        gs.raise_exception(
                            f"FEM entity {fem_entity.uid} does not define a part segmentation contract."
                        )
                    for idx in self.rendered_envs_idx:
                        entity_vertices = vertices_all[
                            fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices, idx
                        ]
                        corner_normals = (
                            _crease_surface_corner_normals(
                                entity_vertices, self._fem_surface_triangles[fem_entity.uid]
                            ).reshape((-1, 3))
                            if render_pass == "part_shaded"
                            else None
                        )
                        part_ids = sorted(
                            key[2]
                            for key in self.part_segmentation_nodes
                            if key[0] == idx and key[1] == fem_entity.uid
                        )
                        for part_id in part_ids:
                            part_key = (idx, fem_entity.uid, part_id)
                            node = self.part_segmentation_nodes[part_key]
                            vertices = entity_vertices[self._part_segmentation_vertex_indices[part_key]]
                            update_data = self._scene.reorder_vertices(node, vertices)
                            self.jit.update_buffer(
                                self._scene.get_buffer_id(node, "pos"),
                                update_data,
                                node=node,
                                buffer_name="pos",
                            )
                            uploaded_bytes += int(update_data.nbytes)
                            if corner_normals is not None:
                                normal_data = corner_normals[self._part_shaded_corner_indices[part_key]]
                                self.jit.update_buffer(
                                    self._scene.get_buffer_id(node, "normal"),
                                    normal_data,
                                    node=node,
                                    buffer_name="normal",
                                )
                                normal_upload_bytes += int(normal_data.nbytes)
                            active_nodes += 1
                self.last_part_segmentation_update = {
                    "fem_state_fetch_count": 1,
                    "active_part_node_count": active_nodes,
                    "position_upload_bytes": uploaded_bytes,
                    "normal_upload_bytes": normal_upload_bytes,
                    "rgb_fem_state_fetch_count": int(render_pass == "part_shaded"),
                    "rgb_position_upload_bytes": uploaded_bytes if render_pass == "part_shaded" else 0,
                    "rgb_normal_upload_bytes": normal_upload_bytes,
                    "rgb_update_node_count": active_nodes if render_pass == "part_shaded" else 0,
                }
                self.last_render_update_stats = dict(self.last_part_segmentation_update)
                return

            uploaded_bytes = 0
            active_nodes = 0
            for fem_entity in self.sim.fem_solver.entities:
                if bool(getattr(fem_entity, "_rgb_visualization_disabled", False)):
                    continue
                if fem_entity.surface.vis_mode == "visual":
                    for idx in self.rendered_envs_idx:
                        node_key = (idx, fem_entity.uid)
                        surf_idx = self._fem_surface_vertex_indices[node_key]
                        vertices = vertices_all[fem_entity.v_start : fem_entity.v_start + fem_entity.n_vertices, idx]
                        vertices = vertices[surf_idx]

                        node = self.static_nodes[node_key]
                        primitive = node.mesh.primitives[0]
                        if primitive.vertex_mapping is None:
                            expected_logical_n = primitive.positions.shape[0]
                        else:
                            expected_logical_n = (
                                int(primitive.vertex_mapping.max()) + 1 if primitive.vertex_mapping.size else 0
                            )
                        if vertices.shape[0] != expected_logical_n:
                            gs.raise_exception(
                                f"FEM entity {fem_entity.uid} update has {vertices.shape[0]} logical vertices, "
                                f"expected {expected_logical_n} for rasterizer node {node_key}."
                            )
                        update_data = self._scene.reorder_vertices(node, vertices)
                        if update_data.shape != primitive.positions.shape:
                            gs.raise_exception(
                                f"FEM entity {fem_entity.uid} update buffer shape {update_data.shape} does not match "
                                f"rasterizer primitive positions shape {primitive.positions.shape} for node {node_key}."
                            )
                        self.jit.update_buffer(
                            self._scene.get_buffer_id(node, "pos"),
                            update_data,
                            node=node,
                            buffer_name="pos",
                        )
                        uploaded_bytes += int(update_data.nbytes)
                        active_nodes += 1
                        normal_data = self.jit.update_normal(node, update_data)
                        if normal_data is not None:
                            self.jit.update_buffer(
                                self._scene.get_buffer_id(node, "normal"),
                                normal_data,
                                node=node,
                                buffer_name="normal",
                            )
            self.last_render_update_stats = {
                "fem_state_fetch_count": 1,
                "active_rgb_node_count": active_nodes,
                "position_upload_bytes": uploaded_bytes,
                "rgb_fem_state_fetch_count": 1,
                "rgb_position_upload_bytes": uploaded_bytes,
                "rgb_update_node_count": active_nodes,
            }

    def update_sensors(self):
        self.sim._sensor_manager.draw_debug(self)

    def on_lights(self):
        for light in self.lights:
            self.add_light(light)

    def draw_debug_line(self, start, end, radius=0.002, color=(1.0, 0.0, 0.0, 1.0)):
        mesh = mu.create_line(
            tensor_to_array(start, dtype=np.float32), tensor_to_array(end, dtype=np.float32), radius, color
        )
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_line_{gs.UID()}", is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_arrow(
        self, pos, vec=(0.0, 0.0, 1.0), radius=0.006, color=(1.0, 0.0, 0.0, 0.5), persistent=True, env_idx=None
    ):
        vec = tensor_to_array(vec, dtype=np.float32)
        length = np.linalg.norm(vec)
        if length > gs.EPS:
            mesh = mu.create_arrow(length=length, radius=radius, body_color=color, head_color=color)

            # Build per-env poses when env_idx is specified for env_separate rendering
            pos = tensor_to_array(pos)
            if env_idx is not None and self.env_separate_rigid:
                poses = np.zeros((len(self.rendered_envs_idx), 4, 4), dtype=np.float32)
                gu.trans_R_to_T(pos, gu.z_up_to_R(vec), out=poses[env_idx])
                env_shared = False
            else:
                poses = np.zeros((1, 4, 4), dtype=np.float32)
                gu.trans_R_to_T(pos, gu.z_up_to_R(vec), out=poses[0])
                env_shared = True

            node = pyrender.Mesh.from_trimesh(
                mesh, name=f"debug_arrow_{gs.UID()}", poses=poses, env_shared=env_shared, is_marker=True
            )
            if persistent:
                self.add_external_node(node)
            else:
                self.add_dynamic_node(None, node)
            return node

    def draw_debug_frame(self, T, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        mesh = trimesh.creation.axis(origin_size=origin_size, axis_radius=axis_radius, axis_length=axis_length)
        if color is not None:
            visual = trimesh.visual.ColorVisuals()
            visual._data["vertex_colors"] = np.tile(mu.color_f32_to_u8(color), (len(mesh.vertices), 1))
            mesh.visual = visual

        n_envs = len(self.rendered_envs_idx)
        poses = tensor_to_array(T)
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_frame_{gs.UID()}", poses=poses, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_frames(self, poses, axis_length=1.0, origin_size=0.015, axis_radius=0.01, color=None):
        mesh = trimesh.creation.axis(origin_size=origin_size, axis_radius=axis_radius, axis_length=axis_length)
        if color is not None:
            visual = trimesh.visual.ColorVisuals()
            visual._data["vertex_colors"] = np.tile(mu.color_f32_to_u8(color), (len(mesh.vertices), 1))
            mesh.visual = visual
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_frame_{gs.UID()}", poses=poses, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_mesh(self, mesh, pos=np.zeros(3), T=None):
        n_envs = len(self.rendered_envs_idx)
        if T is None:
            T = gu.trans_to_T(pos)
        poses = tensor_to_array(T)
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_mesh_{gs.UID()}", poses=poses, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_contact_arrow(self, pos, radius=0.005, force=(0, 0, 1), color=(0.0, 0.9, 0.8, 1.0), env_idx=None):
        length = (tensor_to_array(force) * self.contact_force_scale,)
        self.draw_debug_arrow(pos, length, radius, color=color, persistent=False, env_idx=env_idx)

    def draw_debug_sphere(self, pos, radius=0.01, color=(1.0, 0.0, 0.0, 0.5), persistent=True):
        mesh = mu.create_sphere(radius=radius, color=color)

        n_envs = len(self.rendered_envs_idx)
        poses = gu.trans_to_T(tensor_to_array(pos))
        if poses.ndim != 3:
            poses = np.tile(poses[np.newaxis], (n_envs, 1, 1))
        assert len(poses) == n_envs, "Inconsistent batch size."

        node = pyrender.Mesh.from_trimesh(
            mesh, name=f"debug_sphere_{gs.UID()}", smooth=True, poses=poses, is_marker=True
        )
        if persistent:
            self.add_external_node(node)
        else:
            self.add_dynamic_node(None, node)
        return node

    def draw_debug_pyramid(self, T, base_width=0.05, base_height=0.05, height=0.05, color=(1.0, 1.0, 1.0, 0.5)):
        """
        Draw a debug pyramid representing a camera frustum.
        Parameters
        ----------
        T: array-like, shape (4, 4), optional
            The transformation matrix.
        base_width: float
            The width of the pyramid base.
        base_height: float
            The height of the pyramid base.
        height: float
            The height of the pyramid (distance from apex to base).
        color: RGBA color tuple
        """
        T = tensor_to_array(T, dtype=np.float32)
        right = T[:3, 0]
        up = T[:3, 1]
        forward = -T[:3, 2]

        base_center = forward * height
        half_width = base_width / 2
        half_height = base_height / 2
        vertices = np.array(
            [
                [0, 0, 0],  # apex
                base_center + half_width * right + half_height * up,  # top-right
                base_center - half_width * right + half_height * up,  # top-left
                base_center - half_width * right - half_height * up,  # bottom-left
                base_center + half_width * right - half_height * up,  # bottom-right
            ]
        )
        faces = np.array(
            [
                # Base (2 triangles) - facing away from apex
                [1, 2, 3],
                [1, 3, 4],
                # Sides (4 triangles from apex to base edges)
                [0, 2, 1],  # left side
                [0, 3, 2],  # back side
                [0, 4, 3],  # right side
                [0, 1, 4],  # front side
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.face_colors = np.tile(mu.color_f32_to_u8(color), (len(faces), 1))

        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_pyramid_{gs.UID()}", smooth=False, is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_spheres(self, poss, radius=0.01, color=(1.0, 0.0, 0.0, 0.5), persistent=True):
        mesh = mu.create_sphere(radius=radius, color=color)
        poses = gu.trans_to_T(tensor_to_array(poss))
        node = pyrender.Mesh.from_trimesh(
            mesh, name=f"debug_spheres_{gs.UID()}", smooth=True, poses=poses, is_marker=True
        )
        if persistent:
            self.add_external_node(node)
        else:
            self.add_dynamic_node(None, node)
        return node

    def draw_debug_box(self, bounds, color=(1.0, 0.0, 0.0, 1.0), wireframe=True, wireframe_radius=0.002):
        bounds = tensor_to_array(bounds)
        mesh = mu.create_box(
            bounds=bounds,
            wireframe=wireframe,
            wireframe_radius=wireframe_radius,
            color=color,
        )
        node = pyrender.Mesh.from_trimesh(mesh, name=f"debug_box_{gs.UID()}", is_marker=True)
        self.add_external_node(node)
        return node

    def draw_debug_points(self, poss, colors=(1.0, 0.0, 0.0, 0.5)):
        poss = tensor_to_array(poss)
        colors = tensor_to_array(colors)
        if len(colors.shape) == 1:
            colors = np.tile(colors, [len(poss), 1])
        elif len(colors.shape) == 2:
            assert colors.shape[0] == len(poss)

        node = pyrender.Mesh.from_points(poss, name=f"debug_box_{gs.UID()}", colors=colors, is_marker=True)
        self.add_external_node(node)
        return node

    def update_debug_objects(self, objs, poses):
        n_envs = len(self.rendered_envs_idx)
        for obj, pose in zip(objs, poses):
            if not any(
                obj.name.startswith(prefix)
                for prefix in ("debug_sphere_", "debug_frame_", "debug_mesh_", "debug_arrow_")
            ):
                gs.raise_exception("This method is only supported by individual spheres, frames, meshes, and arrows.")
            pose = tensor_to_array(pose)
            if pose.ndim != 3:
                pose = np.tile(pose[np.newaxis], (n_envs, 1, 1))
            assert len(pose) == n_envs, "Inconsistent batch size."
            obj.primitives[0].poses = pose
            node = self.external_nodes[obj.name]
            self.jit.update_buffer(self._scene.get_buffer_id(node, "model"), pose.transpose((0, 2, 1)))

    def clear_debug_object(self, obj):
        self.clear_external_node(obj)

    def clear_debug_objects(self):
        self.clear_external_nodes()

    def update(self, force_render: bool = False, render_pass: str = "rgb"):
        if render_pass == "part_segmentation_current":
            return

        # Early return if already updated previously
        if not force_render and self._t >= self.scene._t and getattr(self, "_last_render_pass", None) == render_pass:
            return

        update_started = time.perf_counter()

        # Update current time right away
        self._t = self.scene._t
        self._last_render_pass = render_pass
        self.jit.reset_buffer_upload_stats(render_pass)

        if render_pass in {"part_segmentation", "part_shaded"}:
            # The Live diagnostic segmentation pass owns a deliberately tiny visual scene:
            # fixed part/context topology plus current FEM positions. Camera pose is updated
            # by Rasterizer.render_camera; unrelated solver and RGB debug branches stay cold.
            self.update_fem(render_pass=render_pass)
            if render_pass == "part_shaded":
                self._scene._bounds = None
            self.last_render_update_stats["scene_update_seconds"] = time.perf_counter() - update_started
            return

        # Remove up old dynamic nodes
        self.clear_dynamic_nodes(only_outdated=True)

        # Force updating rendering-only quantities that are not updated automatically during simulation
        self.visualizer.update_visual_states(force_render)

        # Reset scene bounds to trigger recomputation. They are involved in shadow map.
        self._scene._bounds = None

        self.update_link_frame()
        self.update_tool()
        self.update_rigid()
        self.update_contact()
        self.update_mpm()
        self.update_sph()
        self.update_pbd()
        self.update_fem(render_pass=render_pass)
        self.update_sensors()

        # Update camera fructum
        for camera in self.visualizer.cameras:
            self.update_camera_frustum(camera)
        self.last_render_update_stats["scene_update_seconds"] = time.perf_counter() - update_started

    def add_light(self, light):
        # light direction is light pose's -z frame
        if isinstance(light, gs.options.vis.DirectionalLight):
            pose = np.eye(4, dtype=np.float32)
            gu.z_up_to_R(-np.array(light.dir, dtype=np.float32), out=pose[:3, :3])
            self.add_node(pyrender.DirectionalLight(color=light.color, intensity=light.intensity), pose=pose)
        elif isinstance(light, gs.options.vis.PointLight):
            pose = gu.trans_to_T(np.array(light.pos, dtype=np.float32))
            self.add_node(pyrender.PointLight(color=light.color, intensity=light.intensity), pose=pose)
        else:
            gs.raise_exception(f"Unsupported light: {light}")

    def create_node_seg(self, seg_key, seg_node):
        seg_idxc = self.seg_color_map.seg_key_to_idxc(seg_key)
        if seg_node:
            self.seg_node_map[seg_node] = self.seg_idxc_to_idxc_rgb(seg_idxc)

    def remove_node_seg(self, seg_node):
        self.seg_node_map.pop(seg_node, None)

    def register_part_segmentation_context_entity(self, entity, *, kind, color):
        color = np.asarray(color, dtype=np.uint8)
        if color.shape != (3,):
            gs.raise_exception(f"Part segmentation context color for {kind!r} must have shape (3,).")
        for geom in entity.vgeoms:
            node = self.rigid_nodes.get(geom.uid)
            if node is None:
                continue
            self.remove_node_seg(node)
            seg_key = ("hag4r_context", kind, geom.uid)
            self.create_node_seg(seg_key, node)
            seg_idxc = self.seg_color_map.key_map[seg_key]
            self.part_segmentation_palette_by_idxc[seg_idxc] = color
            self.segmentation_only_nodes.add(node)

    def replace_part_segmentation_context_boxes(self, records):
        requested = {}
        for index, record in enumerate(records):
            bounds = np.asarray(record["bounds"], dtype=np.float32)
            color = np.asarray(record["color"], dtype=np.uint8)
            if bounds.shape != (2, 3) or np.any(bounds[0] > bounds[1]):
                gs.raise_exception("Part segmentation context box bounds must have shape (2, 3) and be ordered.")
            key = (str(record["kind"]), str(record.get("controller_id") or index))
            if key in requested:
                gs.raise_exception(f"Duplicate part segmentation context box key: {key}.")
            requested[key] = (bounds, color)

        removed = 0
        for key in set(self.part_segmentation_context_nodes) - set(requested):
            node = self.part_segmentation_context_nodes.pop(key)
            seg_key = self._part_segmentation_context_seg_keys.pop(key)
            seg_idxc = self.seg_color_map.remove_seg_key(seg_key)
            if seg_idxc is not None:
                self.part_segmentation_palette_by_idxc.pop(seg_idxc, None)
            self._part_segmentation_context_vertex_counts.pop(key, None)
            self._part_segmentation_context_colors.pop(key, None)
            self.jit.discard_buffer_updates_for_node(node)
            self.remove_node_seg(node)
            self.segmentation_only_nodes.discard(node)
            self.remove_node(node)
            removed += 1

        created = 0
        reused = 0
        position_upload_bytes = 0
        for key, (bounds, color) in requested.items():
            node = self.part_segmentation_context_nodes.get(key)
            if node is None:
                wireframe_mesh = _part_segmentation_wireframe_box_mesh(bounds)
                render_mesh = pyrender.Mesh.from_trimesh(wireframe_mesh, smooth=True, double_sided=True)
                primitive = render_mesh.primitives[0]
                if primitive.indices is None:
                    gs.raise_exception("Part segmentation wireframe context box must remain indexed.")
                node = self.add_node(render_mesh)
                self.part_segmentation_context_nodes[key] = node
                self._part_segmentation_context_vertex_counts[key] = int(primitive.positions.shape[0])
                self._part_segmentation_context_colors[key] = color.copy()
                self.segmentation_only_nodes.add(node)
                seg_key = ("hag4r_context_box", *key)
                self._part_segmentation_context_seg_keys[key] = seg_key
                self.create_node_seg(seg_key, node)
                seg_idxc = self.seg_color_map.key_map[seg_key]
                self.part_segmentation_palette_by_idxc[seg_idxc] = color.copy()
                created += 1
                continue

            if not np.array_equal(color, self._part_segmentation_context_colors[key]):
                gs.raise_exception(f"Part segmentation context box {key} changed its fixed palette color.")
            primitive = node.mesh.primitives[0]
            positions = np.asarray(_part_segmentation_wireframe_box_mesh(bounds).vertices, dtype=np.float32)
            if positions.shape != primitive.positions.shape:
                gs.raise_exception(f"Part segmentation context box {key} changed its fixed wireframe topology.")
            primitive.positions = positions
            self.jit.update_buffer(
                self._scene.get_buffer_id(node, "pos"),
                positions,
                node=node,
                buffer_name="pos",
            )
            position_upload_bytes += int(positions.nbytes)
            reused += 1

        self.last_part_segmentation_context_update = {
            "context_node_create_count": created,
            "context_node_reuse_count": reused,
            "context_node_remove_count": removed,
            "context_position_upload_bytes": position_upload_bytes,
            "context_wireframe_vertex_count": sum(self._part_segmentation_context_vertex_counts.values()),
            "context_wireframe_triangle_count": sum(
                int(node.mesh.primitives[0].indices.shape[0])
                for node in self.part_segmentation_context_nodes.values()
            ),
        }

    def seg_idxc_to_idxc_rgb(self, seg_idxc):
        seg_idxc_rgb = np.array(
            [
                (seg_idxc >> 16) & 0xFF,
                (seg_idxc >> 8) & 0xFF,
                seg_idxc & 0xFF,
            ],
            dtype=np.int32,
        )
        return seg_idxc_rgb

    def seg_idxc_rgb_arr_to_idxc_arr(self, seg_idxc_rgb_arr):
        # Combine the RGB components into a single integer
        seg_idxc_rgb_arr = seg_idxc_rgb_arr.astype(np.int64, copy=False)
        return seg_idxc_rgb_arr[..., 0] * (256 * 256) + seg_idxc_rgb_arr[..., 1] * 256 + seg_idxc_rgb_arr[..., 2]

    def colorize_seg_idxc_arr(self, seg_idxc_arr):
        return self.seg_color_map.colorize_seg_idxc_arr(seg_idxc_arr)

    @property
    def cameras(self):
        return self.visualizer.cameras

    @property
    def seg_idxc_map(self):
        return self.seg_color_map.idxc_map
