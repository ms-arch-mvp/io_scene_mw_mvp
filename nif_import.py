import bpy
import re

import timeit
import pathlib
import collections

import numpy as np
import numpy.linalg as la

from es3 import nif
from es3.utils.math import ID44, compose, decompose

from . import nif_utils
from . import nif_shader

from bpy_extras.io_utils import axis_conversion

biped_axis_correction = np.array(axis_conversion('-X', 'Z', 'Y', 'Z').to_4x4(), dtype="<f")
biped_axis_correction_inverse = la.inv(biped_axis_correction)

other_axis_correction = np.array(axis_conversion('Y', 'Z', '-Z', '-Y').to_4x4(), dtype="<f")
other_axis_correction_inverse = la.inv(other_axis_correction)

def _normalize_prefix(filename, max_length=3):
    # Lowercase prefix of up to max_length letters before an underscore or space
    for n in range(1, max_length + 1):
        if len(filename) >= n + 1 and filename[n] in ('_', ' ') and filename[:n].isalpha():
            return filename[:n].lower() + filename[n:]
    return filename

def normalize_path(path):
    if not path or not isinstance(path, str):
        return path

    # Replace all forward slashes with backslashes
    normalized = path.replace('/', '\\')

    # If there are no slashes, nothing path-specific to do
    if '\\' not in normalized:
        return normalized

    # Split into directory and filename; lowercase the directory
    parts = normalized.rsplit('\\', 1)
    if len(parts) == 2:
        directory, filename = parts
        return directory.lower() + '\\' + filename

    return normalized

def sanitize_name(name, normalize_prefix=False, normalize_prefix_max_length=3):
    # Remove surrogates and non-printable characters
    if not isinstance(name, str):
        name = str(name)
    # Remove surrogate pairs and non-characters
    name = re.sub(r'[\ud800-\udfff]', '', name)
    # Remove other non-printable/control characters
    name = ''.join(c for c in name if c.isprintable())
    # Always normalize path separators and directory casing
    name = normalize_path(name)
    # Optionally normalize the filename prefix
    if normalize_prefix:
        # Extract filename if path present, apply prefix normalization, reassemble
        if '\\' in name:
            parts = name.rsplit('\\', 1)
            parts[1] = _normalize_prefix(parts[1], max_length=normalize_prefix_max_length)
            name = '\\'.join(parts)
        else:
            name = _normalize_prefix(name, max_length=normalize_prefix_max_length)
    # Replace with '_' if empty
    return name or "Object"

def load(context, filepath, **config):
    """load a scene from a nif file"""

    print(f"Import File: {filepath}")
    time = timeit.default_timer()

    importer = Importer(filepath, config)
    importer.execute()

    time = timeit.default_timer() - time
    print(f"Import Done: {time:.4f} seconds")

    return {"FINISHED"}


class Importer:
    vertex_precision = 0.001
    attach_keyframe_data = False
    discard_root_transforms = True
    preserve_root_scale = False
    use_existing_materials = False
    ignore_collision_nodes = False
    ignore_custom_normals = False
    ignore_animations = False
    # new additions:
    ignore_armatures = False
    ignore_billboard_nodes = False
    ignore_particle_nodes = False
    ignore_emissive_color = False
    ignore_tri_shadow = False
    ignore_nodes = ""
    ignore_nodes_under_switches = ""
    filter_best_lod = False
    use_texture_fallbacks = True
    use_texture_path_in_material_name = False
    normalize_prefix = True
    normalize_prefix_root = True
    normalize_prefix_max_length = 3
    always_use_file_name_for_root_name = False
    proxy_mode = False
    
    def __init__(self, filepath, config):
        vars(self).update(config)
        self.nodes = {}
        self.materials = {}
        self.mesh_data = {}
        self.history = collections.defaultdict(set)
        self.armatures = collections.defaultdict(set)
        self.colliders = collections.defaultdict(set)
        self.active_collection = bpy.context.view_layer.active_layer_collection.collection
        self.filepath = pathlib.Path(filepath)
        self.ignored_nodes = {name.strip().lower() for name in str(self.ignore_nodes).split(",") if name.strip()}
        self.ignored_nodes_under_switches = {name.strip().lower() for name in str(self.ignore_nodes_under_switches).split(",") if name.strip()}

    def execute(self):
        data = nif.NiStream()
        data.load(self.filepath)
        data.merge_properties()
        self.discard_detached_skins(data)
        self.repair_scene(data)
        self.apply_hierarchy_scales(data)

        # fix transforms
        if self.discard_root_transforms:
            if self.preserve_root_scale:
                # Drop the root's world placement (translation + rotation) but
                # keep its per-axis scale. Baked actor exports encode the actor's
                # size on the root node as Morrowind's race weight/height model:
                # a non-uniform scale where X/Y is build/weight and Z is height.
                # Averaging it to a single factor (e.g. cube root) mixes width
                # into height and gets relative heights wrong, so keep all three
                # axes. Normal asset meshes have root scale 1.0, so this stays
                # equivalent to zeroing the matrix.
                scale = decompose(np.asarray(data.root.matrix, dtype="<f"))[2]
                data.root.matrix = compose(np.zeros(3), np.eye(3), scale)
            else:
                data.root.matrix = ID44
                if isinstance(data.root, nif.NiNode):
                    data.root.matrix = ID44

        # attach kf file
        if self.attach_keyframe_data:
            self.import_keyframe_data(data)

        # copy file name
        if data.root.name == "" or self.always_use_file_name_for_root_name:
            if self.always_use_file_name_for_root_name:
                data.root.name = self.filepath.stem + self.filepath.suffix
            else:
                data.root.name = self.filepath.stem

        # bake skinned meshes into their static pose, since no armature
        # will exist to deform them (e.g. the siltstrider's arms would
        # otherwise show their raw bind-pose geometry near the origin)
        if self.ignore_armatures:
            for root in data.roots:
                if isinstance(root, nif.NiNode):
                    for mesh in list(root.skinned_meshes()):
                        try:
                            mesh.apply_skin(False)
                        except ValueError:
                            # skin root is not an ancestor of its bones;
                            # nothing to bake against, import as static
                            print(f"Warning: cannot resolve skin of '{mesh.name}', importing as static")
                            mesh.skin = None

        # scale correction
        data.apply_scale(self.scale_correction)

        # time correction
        data.apply_time_scale(bpy.context.scene.render.fps)

        # resolve heirarchy
        roots = self.resolve_nodes(data.roots)

        # resolve armatures
        if not self.ignore_armatures and any(self.armatures):
            self.resolve_armatures()
            self.correct_rest_positions()
            self.apply_axis_corrections()
            self.correct_bone_parenting()

        # discard frame pos
        frame_current = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(0)

        # create bl objects
        for node, cls in self.nodes.items():
            if node.output is None:
                cls(node).create()

        # unmute animations
        if not self.ignore_armatures:
            for node in map(self.get, self.armatures):
                node.animation.set_mute(False)

        # restore frame pos
        bpy.context.scene.frame_current = frame_current

        # set active object
        bpy.context.view_layer.objects.active = self.get_root_output(roots)

    # ------
    # REPAIR
    # ------

    @staticmethod
    def discard_detached_skins(data):
        """Strip skins that reference nodes outside the scene graph.

        In-game "standard" NPC exports flatten the hierarchy into loose
        world-baked trishapes but keep stale skin data pointing at
        skeleton nodes that exist in the file only as detached objects.
        Such skins cannot be resolved (no scene bones to bind to) and
        would crash both import paths, so import those meshes as static
        geometry instead.
        """
        in_scene = set()
        for root in data.roots:
            in_scene.add(id(root))
            if isinstance(root, nif.NiAVObject):
                in_scene.update(id(obj) for obj in root.descendants())

        for mesh in data.objects_of_type(nif.NiGeometry):
            skin = getattr(mesh, "skin", None)
            if not (skin and getattr(skin, "root", None) and getattr(skin, "bones", None)):
                continue
            detached = [n for n in (skin.root, *skin.bones) if id(n) not in in_scene]
            if detached:
                print(f"Warning: skin of '{mesh.name}' references nodes outside the scene "
                      f"(e.g. '{detached[0].name}'), importing as static")
                mesh.skin = None

    def repair_scene(self, data):
        """Fix up scene graphs produced by in-game actor/cell exporters.

        Such files dump the live scene graph, which differs from regular
        asset files in two ways this importer must account for:

        1. Skinned body parts keep a skin root (e.g. 'Chest') that is NOT
           an ancestor of the skin bones (which live under a sibling
           'Bip01' node). The es3 bind pose machinery requires the skin
           root to be a common ancestor, so repoint it to the nearest one
           and update the skin offset matrix accordingly.

        2. Left-side body parts are wrapped in BSMirroredNode, whose
           mirroring is implicit in the node type (the matrix itself has
           a positive determinant). Bake the mirror into the matrix so
           the geometry imports mirrored.
        """
        # bake implicit BSMirroredNode mirroring: the engine applies a
        # uniform scale of -1 (point reflection) to these nodes at runtime;
        # the file stores det=+1 matrices. Verified empirically against an
        # in-game skin-deform reference export (see CLAUDE.md).
        mirror = np.diag((-1.0, -1.0, -1.0, 1.0)).astype("<f")
        for obj in data.objects_of_type(nif.BSMirroredNode):
            if la.det(np.asarray(obj.matrix)[:3, :3]) > 0:
                obj.matrix = np.asarray(obj.matrix) @ mirror

        # build a parent map of the whole scene
        parents = {}
        for obj in data.objects_of_type(nif.NiNode):
            for child in obj.children:
                if child is not None:
                    parents.setdefault(child, obj)

        def ancestry(o):  # root-first chain, including o itself
            chain = [o]
            while o in parents:
                o = parents[o]
                chain.append(o)
            chain.reverse()
            return chain

        for mesh in data.objects_of_type(nif.NiGeometry):
            skin = getattr(mesh, "skin", None)
            if not (skin and getattr(skin, "root", None) and getattr(skin, "bones", None)):
                continue

            chains = [ancestry(skin.root)] + [ancestry(b) for b in skin.bones]
            if all(skin.root in chain for chain in chains):
                continue  # already a common ancestor

            # nearest common ancestor of the old root and all bones
            new_root = None
            for level in zip(*chains):
                if all(n is level[0] for n in level) and isinstance(level[0], nif.NiNode):
                    new_root = level[0]
                else:
                    break

            if (new_root is None) or (new_root is skin.root):
                print(f"Warning: could not repair skin root of '{mesh.name}'")
                continue

            # keep semantics: root_to_skin_new = root_to_skin_old @ inv(old_root relative to new_root)
            offset = skin.root.matrix_relative_to(new_root)
            skin.data.matrix = np.asarray(skin.data.matrix) @ la.inv(offset)
            print(f"Repaired skin root of '{mesh.name}': '{skin.root.name}' -> '{new_root.name}'")
            skin.root = new_root

    @staticmethod
    def apply_hierarchy_scales(data):
        """Freeze non-root node scales into the hierarchy.

        NIF nodes may carry a uniform scale, but Blender rest bones are
        orthonormal, so any scale inside an armature silently breaks the
        animation math: static matrices compensate, while fcurve values
        come out in unscaled bone space and the pose drifts apart down
        the chain.

        Two flavors exist (e.g. the sky minotaur):
        - scale embedded in a non-orthonormal rotation matrix. If the
          node's rotation is keyframe-animated, the engine overwrites
          the rotation while animating, so the embedded scale never
          applies in game -- drop it. On static nodes it does apply --
          treat it like a real scale.
        - the actual scale field. The engine applies it to the whole
          subtree, so push it down: child local translations and
          keyframe translations scale by the inherited factor, and leaf
          geometry absorbs the accumulated product into its vertex data.

        World transforms are unchanged. Roots keep their scale (it
        becomes Blender object scale, which is fine). Vanilla assets
        have scale 1.0 everywhere, making this a no-op.
        """
        import copy

        def embedded_scale(obj):
            return abs(la.det(np.asarray(obj.rotation, dtype=np.float64))) ** (1.0 / 3.0)

        def is_uniform(obj):
            r = np.asarray(obj.rotation, dtype=np.float64)
            axes = la.norm(r, axis=0)
            return float(axes.max() - axes.min()) < 1e-3

        def is_animated(obj):
            if not hasattr(obj, "controllers"):
                return False
            kf = obj.controllers.find_type(nif.NiKeyframeController)
            if not (kf and kf.data):
                return False
            d = kf.data
            return bool(d.rotations.euler_data or len(d.rotations.keys)
                        or len(d.translations.keys) or len(d.scales.keys))

        subtree_animated_cache = {}

        def subtree_animated(obj):
            key = id(obj)
            cached = subtree_animated_cache.get(key)
            if cached is not None:
                return cached
            result = is_animated(obj)
            if not result and isinstance(obj, nif.NiNode):
                result = any(
                    subtree_animated(c) for c in obj.children
                    if c is not None and isinstance(c, nif.NiAVObject)
                )
            subtree_animated_cache[key] = result
            return result

        def freezable(obj):
            # Only freeze uniform scale that sits inside animated content
            # (skeletons): that is where Blender's orthonormal rest bones
            # break the fcurve math. Non-uniform scale (NPC race
            # weight/height and the inverse baked into skinned part
            # nodes) and scale on purely static subtrees import correctly
            # as plain object scale -- leave those exactly as before.
            if abs(obj.scale * embedded_scale(obj) - 1.0) <= 1e-4:
                return False
            if not is_uniform(obj):
                if subtree_animated(obj):
                    print(f"Warning: non-uniform scale on animated node '{obj.name}' cannot be frozen")
                return False
            return subtree_animated(obj)

        roots = [root for root in data.roots if isinstance(root, nif.NiNode)]
        if not any(
            freezable(obj)
            for root in roots
            for obj in root.descendants()
            if isinstance(obj, nif.NiAVObject)
        ):
            return

        seen_data = {}  # id(geometry data) -> factor it was baked with
        factors = {}    # id(node) -> accumulated scale frozen into it

        def freeze(node, inherited):
            if abs(inherited - 1.0) > 1e-6:
                node.translation = node.translation * inherited
                if node.bounding_volume:
                    node.bounding_volume.apply_scale(inherited)
                for controller in node.controllers:
                    if isinstance(controller, nif.NiKeyframeController) and controller.data:
                        controller.data.translations.apply_scale(inherited)

            frozen = freezable(node)
            own = 1.0
            if frozen:
                # normalize scale hidden in the rotation matrix
                embed = embedded_scale(node)
                if abs(embed - 1.0) > 1e-4:
                    node.rotation = node.rotation / embed

                own = node.scale * embed
                kf = node.controllers.find_type(nif.NiKeyframeController)
                kfd = kf.data if (kf and kf.data) else None
                if kfd is not None and len(kfd.scales.keys):
                    svals = kfd.scales.keys[:, 1]
                    if float(svals.max() - svals.min()) < 1e-3:
                        # Constant scale animation (e.g. the minotaur's 'Root
                        # Bone': static rotation embeds 2.0 AND scale keys hold
                        # a constant 2.0 -- the same scale expressed twice).
                        # Fold the animated value into the static freeze and
                        # drop the keys, so the scale is applied exactly once.
                        s_anim = float(svals[0])
                        kfd.scales.keys = kfd.scales.keys[:0]
                        if abs(s_anim - own) > 1e-3:
                            print(f"Warning: '{node.name}' static scale {own:.3f} != animated scale {s_anim:.3f}; using animated")
                        own = node.scale * s_anim
                    elif abs(embed - 1.0) > 1e-4:
                        # variable scale animation cannot be frozen; keep the
                        # keys and only report the (unfixable) embedded part
                        print(f"Warning: '{node.name}' has variable scale animation; import may be inexact")
                elif abs(embed - 1.0) > 1e-4:
                    rot = kfd.rotations if kfd is not None else None
                    if rot is not None and (rot.euler_data or len(rot.keys)):
                        # rotation-animated with no scale keys: the engine
                        # replaces the rotation while animating, so the
                        # embedded scale never renders -- discard it
                        print(f"Discarding junk rotation scale {embed:.3f} on animated node '{node.name}'")
                        own = node.scale
                    else:
                        print(f"Freezing rotation scale {embed:.3f} of static node '{node.name}'")

            # total = scale removed from this node's frame by the freeze;
            # un-frozen nodes keep their own scale (only inherited applies)
            total = inherited * own
            factors[id(node)] = total
            if isinstance(node, nif.NiNode):
                if frozen:
                    node.scale = 1.0
                for child in node.children:
                    if child is not None and isinstance(child, nif.NiAVObject):
                        freeze(child, total)
            elif abs(total - 1.0) > 1e-6 and getattr(node, "data", None) is not None:
                # geometry leaf: bake the accumulated scale into vertices
                prior = seen_data.get(id(node.data))
                if prior is None:
                    node.data.apply_scale(total)
                    seen_data[id(node.data)] = total
                elif abs(prior - total) > 1e-6:
                    # instanced data used at a different accumulated scale
                    node.data = copy.deepcopy(node.data)
                    node.data.apply_scale(total / prior)
                    seen_data[id(node.data)] = total
                morpher = node.controllers.find_type(nif.NiGeomMorpherController)
                if morpher and morpher.data:
                    morpher.data.apply_scale(total)
                if frozen:
                    node.scale = 1.0

        for root in roots:
            print(f"Applying hierarchy scales under '{root.name}'")
            for child in root.children:
                if child is not None and isinstance(child, nif.NiAVObject):
                    freeze(child, 1.0)

        # Rescale skin bind matrices to match the frozen bones (the
        # world-invariant counterpart of the freeze: bones lost factor
        # f_bone from their matrices, meshes gained f_mesh in their
        # vertex data). Then normalize any residual scale left in the
        # binds into the mesh's bind-space vertex data -- assets differ
        # in where they put the scale (the minotaur's binds carry the
        # inverse of the skeleton scale; the seacrab's binds are unit
        # with the scale only on the skin root), and the bind-pose
        # machinery requires orthonormal binds either way.
        skin_seen = {}  # id(geometry data) -> residual factor baked
        for mesh in data.objects_of_type(nif.NiGeometry):
            skin = getattr(mesh, "skin", None)
            if not (skin and getattr(skin, "root", None)
                    and getattr(skin, "bones", None) and getattr(skin, "data", None)):
                continue
            f_mesh = factors.get(id(mesh), 1.0)

            residual = None
            new_binds = []
            for bone, bone_data in zip(skin.bones, skin.data.bone_data):
                f_bone = factors.get(id(bone), 1.0)
                m = np.asarray(bone_data.matrix, dtype=np.float64).copy()
                m[:3, :] *= f_bone
                m[:3, :3] /= f_mesh
                r = abs(la.det(m[:3, :3])) ** (1.0 / 3.0)
                residual = r if residual is None else residual
                new_binds.append(m)

            if residual is None:
                continue
            if abs(residual - 1.0) > 1e-4:
                # fold the residual into the bind-space vertex data
                print(f"Normalizing residual bind scale {residual:.3f} of '{mesh.name}'")
                for m in new_binds:
                    m[:3, :3] /= residual
                if getattr(mesh, "data", None) is not None:
                    prior = skin_seen.get(id(mesh.data))
                    if prior is None:
                        mesh.data.apply_scale(residual)
                        skin_seen[id(mesh.data)] = residual
                    elif abs(prior - residual) > 1e-6:
                        mesh.data = copy.deepcopy(mesh.data)
                        mesh.data.apply_scale(residual / prior)
                        skin_seen[id(mesh.data)] = residual

            changed = abs(f_mesh - 1.0) > 1e-6 or abs(residual - 1.0) > 1e-4
            for bone, bone_data, m in zip(skin.bones, skin.data.bone_data, new_binds):
                if changed or abs(factors.get(id(bone), 1.0) - 1.0) > 1e-6:
                    bone_data.matrix = m

            # root_to_skin relates skin-root space to mesh space; both may
            # have been rescaled by the freeze (crab: skin.root is 'Bip01'
            # itself, factor 3)
            f_root = factors.get(id(skin.root), 1.0)
            if abs(f_mesh - 1.0) > 1e-6 or abs(f_root - 1.0) > 1e-6:
                m = np.asarray(skin.data.matrix, dtype=np.float64).copy()
                m[:3, :] *= f_mesh
                m[:3, :3] /= f_root
                skin.data.matrix = m

    # -------
    # RESOLVE
    # -------

    @staticmethod
    def best_lod_children(lod_node, children):
        # Keep only the highest-detail level of a NiLODNode. lod_levels pairs up
        # index-for-index with children as (near, far); the level starting nearest
        # the camera is the detailed one. Children may be any NiAVObject (a NiNode
        # wrapping several shapes, or a bare NiTriShape), so type is not a filter.
        levels = lod_node.lod_levels
        # Pair by ORIGINAL index: a null child still consumes a level slot, so
        # compacting the list before indexing lod_levels would misalign them.
        paired = [(c, levels[i]) for i, c in enumerate(children)
                  if c is not None and i < len(levels)]

        if len(paired) < 2:
            # Nothing to choose between, or level data too short to trust.
            present = [c for c in children if c is not None]
            return present[:1] if len(present) > 1 else children

        # Smallest near-range is the detailed level; on a tie prefer the one
        # visible over the longer distance (a far == near level never renders).
        best = min(paired, key=lambda p: (p[1][0], -p[1][1]))[0]
        return [best]

    def resolve_nodes(self, ni_roots, parent=None):
        # Only process objects that have transformations (NiAVObject)
        root_nodes = [SceneNode(self, root, parent) for root in ni_roots 
                      if root and isinstance(root, nif.NiAVObject)]

        queue = collections.deque(root_nodes)
        while queue:
            node = queue.popleft()

            if self.process(node):
                self.history[node.source].add(node)
                if hasattr(node.source, "children"):
                    children = node.source.children
                    if self.filter_best_lod and isinstance(node.source, nif.NiLODNode):
                        children = self.best_lod_children(node.source, children)

                    for child in children:
                        if not (child and isinstance(child, nif.NiAVObject)):
                            continue

                        if self.ignored_nodes_under_switches and isinstance(node.source, nif.NiSwitchNode):
                            if re.sub(r'\.\d+$', '', child.name.lower()) in self.ignored_nodes_under_switches:
                                continue

                        child_node = SceneNode(self, child, node)

                        # Proxy Mode: Each child of the container root starts a fresh object branch
                        if self.proxy_mode and node.parent is None:
                            child_node.branch_mesh_found = [False]

                        if self.proxy_mode:
                            is_mesh = child.type in ("NiTriShape", "NiTriStrips")
                            if is_mesh:
                                # Only import the first mesh for this branch if not already proxied at root
                                if not child_node.branch_mesh_found[0]:
                                    queue.append(child_node)
                                    child_node.branch_mesh_found[0] = True
                                continue
                        
                        queue.append(child_node)

        return root_nodes

    def resolve_armatures(self):
        if self.ignore_armatures:
            return
        """ TODO
            support for multiple skeleton roots
        """
        orphan_bones = self.armatures.pop(None, {})

        # sort roots via heirarchy
        roots = list(map(self.get, self.armatures))
        roots.sort(key=lambda r: len([*r.parents]))

        # select the top-most root
        root = roots[0].source
        bones = self.armatures[root]

        # collect all orphan bones
        bones.update(orphan_bones)

        # collect all others bones
        for other_root in self.armatures.keys() - {root}:
            other_bones = self.armatures.pop(other_root)
            bones.add(other_root)
            bones.update(other_bones)

        # only descendants of root
        root_node = self.get(root)
        bones -= {node.source for node in (root_node, *root_node.parents)}

        # bail if no bones present
        if len(bones) == 0:
            self.armatures.clear()
            return

        def validate_bone_chains():
            # ensure all ancestors between each bone and the root are bones,
            # and discard bones that are not descendants of the root at all
            # (e.g. a second skeleton elsewhere in a cell export) so that
            # scene-level nodes never get dragged into the armature
            for source in list(bones):
                chain = []
                for parent in self.get(source).parents:
                    if parent.source is root:
                        bones.update(chain)
                        break
                    chain.append(parent.source)
                else:
                    bones.discard(source)
                    print(f"Warning: '{source.name}' is not a descendant of "
                          f"'{root.name}' and will not become a bone")

        # connect skin bones to the root before promoting animated
        # descendants, so that root children which are merely ancestors
        # of skin bones (e.g. the siltstrider's 'Movement' node, whose
        # 'Body' subtree holds the rigid-animated legs) count as bones
        validate_bone_chains()

        # consider any descendants which are animated to be bones
        # this is usually desired, and to not do so would mean we
        # have to fix the animations of any node who's transforms
        # are modified by a parent bone receiving axis correction
        # (mirrored nodes are excluded: negative-scale bones would
        # break matrix decomposition)
        for root_bone in filter(bones.__contains__, root.children):
            for child in root_bone.descendants():
                if isinstance(child, nif.NiNode) and not isinstance(child, nif.BSMirroredNode):
                    if child.controllers.find_type(nif.NiKeyframeController):
                        bones.add(child)

        # validate all bone chains
        validate_bone_chains()

        # bail if validation discarded everything
        if len(bones) == 0:
            self.armatures.clear()
            return

        # order bones by heirarchy
        self.armatures[root] = dict.fromkeys(node.source for node in self.nodes if node.source in bones).keys()

        # preserve bone pose matrices
        for node in self.iter_bones(root_node):
            node.matrix_posed = node.matrix_world

        # send all bones to rest pose
        root.apply_bone_bind_poses()
        root.apply_skins(keep_skins=True)

        # apply updated rest matrices
        for node in self.iter_bones(root_node):
            node.matrix_local = node.source.matrix

        # specify node as Armature
        self.nodes[root_node] = Armature

    def correct_rest_positions(self):
        if self.ignore_armatures or not self.armatures:
            return

        root = self.get_armature_node()
        root_bone = next(self.iter_bones(root))

        # calculate corrected transformation matrix
        t, r, s = decompose(root_bone.matrix_posed)
        r = nif_utils.snap_rotation(r)
        corrected_matrix = compose(t, r, s)

        # only do corrections if they are necessary
        if np.allclose(root_bone.matrix_world, corrected_matrix, rtol=0, atol=1e-6):
            return

        # correct the rest matrix of skinned meshes
        inverse = la.inv(root_bone.matrix_world)
        for node in self.get_skinned_meshes():
            if root_bone not in node.parents:
                node.matrix_world = corrected_matrix @ (inverse @ node.matrix_world)

        # correct the rest matrix of the root bone
        root_bone.matrix_world = corrected_matrix

    def apply_axis_corrections(self):
        if self.ignore_armatures or not self.armatures:
            return

        root = self.get_armature_node()
        bones = list(self.iter_bones(root))

        # Aim each non-biped bone's corrected Y axis (the Blender bone
        # axis) at the mean of its child bones, so bone sticks follow the
        # limbs like NifSkope's joint lines -- custom rigs (minotaur,
        # seacrab) have arbitrary bind axes and the fixed correction
        # leaves their sticks pointing sideways. Purely a change of rest
        # frame: the same per-bone correction feeds the animation
        # conversion below, so poses and animations are unaffected.
        for node in bones:
            if "Bip01" in node.name:
                continue
            positions = [c.matrix_posed[:3, 3] for c in node.children if hasattr(c, "matrix_posed")]
            if not positions:
                continue
            direction = np.mean(positions, axis=0) - node.matrix_posed[:3, 3]
            length = la.norm(direction)
            if length < 1e-5:
                continue
            # child direction in the bone's local frame -> corrected Y column
            y = node.matrix_posed[:3, :3].T @ (direction / length)
            y = y / la.norm(y)
            # pick the remaining axes with minimal twist vs the default correction
            x = other_axis_correction[:3, 0].astype(np.float64)
            x = x - np.dot(x, y) * y
            if la.norm(x) < 1e-5:
                x = other_axis_correction[:3, 2].astype(np.float64)
                x = x - np.dot(x, y) * y
            x = x / la.norm(x)
            z = np.cross(x, y)
            correction = np.identity(4, dtype="<f")
            correction[:3, 0] = x
            correction[:3, 1] = y
            correction[:3, 2] = z
            node.axis_correction_override = correction

        # apply bone axis corrections
        for node in reversed(bones):
            node.matrix_posed = node.matrix_posed @ node.axis_correction
            node.matrix_local = node.matrix_local @ node.axis_correction
            for child in node.children:
                child.matrix_local = node.axis_correction_inverse @ child.matrix_local

        # apply anim axis corrections
        root_inverse = la.inv(root.matrix_world)
        for node in bones:
            kf_controller = node.source.controllers.find_type(nif.NiKeyframeController)
            if not (kf_controller and kf_controller.data):
                continue

            try:
                parent_matrix = node.parent.matrix_posed
                parent_matrix_uncorrected = parent_matrix @ node.parent.axis_correction_inverse
            except AttributeError:  # parent is not bone
                parent_matrix = node.parent.matrix_world if node.parent else ID44
                parent_matrix_uncorrected = parent_matrix

            matrix_world = parent_matrix @ node.matrix_local
            matrix_relative_to_root = root_inverse @ matrix_world

            posed_offset = la.solve(matrix_relative_to_root, root_inverse)
            posed_offset = posed_offset @ parent_matrix_uncorrected

            t = kf_controller.data.translations
            if len(t.values):
                rotation = posed_offset[:3, :3].T
                translation = posed_offset[:3, 3]
                # convert to pose space
                t.values[:] = t.values @ rotation + translation
                if t.key_type.name == "BEZ_KEY":
                    t.in_tans[:] = t.in_tans @ rotation
                    t.out_tans[:] = t.out_tans @ rotation

            r = kf_controller.data.rotations
            if r.euler_data:
                # Euler-keyed rotations expose no .values, so they would skip
                # the corrections below and reach Blender in raw NIF space
                # (mangling the animation exactly like unpromoted rigid nodes
                # once did). Bones need quaternions anyway -- convert now so
                # the standard correction path applies.
                r.convert_to_quaternions()
            if len(r.values):
                # apply axis correction
                axis_fix = nif_utils.quaternion_from_matrix(node.axis_correction)
                r.values[:] = nif_utils.quaternion_mul(r.values, axis_fix)
                # convert to pose space
                to_posed = nif_utils.quaternion_from_matrix(posed_offset)
                r.values[:] = nif_utils.quaternion_mul(to_posed, r.values)

    def correct_bone_parenting(self):
        if self.ignore_armatures or not self.armatures:
            return
        """Set the parent of skinned meshes to the armature responsible for deforming them.

        This must be done as using both skinning and bone-parenting at the same time does not
        behave correctly in Blender.

        Usually this occurs when a file contains nested armatures.
        See `Tri Hand01` in the vanilla `r/skeleton.nif` model for example.
        """
        if not self.armatures:
            return

        armature = self.get_armature_node()
        for node in self.get_skinned_meshes():
            if node.parent != armature:
                matrix_world = node.matrix_world
                node.parent = armature
                node.matrix_world = matrix_world

    # -------
    # PROCESS
    # -------

    @nif_utils.dispatcher
    def process(self, node):
        # Skip nodes with "custom_normals_" in their name
        if any(x in node.name.lower() for x in ("custom_normals", "custom normals")):
            print(f"Skipping node: {node.name}")
            return False
        print(f"Warning: Unhandled Type: {node.source.type}")
        return False

    @process.register("NiSwitchNode")
    def process_switch(self, node):
        if self.ignored_nodes and re.sub(r'\.\d+$', '', node.name.lower()) in self.ignored_nodes:
            return False
        return self.process_empty(node)

    @process.register("NiNode")
    @process.register("NiLODNode")
    @process.register("NiBSAnimationNode")
    @process.register("NiCollisionSwitch")
    @process.register("NiSortAdjustNode")
    @process.register("BSMirroredNode")
    def process_empty(self, node):
        stripped = re.sub(r'\.\d+$', '', node.name.lower())
        if (self.ignored_nodes and stripped in self.ignored_nodes) or \
           (self.ignore_collision_nodes and stripped == "collision"):
            return False
        self.nodes[node] = Empty

        # detect bones via name conventions
        name = node.name.lower()
        if (name == "bip01") or (name == "root bone"):
            self.armatures[node.source].update()
        elif ("bip01" in name) or name.endswith(" bone"):
            self.armatures[None].add(node.source)

        return True

    @process.register("NiBillboardNode")
    def process_billboard(self, node):
        if self.ignore_billboard_nodes:
            return False
        return self.process_empty(node)
    
    @process.register("NiBSParticleNode")
    def process_particles(self, node):
        if self.ignore_particle_nodes:
            return False
        return self.process_empty(node)

    @process.register("NiTriShape")
    def process_mesh(self, node):
        # Skip "Tri Shadow" nodes or nodes named exactly "shadow"
        if self.ignore_tri_shadow and (node.name.lower().startswith("tri shadow") or node.name.lower() == "shadow"):
            print(f"Skipping shadow mesh: {node.name}")
            return False

        self.nodes[node] = Mesh

        # track skinned meshes
        skin = getattr(node.source, "skin", None)
        if skin and getattr(skin, "root", None) and getattr(skin, "bones", None):
            self.armatures[skin.root].update(skin.bones)

        return True

    @process.register("RootCollisionNode")
    def process_collision(self, node):
        if self.ignore_collision_nodes:
            return False

        self.nodes[node] = Empty
        self.colliders[node.source].update(node.source.descendants())
        return True

    @process.register("NiLight")
    @process.register("NiDirectionalLight")
    @process.register("NiPointLight")
    @process.register("NiSpotLight")
    def process_light(self, node):
        self.nodes[node] = Light
        return True

    # -------
    # UTILITY
    # -------

    def get(self, source):
        return next(iter(self.history[source]))

    def iter_bones(self, root):
        if self.ignore_armatures:
            return
        yield from map(self.get, self.armatures[root.source])

    def get_root_output(self, roots):
        if not roots:
            return None
        root_out = roots[0].output
        if root_out is None:
            print("No root object created; skipping active object assignment")
            return None
        try:
            return root_out.id_data
        except AttributeError:
            return root_out

    def get_armature_node(self):
        if self.ignore_armatures:
            return None
        return self.get(*self.armatures)

    def get_skinned_meshes(self):
        if self.ignore_armatures:
            return
        for node in self.nodes:
            if getattr(node.source, "skin", None):
                yield node

    def get_mesh_data(self, shape, name) -> bpy.types.Mesh:
        if self.proxy_mode:
            # Use a shared mesh block called "proxy" for all meshes
            try:
                return bpy.data.meshes["proxy"]
            except KeyError:
                return bpy.data.meshes.new("proxy")

        if shape.morph_targets or shape.bone_influences:
            # TODO: Support instancing of animated and skinned meshes.
            return bpy.data.meshes.new(name)

        try:
            bl_data = self.mesh_data[shape.data]
        except KeyError:
            bl_data = self.mesh_data[shape.data] = bpy.data.meshes.new(name)

        return bl_data

    def import_keyframe_data(self, data):
        kf_path = self.filepath.with_suffix(".kf")
        if not kf_path.exists():
            print(f'import_keyframe_data: "{kf_path}" does not exist')
        else:
            kf_data = nif.NiStream()
            kf_data.load(kf_path)
            data.attach_keyframe_data(kf_data)

    @property
    def scale_correction(self):
        addon = bpy.context.preferences.addons[__package__]
        return addon.preferences.scale_correction


class SceneNode:

    def __init__(self, importer, source, parent=None):
        self.importer = importer
        #
        self.source = source
        self.output = None
        #
        self.parent = parent
        self.children = list()
        self.matrix_local = np.asarray(source.matrix, dtype="<f")

        # Proxy Mode: track if a mesh has already been found in this branch
        if parent:
            self.branch_mesh_found = parent.branch_mesh_found
        else:
            self.branch_mesh_found = [False]

    def __repr__(self):
        if not self.parent:
            return f'SceneNode("{self.name}", parent=None)'
        return f'SceneNode("{self.name}", parent={self.parent.name})'

    def create(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return sanitize_name(self.source.name, normalize_prefix=(self.importer.normalize_prefix and (self.parent is not None or self.importer.normalize_prefix_root)), normalize_prefix_max_length=self.importer.normalize_prefix_max_length)

    @property
    def bone_name(self):
        name = self.name
        if name.startswith("Bip01 L "):
            return f"Bip01 {name[8:]}.L"
        if name.startswith("Bip01 R "):
            return f"Bip01 {name[8:]}.R"
        return name

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, node):
        try:  # remove from old children list
            self._parent.children.remove(self)
        except (AttributeError, ValueError):
            pass
        self._parent = node
        try:  # append onto new children list
            self._parent.children.append(self)
        except (AttributeError, ValueError):
            pass

    @property
    def parents(self):
        node = self.parent
        while node:
            yield node
            node = node.parent

    @property
    def properties(self):
        props = {type(p): p for p in self.source.properties}
        if self.parent:
            return {**self.parent.properties, **props}
        return props

    @property
    def matrix_world(self):
        if self.parent:
            return self.parent.matrix_world @ self.matrix_local
        return self.matrix_local

    @matrix_world.setter
    def matrix_world(self, matrix):
        if self.parent:
            matrix = la.solve(self.parent.matrix_world, matrix)
        self.matrix_local = matrix

    @property
    def axis_correction(self):
        override = self.__dict__.get("axis_correction_override")
        if override is not None:
            return override
        if "Bip01" in self.name:
            return biped_axis_correction
        return other_axis_correction

    @property
    def axis_correction_inverse(self):
        override = self.__dict__.get("axis_correction_override")
        if override is not None:
            # pure rotation: inverse == transpose
            return np.ascontiguousarray(override.T)
        if "Bip01" in self.name:
            return biped_axis_correction_inverse
        return other_axis_correction_inverse

    @property
    def animation(self):
        return Animation(self)

    @property
    def material(self):
        return Material(self)


class Empty(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self, bl_data=None):
        self.output = self.create_object(bl_data)
        self.output.empty_display_size *= self.importer.scale_correction
        self.output.mw.object_flags = self.source.flags

        bl_parent = getattr(self.parent, "output", None)
        try:
            self.output.parent = bl_parent
        except TypeError:
            # parent is an armature bone
            self.output.parent = bl_parent.id_data
            self.output.parent_type = "BONE"
            self.output.parent_bone = bl_parent.name
            self.output.matrix_world = (self.parent.matrix_posed @ self.matrix_local).T
        else:
            # parent is an empty or None
            self.output.matrix_local = self.matrix_local.T

        if self.source in self.importer.colliders:
            self.output.name = "Collision"
            self.output.display_type = "WIRE"

        if self.source.is_bounding_box:
            self.convert_to_bounding_box()

        self.animation.create()

        return self.output

    def create_object(self, bl_data=None):
        name = self.name
        # Apply NIF naming if in Proxy Mode and this is a mesh-like object
        is_geom = hasattr(self.source, "type") and self.source.type in ("NiTriShape", "NiTriStrips")
        if self.importer.proxy_mode and is_geom:
            # Find the first parent that has a .nif name (the containing asset node)
            for p in self.parents:
                if ".nif" in p.name.lower():
                    name = p.name
                    break
            
        bl_object = bpy.data.objects.new(name, bl_data)
        self.importer.active_collection.objects.link(bl_object)
        bl_object.select_set(True)
        return bl_object

    def convert_to_bounding_box(self):
        self.output.empty_display_size = 1.0
        self.output.empty_display_type = 'CUBE'
        self.output.matrix_world = self.source.bounding_volume.matrix.T


class Armature(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if self.importer.ignore_armatures:
            return
        # create armature object
        bl_data = bpy.data.armatures.new(self.name)
        bl_object = Empty(self).create(bl_data)

        # apply default settings
        bl_data.display_type = "STICK"
        bl_object.show_in_front = True

        # swap to edit mode to allow creation of bones
        bpy.context.view_layer.objects.active = bl_object
        was_hidden = bl_object.hide_viewport
        bl_object.hide_viewport = False

        def find_layer_collection(layer_coll, collection):
            if layer_coll.collection == collection:
                return layer_coll
            for child in layer_coll.children:
                result = find_layer_collection(child, collection)
                if result:
                    return result
            return None

        layer_collection = find_layer_collection(
            bpy.context.view_layer.layer_collection,
            self.importer.active_collection
        )
        was_collection_hidden = layer_collection.hide_viewport if layer_collection else False
        if layer_collection:
            layer_collection.hide_viewport = False
        bpy.ops.object.mode_set(mode="EDIT")

        # used for calculating armature space matrices
        root_inverse = la.inv(self.matrix_world)

        # bone mappings cache
        bones = {}

        # position bone heads
        for node in self.importer.iter_bones(self):
            # create bone and assign its parent
            bone = bones[node] = bl_data.edit_bones.new(node.bone_name)
            bone.parent = bones.get(node.parent)
            bone.select = True

            # compute the armature-space matrix
            matrix = root_inverse @ node.matrix_world

            # calculate axis/roll and head/tail
            bone.matrix = matrix.T
            bone.tail = matrix[:3, 1] + matrix[:3, 3]  # axis + head

        # position bone tails
        deferred = []
        for node, bone in bones.items():
            # edit_bones will not persist outside of edit mode
            bones[node] = bone.name

            length = 0.0
            if bone.children:
                # calculate length from children mean location
                locations = [c.matrix_posed[:3, 3] for c in node.children if c in bones]
                if locations:
                    length = la.norm(node.matrix_posed[:3, 3] - np.mean(locations, axis=0))

            if length > 1e-4:
                bone.length = length
            else:
                deferred.append(bone)

        # Bones without a usable child-derived length (childless bones, or
        # bones whose children coincide with their head, e.g. Bip01 vs
        # Bip01 Pelvis) get a modest display length. Never collapse a bone
        # (or nudge a collapsed tail): that resets its rest orientation,
        # which corrupts animations, as fcurve channels are expressed
        # relative to the rest frame.
        if deferred:
            valid = [b.length for b in bl_data.edit_bones if b not in deferred]
            fallback = float(np.median(valid)) if valid else 1.0
            for bone in deferred:
                if bone.parent and bone.parent not in deferred:
                    bone.length = bone.parent.length / 2
                else:
                    bone.length = fallback / 2

        # back to object mode now that all bones exist
        bpy.ops.object.mode_set(mode="OBJECT")
        bl_object.hide_viewport = was_hidden
        if layer_collection:
            layer_collection.hide_viewport = was_collection_hidden

        # assign node.output and apply pose transforms
        for node, name in bones.items():
            pose_bone = node.output = bl_object.pose.bones[name]
            # compute the armature-space matrix
            pose_bone.matrix = (root_inverse @ node.matrix_posed).T
            # TODO try not to call scene update
            bpy.context.view_layer.depsgraph.update()
            # create animations, preserve poses
            node.animation.create()
            node.animation.set_mute(True)

        return bl_object


class Mesh(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        bl_data = self.importer.get_mesh_data(self.source, self.name)
        bl_object = Empty(self).create(bl_data)

        if self.importer.proxy_mode:
            # Only create the cube geometry once for the shared "proxy" mesh
            if bl_data.users == 1:
                # Standard cube vertices (centered at origin, 2x2x2)
                vertices = np.array([
                    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]
                ], dtype="<f")
                
                # Standard cube triangles (12 triangles)
                triangles = np.array([
                    [0, 2, 1], [0, 3, 2], # Bottom
                    [4, 5, 6], [4, 6, 7], # Top
                    [0, 1, 5], [0, 5, 4], # Back
                    [1, 2, 6], [1, 6, 5], # Right
                    [2, 3, 7], [2, 7, 6], # Front
                    [3, 0, 4], [3, 4, 7]  # Left
                ], dtype="<i4")
                
                self.create_vertices(bl_object, vertices)
                self.create_triangles(bl_object, triangles)
                bl_object.data.update()
            
            # Match display type to parent if possible
            try:
                self.output.display_type = self.parent.output.display_type
            except AttributeError:
                pass
            return bl_object

        if len(self.source.data.vertices) == 0:
            return bl_object

        # We only need to calculate the geometry once per mesh instance.
        if bl_data.users == 1:
            ni_data = self.calc_geometry_data()

            self.create_vertices(bl_object, ni_data.vertices)
            self.create_triangles(bl_object, ni_data.triangles)

            self.create_vertex_colors(bl_object, ni_data.vertex_colors)
            self.create_uv_sets(bl_object, ni_data.uv_sets)

            self.create_vertex_weights(bl_object, ni_data.vertex_weights)
            self.create_vertex_morphs(bl_object, ni_data.vertex_morphs)

            self.create_normals(bl_object, ni_data.normals)

        try:
            self.output.display_type = self.parent.output.display_type
        except AttributeError:
            pass

        self.material.create()

        return bl_object

    def create_vertices(self, ob, vertices):
        ob.data.vertices.add(len(vertices))
        ob.data.vertices.foreach_set("co", vertices.ravel())

    def create_triangles(self, ob, triangles):
        n = len(triangles)
        ob.data.loops.add(3 * n)
        ob.data.loops.foreach_set("vertex_index", triangles.ravel())

        ob.data.polygons.add(n)
        ob.data.polygons.foreach_set("loop_total", [3] * n)
        ob.data.polygons.foreach_set("loop_start", range(0, 3 * n, 3))
        ob.data.polygons.foreach_set("use_smooth", [True] * n)

        ob.data.update()

    def create_normals(self, ob, normals):
        if len(normals) == 0:
            ob.data["ignore_normals"] = True
        else:
            # Each polygon has a "use_smooth" flag that controls whether it
            # should use flat shading or smoooth shading. Our custom normals
            # will override this behavior, but the user may decide to remove
            # custom data layers at some point after importing, which would
            # make the renderer fall back to using said flags. We calculate
            # these flags as best we can by checking if the polygon's normals
            # are all equivalent, which would mean it is NOT smooth shaded.
            n0, n1, n2 = np.swapaxes(normals.reshape(-1, 3, 3), 0, 1)
            n0__eq__n1 = np.isclose(n0, n1, rtol=0, atol=1e-04)
            n1__eq__n2 = np.isclose(n1, n2, rtol=0, atol=1e-04)
            use_smooth = ~(n0__eq__n1 & n1__eq__n2).all(axis=1)
            ob.data.polygons.foreach_set("use_smooth", use_smooth)

            # apply custom normals
            if not self.importer.ignore_custom_normals:
                if ob.data.validate(verbose=False, clean_customdata=False):
                    print(f"Warning: Invalid mesh data, custom normals will be skipped: ({ob.name})")
                else:
                    ob.data.normals_split_custom_set(normals)
                    if bpy.app.version < (4, 1, 0):
                        ob.data.use_auto_smooth = True

    def create_uv_sets(self, ob, uv_sets):
        for uv_set in uv_sets[:8]:  # max 8 uv sets (blender limitation)
            uv = ob.data.uv_layers.new()
            uv.data.foreach_set("uv", uv_set.ravel())

    def create_vertex_colors(self, ob, vertex_colors):
        if len(vertex_colors):
            vc = ob.data.vertex_colors.new()
            vc.data.foreach_set("color", vertex_colors.ravel())

    def create_vertex_weights(self, ob, vertex_weights):
        if self.importer.ignore_armatures:
            return
        if not len(vertex_weights):
            return
        if not self.importer.armatures:
            return

        bones = map(self.importer.get, self.source.skin.bones)

        # Make Armature
        # The modifier must target the single armature that owns the skin
        # bones; skin.root itself may be a plain empty in actor exports.
        armature_node = self.importer.get_armature_node()
        armature = ob.modifiers.new("", "ARMATURE")
        armature.object = armature_node.output.id_data

        # Vertex Weights
        for i, node in enumerate(bones):
            vg = ob.vertex_groups.new(name=node.output.name)

            weights = vertex_weights[i]
            for j in np.flatnonzero(weights).tolist():
                vg.add((j,), weights[j], "ADD")

    def create_vertex_morphs(self, ob, vertex_morphs):
        if self.importer.ignore_animations: # NEW
            return
        if not len(vertex_morphs):
            return

        animation = self.animation

        # add basis key
        ob.shape_key_add(name="Basis")

        # add anim data
        action = animation.get_action(ob.data.shape_keys)

        # add morph keys
        for i, target in enumerate(self.source.morph_targets):
            # create morph targets
            shape_key = ob.shape_key_add(name="")
            shape_key.data.foreach_set("co", vertex_morphs[i].ravel())

            # create morph fcurves
            data_path = shape_key.path_from_id("value")
            fc = animation.create_fcurves(action, data_path)

            # add fcurve keyframes
            fc.keyframe_points.add(len(target.keys))
            fc.keyframe_points.foreach_set("co", target.keys[:, :2].ravel())
            animation.create_interpolation_data(target, fc)
            fc.update()

        # update frame range
        animation.update_frame_range(self.source.controller)

    def calc_geometry_data(self):
        vertices = self.source.data.vertices
        normals = self.source.data.normals
        uv_sets = self.source.data.uv_sets.copy()
        vertex_colors = self.source.data.vertex_colors
        vertex_weights = self.source.vertex_weights()
        vertex_morphs = self.source.vertex_morphs()
        triangles = self.source.data.triangles

        if len(normals):
            # re-unitize, fixes landscape meshes
            normals /= la.norm(normals, axis=1, keepdims=True)
            # reconstruct as per-triangle layout
            normals = normals[triangles].reshape(-1, 3)

        if len(uv_sets):
            # convert OpenGL into Blender format
            uv_sets[..., 1] = 1 - uv_sets[..., 1]
            # reconstruct as per-triangle layout
            uv_sets = uv_sets[:, triangles].reshape(-1, triangles.size, 2)

        if len(vertex_colors):
            # reconstruct as per-triangle layout
            vertex_colors = vertex_colors[triangles].reshape(-1, 3)

        # remove doubles
        scale = decompose(self.matrix_world)[-1]
        indices, inverse = nif_utils.unique_rows(
            vertices * scale,
            *vertex_weights,
            *vertex_morphs,
            precision=self.importer.vertex_precision,
        )
        if len(vertices) > len(indices) > 3:
            vertices = vertices[indices]
            vertex_weights = vertex_weights[:, indices]
            vertex_morphs = vertex_morphs[:, indices]
            triangles = inverse[triangles]

        # '''
        # Blender does not allow two faces to use identical vertex indices, regardless of order.
        # This is problematic as such occurances are commonly found throughout most nif data sets.
        # The usual case is "double-sided" faces, which share vertex indices but differ in winding.
        # Identify the problematic faces and duplicate their vertices to ensure the indices are unique.
        uniques, indices = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
        if len(triangles) > len(uniques):
            # boolean mask of the triangles to be updated
            target_faces = np.full(len(triangles), True)
            target_faces[indices] = False

            # indices of the vertices that must be copied
            target_verts = triangles[target_faces].ravel()

            # find the vertices used in problematic faces
            new_vertices = vertices[target_verts]
            new_vertex_weights = vertex_weights[:, target_verts]
            new_vertex_morphs = vertex_morphs[:, target_verts]
            new_vertex_indices = np.arange(len(new_vertices)) + len(vertices)

            # update our final mesh data with new geometry
            vertices = np.vstack((vertices, new_vertices))
            vertex_weights = np.hstack((vertex_weights, new_vertex_weights))
            vertex_morphs = np.hstack((vertex_morphs, new_vertex_morphs))

            # copy is needed since shapes could share data
            triangles = triangles.copy()
            triangles[target_faces] = new_vertex_indices.reshape(-1, 3)
        # '''

        return nif_utils.Namespace(
            triangles=triangles,
            vertices=vertices,
            normals=normals,
            uv_sets=uv_sets,
            vertex_colors=vertex_colors,
            vertex_weights=vertex_weights,
            vertex_morphs=vertex_morphs,
        )


class Material(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        properties = self.properties
        if len(properties) == 0:
            return

        ni_alpha = properties.get(nif.NiAlphaProperty)
        ni_material = properties.get(nif.NiMaterialProperty)
        ni_stencil = properties.get(nif.NiStencilProperty)
        ni_texture = properties.get(nif.NiTexturingProperty)
        ni_wireframe = properties.get(nif.NiWireframeProperty)

        # Vertex colors need a texturing property present to be visible.
        if (ni_texture is None) and len(self.source.data.vertex_colors):
            ni_texture = nif.NiTexturingProperty()

        # Blender stores wireframe on the object rather than a material.
        if ni_wireframe and ni_wireframe.wireframe:
            self.output.display_type = "WIRE"

        # Re-Use Materials
        name = self.calc_name_from_textures(ni_texture, ni_alpha)
        if not name:
            # Fall back to material property name if no texture name found
            ni_material_prop = properties.get(nif.NiMaterialProperty)
            if ni_material_prop and ni_material_prop.name:
                name = ni_material_prop.name

        if self.apply_existing_material(name, ni_alpha):
            return

        # Merge Duplicates
        props_hash = (
            *properties.values(),
            # "use_vertex_colors" is stored on the material
            len(self.source.data.vertex_colors),
            # uv animations are also stored on the material
            self.source.controllers.find_type(nif.NiUVController),
        )
        try:
            bl_prop = self.importer.materials[props_hash]
        except KeyError:
            bl_prop = self.importer.materials[props_hash] = nif_shader.execute(self.output)
        else:
            # material already exists, reuse it
            self.link_object_material(self.output, bl_prop.material)
            return
        finally:
            if self.importer.use_existing_materials:
                bl_prop.material.name = name

        # Setup Properties
        if ni_alpha:
            self.create_alpha_property(bl_prop, ni_alpha)
        if ni_material:
            self.create_material_property(bl_prop, ni_material)
        if ni_stencil:
            self.create_stencil_property(bl_prop, ni_stencil)
        if ni_texture:
            self.create_texturing_property(bl_prop, ni_texture)
        if ni_wireframe:
            self.create_wireframe_property(bl_prop, ni_wireframe)

    def create_alpha_property(self, bl_prop, ni_prop):
        # Alpha Flags
        bl_prop.alpha_flags = ni_prop.flags
        # Alpha Threshold
        bl_prop.material.alpha_threshold = float(ni_prop.test_ref / 255)
        # Blending Method
        if ni_prop.alpha_blending:
            bl_prop.use_alpha_blend = True
        if ni_prop.alpha_testing:
            bl_prop.use_alpha_clip = True

    def create_material_property(self, bl_prop, ni_prop):
        # Material Name
        if not self.importer.use_existing_materials:
            bl_prop.material.name = ni_prop.name
        # Material Color
        bl_prop.ambient_color[:3] = ni_prop.ambient_color
        bl_prop.diffuse_color[:3] = ni_prop.diffuse_color
        bl_prop.specular_color[:3] = ni_prop.specular_color
        # Respect importer option to ignore emissive color
        if not getattr(self.importer, "ignore_emissive_color", False):
            bl_prop.emissive_color[:3] = ni_prop.emissive_color
        # Material Shine
        bl_prop.shine = ni_prop.shine
        # Material Alpha
        bl_prop.alpha = ni_prop.alpha
        # Material Anims
        self.animation.create_color_controller(bl_prop, ni_prop)
        self.animation.create_alpha_controller(bl_prop, ni_prop)

    def create_texturing_property(self, bl_prop, ni_prop):
        # Texture Slots
        for name in nif.NiTexturingProperty.texture_keys:
            self.create_texturing_property_map(bl_prop, ni_prop, name)
        # Vertex Colors
        if self.output.data.vertex_colors:
            bl_prop.vertex_color.layer_name = self.output.data.vertex_colors[0].name
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Color", "Diffuse Color")
            bl_prop.create_link(bl_prop.vertex_color, bl_prop.shader, "Alpha", "Diffuse Alpha")
        # UV Animations
        for controller in self.source.controllers:
            if isinstance(controller, nif.NiUVController):
                self.animation.create_uv_controller(controller)

    def create_wireframe_property(self, bl_prop, ni_prop):
        if ni_prop.wireframe:
            self.output.display_type = "WIRE"

    def create_stencil_property(self, bl_prop, ni_prop):
        bl_prop.material.use_backface_culling = False
        bl_prop.material.show_transparent_back = True

    def create_texturing_property_map(self, bl_prop, ni_prop, slot_name):
        try:
            bl_slot = getattr(bl_prop, slot_name)
            ni_slot = getattr(ni_prop, slot_name)
            # only supports slots with texture image attached
            image = self.create_image(ni_slot.source.filename)
        except (AttributeError, LookupError):
            return

        # texture image
        bl_slot.image = image

        # use repeat
        if ni_slot.clamp_mode.name == 'CLAMP_S_CLAMP_T':
            bl_slot.use_repeat = False

        # use mipmaps
        if ni_slot.source.use_mipmaps.name == 'NO':
            bl_slot.use_mipmaps = False

        # uv layer
        try:
            bl_slot.layer = self.output.data.uv_layers[ni_slot.uv_set].name
        except IndexError:
            pass

    def create_image(self, filepath):
        abspath = self.resolve_texture_path(filepath)

        if abspath.exists():
            image = bpy.data.images.load(str(abspath), check_existing=True)
        else:  # placeholder
            image = bpy.data.images.new(name=abspath.name, width=1, height=1)
            image.filepath = str(abspath)
            image.source = "FILE"

        return image


    def calc_name_from_textures(self, ni_prop, ni_alpha=None):
        if not self.importer.use_existing_materials:
            return ""

        names = {}
        has_texture = False
        texture_path = None
        
        # Get texture names and UV settings
        if ni_prop is not None:
            for tex_key, tex_map in zip(ni_prop.texture_keys, ni_prop.texture_maps):
                # Skip decal_1, decal_2, etc., but keep decal_0 and all other keys
                if tex_key.startswith("decal_") and not tex_key.startswith("decal_0"):
                    continue
                try:
                    filepath = pathlib.Path(tex_map.source.filename)
                    texture_name = filepath.stem.lower()
                    
                    # Skip if texture name is empty
                    if not texture_name:
                        continue
                    
                    has_texture = True

                    # Capture the directory of the base texture only
                    if self.importer.use_texture_path_in_material_name and texture_path is None and tex_key == "base_texture":
                        parent_path = bpy.path.native_pathsep(str(filepath.parent))
                        parent_path = parent_path.lower()
                        parent = pathlib.Path(parent_path)
                        # Strip leading "textures\" prefix
                        try:
                            parent = parent.relative_to("textures")
                        except ValueError:
                            pass
                        if str(parent) != '.':
                            texture_path = str(parent)
                    
                    # Check for non-default UV settings
                    uv_flags = []
                    
                    # Check clamp mode (default is repeat/wrap)
                    if tex_map.clamp_mode.name == 'CLAMP_S_CLAMP_T':
                        uv_flags.append("Clamp")
                    
                    # Check UV set (default is 0, which becomes UVMap)
                    if tex_map.uv_set != 0:
                        uv_flags.append(f"UV{tex_map.uv_set}")
                    
                    # Build the texture entry
                    if uv_flags:
                        names[tex_key] = f"{texture_name}({','.join(uv_flags)})"
                    else:
                        names[tex_key] = texture_name
                        
                except AttributeError:
                    pass

        # Add material property information
        ni_material = self.properties.get(nif.NiMaterialProperty)
        if ni_material:
            # If no texture, use material property name as base
            if not has_texture and ni_material.name:
                names["property"] = ni_material.name
            
            # Check if vertex colors are used (only if NiTexturingProperty exists and vertex colors present)
            use_vertex_colors = ni_prop is not None and len(self.source.data.vertex_colors) > 0
            
            # Check diffuse color
            diffuse = ni_material.diffuse_color[:3]
            if use_vertex_colors:
                names["diffuse"] = "Col"
            elif not np.allclose(diffuse, [1.0, 1.0, 1.0], rtol=0, atol=1e-6):
                hex_color = "#{:02x}{:02x}{:02x}".format(
                    int(diffuse[0] * 255),
                    int(diffuse[1] * 255),
                    int(diffuse[2] * 255)
                )
                names["diffuse"] = hex_color
            
            # Check emissive color (skip if importer option requests it)
            if not getattr(self.importer, "ignore_emissive_color", False):
                emissive = ni_material.emissive_color[:3]
                if not np.allclose(emissive, [0.0, 0.0, 0.0], rtol=0, atol=1e-6):
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(emissive[0] * 255),
                        int(emissive[1] * 255),
                        int(emissive[2] * 255)
                    )
                    names["emissive"] = hex_color
            
            # Check alpha
            if not np.isclose(ni_material.alpha, 1.0, rtol=0, atol=1e-6):
                names["alpha"] = f"{ni_material.alpha:.3f}".rstrip('0').rstrip('.')

        # Add alpha property information
        if ni_alpha:
            alpha_info = ""
            if ni_alpha.alpha_testing:
                alpha_info += "t"
            elif ni_alpha.src_blend_mode == ni_alpha.AlphaBlendFunction.ONE and ni_alpha.dst_blend_mode == ni_alpha.AlphaBlendFunction.ONE:
                alpha_info += "a"
            elif ni_alpha.alpha_blending:
                alpha_info += "b"
            
            if alpha_info:
                if "alpha" in names:
                    names["alpha"] = f"{alpha_info} {names['alpha']}"
                else:
                    names["alpha"] = alpha_info

        # If only a single base texture with no other properties, return just the texture name
        if len(names) == 1 and "base_texture" in names:
            base = names["base_texture"]
            if texture_path:
                return sanitize_name(f"{base} | path:{texture_path}", normalize_prefix=self.importer.normalize_prefix, normalize_prefix_max_length=self.importer.normalize_prefix_max_length)
            return base

        # Build the final name, ensuring it's not empty
        final_name = " | ".join(f"{k.rpartition('_')[0] if '_' in k else k}:{v}" for k, v in names.items())

        # Append texture path if present
        if texture_path:
            final_name = f"{final_name} | path:{texture_path}"

        # Sanitize the final name to ensure it's safe for Blender
        return sanitize_name(final_name, normalize_prefix=self.importer.normalize_prefix, normalize_prefix_max_length=self.importer.normalize_prefix_max_length) if final_name else ""
        
    def apply_existing_material(self, name, ni_alpha):
        """
        Check if a material with the same name and properties already exists
        in the current blend file. If so, use it instead of making a new one.
        """
        if not self.importer.use_existing_materials:
            return

        use_vertex_colors = bool(len(self.source.data.vertex_colors))
        use_alpha_blend = getattr(ni_alpha, "alpha_blending", False)
        use_alpha_clip = getattr(ni_alpha, "alpha_testing", False)

        base_name, index = name, 0
        while True:
            try:
                bl_prop = bpy.data.materials[name].mw.validate()
            except (LookupError, TypeError):
                break

            if (
                bl_prop.use_vertex_colors == use_vertex_colors
                and bl_prop.use_alpha_blend == use_alpha_blend
                and bl_prop.use_alpha_clip == use_alpha_clip
            ):
                self.link_object_material(self.output, bl_prop.material)
                return True

            index += 1
            name = f"{base_name}.{index:03}"

    @staticmethod
    def link_object_material(bl_object, bl_material):
        # Use any existing empty material slot first.
        for slot in bl_object.material_slots:
            if slot.material is None:
                break
        else:
            bl_object.data.materials.append(None)
            slot = bl_object.material_slots[-1]

        slot.link = "OBJECT"
        slot.material = bl_material

    def resolve_texture_path(
        self,
        relpath,
        use_texture_fallbacks=None,
        case_insensitive = pathlib.Path(__file__.upper()).exists()
    ):
        # determine fallback preference
        if use_texture_fallbacks is None:
            use_texture_fallbacks = getattr(self.importer, "use_texture_fallbacks", True)

        # get the initial filepath (preserve original case and also a lowercase variant)
        orig_path = pathlib.Path(bpy.path.native_pathsep(relpath))
        path = pathlib.Path(bpy.path.native_pathsep(relpath).lower())

        # discard "data files" prefix from both
        if path.parts and path.parts[0] == "data files":
            path = path.relative_to("data files")
        if orig_path.parts and orig_path.parts[0].lower() == "data files":
            orig_path = orig_path.relative_to(orig_path.parts[0])

        # discard "textures" prefix from both
        if path.parts and path.parts[0] == "textures":
            path = path.relative_to("textures")
        if orig_path.parts and orig_path.parts[0].lower() == "textures":
            orig_path = orig_path.relative_to(orig_path.parts[0])

        # build ordered suffix list: try original first, then fallbacks if enabled
        orig = path.suffix.lower()
        if use_texture_fallbacks:
            # original extension first, then common alternatives (excluding original to avoid duplicates)
            fallbacks = [s for s in (".dds", ".tga", ".bmp") if s != orig]
            suffixes = ([orig] if orig else []) + fallbacks
        else:
            # only try the original extension (if present)
            suffixes = [orig] if orig else [""]

        # evaluate final image path
        # suffix is outer loop so the original extension is tried across ALL roots
        # before any fallback extension is attempted
        addon = bpy.context.preferences.addons[__package__]
        texture_paths = list(addon.preferences.texture_paths)
        for suffix in suffixes:
            for item in texture_paths:
                base_orig = item.name / orig_path
                base_low = item.name / path
                for base in (base_orig, base_low):
                    abspath = base.with_suffix(suffix)
                    if not case_insensitive:
                        try:
                            abspath = pathlib.Path(bpy.path.resolve_ncase(str(abspath)))
                        except Exception:
                            pass
                    if abspath.exists():
                        return abspath

        # not found; return the original relative path under 'textures'
        return ("textures" / path)


class Animation(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        if self.importer.ignore_animations:
            return

        bl_object = self.output.id_data

        if self.source.extra_data:
            self.create_text_keys(bl_object)

        if self.source.controller:
            self.create_kf_controller(bl_object)
            self.create_vis_controller(bl_object)

    def create_text_keys(self, bl_object):
        text_data = self.source.extra_datas.find_type(nif.NiTextKeyExtraData)
        if text_data is None:
            return

        action = self.get_action(bl_object)

        for frame, text in text_data.keys.tolist():
            for name in filter(None, text.splitlines()):
                action.pose_markers.new(name).frame = round(frame)

    def create_kf_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiKeyframeController)
        if not (controller and controller.data):
            return

        # get animation action
        action = self.get_action(bl_object)

        # translation keys
        self.create_translations(controller, action)
        # rotation keys
        self.create_rotations(controller, action)
        # scale keys
        self.create_scales(controller, action)

        self.update_frame_range(controller)

    def create_translations(self, controller, action):
        data = controller.data.translations
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("location")

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.bone_name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            self.create_interpolation_data(data, fc, axis=i)
            fc.update()

    def create_rotations(self, controller, action):
        if controller.data.rotations.euler_data:
            if isinstance(self.output, bpy.types.PoseBone):
                print(f"[INFO] Euler animations on bones are not currently supported. ({self.name})")
                controller.data.rotations.convert_to_quaternions()
            else:
                self.output.rotation_mode = controller.data.rotations.euler_axis_order.name
                self.create_euler_rotations(controller, action)
                return

        self.output.rotation_mode = 'QUATERNION'
        self.create_quaternion_rotations(controller, action)

    def create_euler_rotations(self, controller, action):
        for i, data in enumerate(controller.data.rotations.euler_data):
            if len(data.keys) == 0:
                continue

            # get blender data path
            data_path = self.output.path_from_id("rotation_euler")

            # build blender fcurves
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

    def create_quaternion_rotations(self, controller, action):
        data = controller.data.rotations
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("rotation_quaternion")

        # build blender fcurves
        for i in range(4):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, (0, i+1)].ravel())
            fc.update()

    def create_scales(self, controller, action):
        data = controller.data.scales
        if len(data.keys) == 0:
            return

        # get blender data path
        data_path = self.output.path_from_id("scale")

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=self.output.name)
            fc.keyframe_points.add(len(data.keys))
            fc.keyframe_points.foreach_set("co", data.keys[:, :2].ravel())
            self.create_interpolation_data(controller.data.scales, fc)
            fc.update()

    def create_vis_controller(self, bl_object):
        controller = self.source.controllers.find_type(nif.NiVisController)
        if controller is None:
            return

        data = controller.data
        if (data is None) or len(data.keys) == 0:
            return

        keys = np.empty((len(data.keys), 2), dtype=np.float32)

        # invert appculled flag
        keys[:, 1] = 1 - data.values

        # get animations action
        action = self.get_action(bl_object)

        # get blender data path
        try:
            data_path = self.output.path_from_id("hide_viewport")
        except AttributeError:
            print(f"Warning: NiVisController on bones are not supported ({self.name})")
            return

        # build blender fcurves
        fc = self.animation.create_fcurves(action, data_path, index=0, action_group=self.output.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys.ravel())
        fc.update()

    def create_uv_controller(self, controller):
        if self.importer.ignore_animations:
            return

        data = controller.data
        if data is None:
            return

        # get blender property
        try:
            bl_prop = self.output.active_material.mw
        except AttributeError:
            return

        # get animation action
        action = self.get_action(bl_prop.texture_group.node_tree)

        # get the texture slot
        try:
            uv_name = self.output.data.uv_layers[controller.texture_set].name
            bl_slot = next(s for s in bl_prop.texture_slots if s.layer == uv_name)
            bl_node = bl_slot.mapping_node
        except (IndexError, StopIteration):
            print("Warning: skipping NiUVController due to invalid texture set")
            return

        channels = {
            (data.u_offset_data, data.v_offset_data):
                bl_node.inputs["Location"].path_from_id("default_value"),
            (data.u_tiling_data, data.v_tiling_data):
                bl_node.inputs["Scale"].path_from_id("default_value"),
        }

        try:
            # TODO: do these in shader instead
            data.u_offset_data.keys[:, 1] *= -1
            data.v_offset_data.keys[:, 1] *= -1
        except AttributeError:
            pass

        for sources, data_path in channels.items():
            for i, uv_data in enumerate(sources):
                # skip empty or invalid key data
                try:
                    nkeys = len(uv_data.keys)
                except Exception:
                    continue
                if nkeys == 0:
                    continue

                # coerce to a contiguous float32 numpy array of shape (n,2)
                try:
                    arr2 = np.asarray(uv_data.keys[:, :2], dtype=np.float32)
                except Exception:
                    print(f"Warning: Invalid UV key data for {self.name}, channel {i}; skipping UV animation")
                    continue

                if arr2.ndim != 2 or arr2.shape[0] != nkeys or arr2.shape[1] != 2:
                    print(f"Warning: UV key shape mismatch for {self.name}, channel {i}; skipping UV animation")
                    continue

                flat = arr2.ravel()

                # build blender fcurves
                fc = self.animation.create_fcurves(action, data_path, index=i, action_group=uv_name)
                fc.keyframe_points.add(nkeys)
                try:
                    fc.keyframe_points.foreach_set("co", flat)
                except RuntimeError as e:
                    print(f"Warning: Failed to set UV keyframe points for {self.name}: {e}")
                    continue

                self.create_interpolation_data(uv_data, fc)
                fc.update()

        self.update_frame_range(controller)

    def create_color_controller(self, bl_prop, ni_prop):
        if self.importer.ignore_animations:
            return

        controller = ni_prop.controllers.find_type(nif.NiMaterialColorController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        keys = controller.data.keys
        if len(keys) == 0:
            return

        # get blender data path
        if controller.color_field == 'DIFFUSE':
            data_path = bl_prop.diffuse_input.path_from_id("default_value")
        elif controller.color_field == 'EMISSIVE':
            data_path = bl_prop.emissive_input.path_from_id("default_value")
        else:
            raise NotImplementedError(f"'{controller.color_field}' animations are not supported")

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # build blender fcurves
        for i in range(3):
            fc = self.animation.create_fcurves(action, data_path, index=i, action_group=bl_prop.material.name)
            fc.keyframe_points.add(len(keys))
            fc.keyframe_points.foreach_set("co", keys[:, (0, i+1)].ravel())
            self.create_interpolation_data(data, fc)
            fc.update()

        self.update_frame_range(controller)

    def create_alpha_controller(self, bl_prop, ni_prop):
        if self.importer.ignore_animations:
            return

        controller = ni_prop.controllers.find_type(nif.NiAlphaController)
        if controller is None:
            return

        data = controller.data
        if data is None:
            return

        keys = controller.data.keys
        if len(keys) == 0:
            return

        # create blender action
        action = self.get_action(bl_prop.material.node_tree)

        # get blender data path
        data_path = bl_prop.opacity_input.path_from_id("default_value")

        # build blender fcurves
        fc = self.animation.create_fcurves(action, data_path, index=0, action_group=bl_prop.material.name)
        fc.keyframe_points.add(len(keys))
        fc.keyframe_points.foreach_set("co", keys[:, :2].ravel())
        self.create_interpolation_data(data, fc)
        fc.update()

        self.update_frame_range(controller)

    @staticmethod
    def get_action(bl_object):
        try:
            action = bl_object.animation_data.action
        except AttributeError:
            action = bpy.data.actions.new(f"{bl_object.name}Action")
            anim_data = bl_object.animation_data_create()
            anim_data.action = action

            if bpy.app.version >= (4, 4, 0):
                anim_data.action_slot = action.slots.new(id_type=bl_object.id_type, name=bl_object.name)

        return action
    
    @staticmethod
    def create_fcurves(action, data_path, index=0, action_group=""):
        if bpy.app.version >= (5, 0, 0):
            from bpy_extras import anim_utils
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, *action.slots)
            return channelbag.fcurves.ensure(data_path, index=index, group_name=action_group)
        else:
            return action.fcurves.new(data_path, index=index, action_group=action_group)

    @staticmethod
    def create_interpolation_data(ni_data, fcurves, axis=...):
        if ni_data.key_type.name  == 'LIN_KEY':
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'LINEAR'
        else:
            for kp in fcurves.keyframe_points:
                kp.interpolation = 'BEZIER'
                kp.handle_left_type = kp.handle_right_type = 'FREE'
            handles = ni_data.get_tangent_handles()  # TODO: call this once per controller rather than per axis
            fcurves.keyframe_points.foreach_set("handle_left", handles[0, axis].ravel())
            fcurves.keyframe_points.foreach_set("handle_right", handles[1, axis].ravel())

    @staticmethod
    def update_frame_range(controller):
        scene = bpy.context.scene
        frame_end = int(np.ceil(controller.stop_time))
        scene.frame_end = scene.frame_preview_end = max(scene.frame_end, frame_end)

    def set_mute(self, state, fcurves=None):
        if self.importer.ignore_animations:
            return

        if fcurves is None:
            try:
                fcurves = self.output.id_data.animation_data.action.fcurves
            except AttributeError:
                return
        fcurves.foreach_set("mute", [state] * len(fcurves))

class Light(SceneNode):
    __slots__ = ()

    def __init__(self, node):
        self.__dict__ = node.__dict__

    def create(self):
        # Determine light type from NIF data
        light_type = self.get_light_type()
        
        # Create Blender light data
        bl_data = bpy.data.lights.new(self.name, light_type)
        bl_object = Empty(self).create(bl_data)
        
        # Set light properties from NIF data
        self.setup_light_properties(bl_data)
        
        return bl_object
    
    def get_light_type(self):
        # Map NIF light types to Blender light types
        nif_type = self.source.type
        if "Directional" in nif_type:
            return 'SUN'
        elif "Point" in nif_type:
            return 'POINT'
        elif "Spot" in nif_type:
            return 'SPOT'
        else:
            return 'POINT'  # default
    
    def setup_light_properties(self, bl_data):
        # Set properties based on NIF light data
        if hasattr(self.source, 'diffuse_color'):
            bl_data.color = self.source.diffuse_color[:3]
        if hasattr(self.source, 'intensity'):
            bl_data.energy = self.source.intensity
        # Add other properties as needed