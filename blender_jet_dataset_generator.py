# Blender Synthetic Jet Dataset Generator 1
# Tested by syntax only outside Blender. Run this INSIDE Blender.
#
# Output structure:
# outpouts/
#   images/
#   labels/
#   metadata/
#   classes.txt
#
# Labels are YOLO object-detection labels:
# class_id center_x center_y width height  (all normalized 0..1)

import bpy
import math
import random
import json
import traceback
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


# ============================================================
#                       USER SETTINGS
# ============================================================

MODEL_FOLDER = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\models\extracted")
ENV_FOLDER = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\spaces")
OUTPUT_FOLDER = Path(r"C:\Users\tahas\Pictures\link din post\Make dataset with blender\outpouts")

# 1) Maximum number of images before the environment MUST change
MAX_IMAGES_PER_ENV = 10

# 2) Number of images for each subject/model before moving to next subject
IMAGES_PER_OBJECT = 5

# 3) Total number of images to generate
TOTAL_IMAGES = 100

# Image settings
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
JPEG_QUALITY = 95

# Randomness
RANDOM_SEED = 1337

# Normalize all imported models to roughly this largest dimension in Blender units
TARGET_MODEL_SIZE = 4.0

# Camera variation
CAMERA_ELEVATION_MIN_DEG = -20.0
CAMERA_ELEVATION_MAX_DEG = 35.0
CAMERA_FOCAL_MIN_MM = 35.0
CAMERA_FOCAL_MAX_MM = 75.0
CAMERA_SHIFT_X = 0.16
CAMERA_SHIFT_Y = 0.12

# Keep projected object bbox within reasonable dataset size
MIN_BBOX_AREA = 0.015   # 3.5% of image
MAX_BBOX_AREA = 0.80    # 65% of image
MAX_CAMERA_ATTEMPTS = 50

DESIRED_FILL_MIN = 0.15
DESIRED_FILL_MAX = 0.75

CAMERA_DISTANCE_SCALE_MIN = 0.55
CAMERA_DISTANCE_SCALE_MAX = 1.90


# Subject pose variation (kept physically plausible-ish for jets)
MAX_PITCH_DEG = 15.0
MAX_ROLL_DEG = 30.0


# HDRI variation
HDRI_STRENGTH_MIN = 0.7
HDRI_STRENGTH_MAX = 1.3

# Output extras
SAVE_YOLO_LABELS = True
SAVE_METADATA_JSON = True

# Optional: add a subtle sun for extra directional light variation.
# HDRI still remains the main environment lighting.
USE_EXTRA_SUN = True
SUN_ENERGY_MIN = 0.3
SUN_ENERGY_MAX = 1.5
SUN_ANGLE_DEG = 5.0


# ============================================================
#                         CONSTANTS
# ============================================================

MODEL_EXTS = {".glb", ".gltf", ".fbx", ".obj", ".stl", ".blend"}
ENV_EXTS = {".exr", ".hdr"}

DATASET_CAMERA_NAME = "__DATASET_CAMERA__"
DATASET_SUN_NAME = "__DATASET_SUN__"
SUBJECT_ROOT_NAME = "__SUBJECT_ROOT__"
WORLD_NAME = "__DATASET_WORLD__"


# ============================================================
#                         UTILITIES
# ============================================================

def log(msg):
    print(f"[DATASET] {msg}")


def scan_files(folder: Path, extensions):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions],
        key=lambda p: str(p).lower()
    )


def safe_name(path: Path):
    name = path.stem.strip().replace(" ", "_")
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)


def ensure_output_dirs():
    images_dir = OUTPUT_FOLDER / "images"
    labels_dir = OUTPUT_FOLDER / "labels"
    metadata_dir = OUTPUT_FOLDER / "metadata"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return images_dir, labels_dir, metadata_dir


def purge_orphans():
    # Safe memory cleanup between model changes.
    try:
        bpy.data.orphans_purge(do_recursive=True)
    except Exception:
        # Older Blender fallback: not critical.
        pass


def remove_object(obj):
    if obj and obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)


def remove_objects(objects):
    for obj in list(objects):
        if obj and obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def clean_previous_dataset_helpers():
    for name in (DATASET_CAMERA_NAME, DATASET_SUN_NAME, SUBJECT_ROOT_NAME):
        obj = bpy.data.objects.get(name)
        if obj:
            remove_object(obj)


# ============================================================
#                         IMPORTERS
# ============================================================

def import_model_file(filepath: Path):
    """
    Imports one model and returns only the objects created by that import.
    Supports GLB/GLTF/FBX/OBJ/STL/BLEND.
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    before_ptrs = {obj.as_pointer() for obj in bpy.data.objects}

    # Make sure old selection does not confuse import operators.
    bpy.ops.object.select_all(action='DESELECT')

    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(
            filepath=str(filepath),
            import_pack_images=True
        )

    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(filepath))

    elif ext == ".obj":
        # Blender 3.3+ / 4.x / 5.x
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(filepath))
        else:
            bpy.ops.import_scene.obj(filepath=str(filepath))

    elif ext == ".stl":
        # STL has geometry only; it normally has no materials/textures.
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=str(filepath))
        else:
            bpy.ops.import_mesh.stl(filepath=str(filepath))

    elif ext == ".blend":
        with bpy.data.libraries.load(str(filepath), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)

        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)

    else:
        raise ValueError(f"Unsupported model format: {ext}")

    bpy.context.view_layer.update()

    imported = [
        obj for obj in bpy.data.objects
        if obj.as_pointer() not in before_ptrs
    ]

    # Model files sometimes contain their own cameras/lights.
    # We do not want them influencing the synthetic dataset.
    unwanted = [obj for obj in imported if obj.type in {"CAMERA", "LIGHT"}]
    remove_objects(unwanted)

    imported = [
        obj for obj in imported
        if obj.name in bpy.data.objects
    ]

    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        remove_objects(imported)
        raise RuntimeError("Model imported, but no MESH object was found.")

    return imported


def validate_model(filepath: Path):
    imported = []
    try:
        imported = import_model_file(filepath)
        mesh_count = sum(1 for obj in imported if obj.type == "MESH")
        return True, f"{mesh_count} mesh(es)"
    except Exception as e:
        return False, str(e)
    finally:
        remove_objects(imported)
        purge_orphans()


def validate_environment(filepath: Path):
    img = None
    try:
        img = bpy.data.images.load(str(filepath), check_existing=False)
        if img.size[0] <= 0 or img.size[1] <= 0:
            raise RuntimeError("Image loaded but has invalid dimensions.")
        return True, f"{img.size[0]}x{img.size[1]}"
    except Exception as e:
        return False, str(e)
    finally:
        if img and img.name in bpy.data.images:
            bpy.data.images.remove(img)


# ============================================================
#                    SUBJECT MANAGEMENT
# ============================================================

def create_subject_root(imported_objects):
    root = bpy.data.objects.new(SUBJECT_ROOT_NAME, None)
    bpy.context.collection.objects.link(root)

    imported_set = set(imported_objects)

    # Parent only top-level imported objects, preserving world transforms.
    for obj in imported_objects:
        if obj.parent not in imported_set:
            world_matrix = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world_matrix

    return root


def get_mesh_objects_under(root):
    result = []
    for obj in bpy.context.scene.objects:
        parent = obj.parent
        while parent is not None:
            if parent == root:
                if obj.type == "MESH":
                    result.append(obj)
                break
            parent = parent.parent
    return result


def world_bbox(mesh_objects):
    points = []
    for obj in mesh_objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))

    if not points:
        raise RuntimeError("Could not calculate model bounding box.")

    min_v = Vector((
        min(p.x for p in points),
        min(p.y for p in points),
        min(p.z for p in points),
    ))
    max_v = Vector((
        max(p.x for p in points),
        max(p.y for p in points),
        max(p.z for p in points),
    ))

    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    max_dim = max(size.x, size.y, size.z)

    return min_v, max_v, center, size, max_dim


def normalize_subject(root):
    bpy.context.view_layer.update()
    mesh_objects = get_mesh_objects_under(root)
    _, _, center, _, max_dim = world_bbox(mesh_objects)

    if max_dim <= 1e-8:
        raise RuntimeError("Model has zero/invalid size.")

    scale_factor = TARGET_MODEL_SIZE / max_dim
    root.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()

    # Recompute after scaling, then center the model at world origin.
    _, _, center2, _, _ = world_bbox(mesh_objects)
    root.location -= center2
    bpy.context.view_layer.update()

    return mesh_objects


def load_subject(filepath: Path):
    delete_subject()

    imported = import_model_file(filepath)
    root = create_subject_root(imported)
    mesh_objects = normalize_subject(root)

    return root, mesh_objects


def delete_subject():
    root = bpy.data.objects.get(SUBJECT_ROOT_NAME)
    if not root:
        return

    # Gather the whole hierarchy first.
    descendants = []
    for obj in list(bpy.data.objects):
        parent = obj.parent
        while parent:
            if parent == root:
                descendants.append(obj)
                break
            parent = parent.parent

    remove_objects(descendants)
    remove_object(root)
    purge_orphans()


# ============================================================
#                     WORLD / ENVIRONMENT
# ============================================================

def ensure_world():
    world = bpy.data.worlds.get(WORLD_NAME)
    if world is None:
        world = bpy.data.worlds.new(WORLD_NAME)

    world.use_nodes = True
    bpy.context.scene.world = world

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    env_tex = nodes.new("ShaderNodeTexEnvironment")
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")

    env_tex.name = "__HDRI_TEXTURE__"
    mapping.name = "__HDRI_MAPPING__"
    background.name = "__HDRI_BACKGROUND__"

    # Reflection coordinates are appropriate for environment maps.
    links.new(texcoord.outputs["Reflection"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    return world


def load_environment(filepath: Path):
    world = ensure_world()
    nodes = world.node_tree.nodes

    env_tex = nodes.get("__HDRI_TEXTURE__")
    mapping = nodes.get("__HDRI_MAPPING__")
    background = nodes.get("__HDRI_BACKGROUND__")

    # Free previous HDRI image if one is already attached.
    old_image = env_tex.image
    env_tex.image = None
    if old_image and old_image.name in bpy.data.images:
        bpy.data.images.remove(old_image)

    image = bpy.data.images.load(str(filepath), check_existing=False)
    env_tex.image = image
    env_tex.projection = 'EQUIRECTANGULAR'

    # Randomize HDRI rotation and intensity.
    mapping.inputs["Rotation"].default_value[2] = random.uniform(0.0, math.tau)
    background.inputs["Strength"].default_value = random.uniform(
        HDRI_STRENGTH_MIN,
        HDRI_STRENGTH_MAX
    )

    return {
        "hdri_rotation_z": mapping.inputs["Rotation"].default_value[2],
        "hdri_strength": background.inputs["Strength"].default_value,
    }


# ============================================================
#                       CAMERA / LIGHT
# ============================================================

def ensure_camera():
    cam_obj = bpy.data.objects.get(DATASET_CAMERA_NAME)
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(DATASET_CAMERA_NAME)
        cam_obj = bpy.data.objects.new(DATASET_CAMERA_NAME, cam_data)
        bpy.context.collection.objects.link(cam_obj)

    bpy.context.scene.camera = cam_obj
    return cam_obj


def ensure_sun():
    if not USE_EXTRA_SUN:
        old = bpy.data.objects.get(DATASET_SUN_NAME)
        if old:
            remove_object(old)
        return None

    sun_obj = bpy.data.objects.get(DATASET_SUN_NAME)
    if sun_obj is None:
        sun_data = bpy.data.lights.new(DATASET_SUN_NAME, type='SUN')
        sun_obj = bpy.data.objects.new(DATASET_SUN_NAME, sun_data)
        bpy.context.collection.objects.link(sun_obj)

    sun_obj.data.angle = math.radians(SUN_ANGLE_DEG)
    return sun_obj


def randomize_sun(sun_obj):
    if sun_obj is None:
        return {}

    sun_obj.rotation_euler = (
        random.uniform(0.0, math.pi),
        random.uniform(0.0, math.pi),
        random.uniform(0.0, math.tau)
    )
    sun_obj.data.energy = random.uniform(SUN_ENERGY_MIN, SUN_ENERGY_MAX)

    return {
        "sun_energy": sun_obj.data.energy,
        "sun_rotation": [float(v) for v in sun_obj.rotation_euler],
    }


def look_at(camera, target: Vector):
    direction = target - camera.location
    if direction.length <= 1e-8:
        return
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def randomize_subject_pose(root):
    # Full yaw, modest pitch/roll.
    pitch = math.radians(random.uniform(-MAX_PITCH_DEG, MAX_PITCH_DEG))
    roll = math.radians(random.uniform(-MAX_ROLL_DEG, MAX_ROLL_DEG))
    yaw = random.uniform(0.0, math.tau)

    root.rotation_euler = (pitch, roll, yaw)
    bpy.context.view_layer.update()

    return {
        "subject_pitch_deg": math.degrees(pitch),
        "subject_roll_deg": math.degrees(roll),
        "subject_yaw_deg": math.degrees(yaw),
    }


def projected_bbox(scene, camera, mesh_objects, clamp=True):
    """
    Calculates a 2D normalized bounding box from all mesh bound-box corners.
    Coordinates returned are top-left-origin compatible for YOLO after conversion.
    """
    coords = []

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_co = obj.matrix_world @ Vector(corner)
            ndc = world_to_camera_view(scene, camera, world_co)

            # z <= 0 is behind the camera.
            if ndc.z <= 0:
                continue

            coords.append((ndc.x, ndc.y))

    if not coords:
        return None

    x1 = min(x for x, y in coords)
    x2 = max(x for x, y in coords)
    y1 = min(y for x, y in coords)
    y2 = max(y for x, y in coords)

    if clamp:
        x1 = max(0.0, min(1.0, x1))
        x2 = max(0.0, min(1.0, x2))
        y1 = max(0.0, min(1.0, y1))
        y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_area(bbox):
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def randomize_camera_for_subject(camera, mesh_objects):
    scene = bpy.context.scene

    _, _, center, size, max_dim = world_bbox(mesh_objects)

    for attempt in range(MAX_CAMERA_ATTEMPTS):
        focal = random.uniform(CAMERA_FOCAL_MIN_MM, CAMERA_FOCAL_MAX_MM)
        camera.data.lens = focal
        camera.data.sensor_width = 36.0

        desired_fill = random.uniform(DESIRED_FILL_MIN, DESIRED_FILL_MAX)
        angle_x = 2.0 * math.atan(camera.data.sensor_width / (2.0 * focal))

        denom = max(1e-4, 2.0 * math.tan(angle_x / 2.0) * desired_fill)
        base_distance = max_dim / denom

        # این بخش باعث می‌شود فاصله واقعاً نزدیک/دور شود
        distance_scale = random.uniform(
            CAMERA_DISTANCE_SCALE_MIN,
            CAMERA_DISTANCE_SCALE_MAX
        )
        distance = base_distance * distance_scale

        azimuth = random.uniform(0.0, math.tau)
        elevation = math.radians(
            random.uniform(CAMERA_ELEVATION_MIN_DEG, CAMERA_ELEVATION_MAX_DEG)
        )

        camera.location = center + Vector((
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ))

        target = center + Vector((
            random.uniform(-0.12, 0.12) * max_dim,
            random.uniform(-0.12, 0.12) * max_dim,
            random.uniform(-0.10, 0.10) * max_dim,
        ))
        look_at(camera, target)

        camera.data.shift_x = random.uniform(-CAMERA_SHIFT_X, CAMERA_SHIFT_X)
        camera.data.shift_y = random.uniform(-CAMERA_SHIFT_Y, CAMERA_SHIFT_Y)

        bpy.context.view_layer.update()

        bbox = projected_bbox(scene, camera, mesh_objects, clamp=True)
        area = bbox_area(bbox)

        if bbox is not None and MIN_BBOX_AREA <= area <= MAX_BBOX_AREA:
            x1, y1, x2, y2 = bbox
            if (x2 - x1) > 0.05 and (y2 - y1) > 0.05:
                return bbox, {
                    "camera_distance": float(distance),
                    "camera_distance_scale": float(distance_scale),
                    "desired_fill": float(desired_fill),
                    "camera_focal_mm": float(focal),
                    "camera_azimuth_deg": math.degrees(azimuth),
                    "camera_elevation_deg": math.degrees(elevation),
                    "camera_shift_x": float(camera.data.shift_x),
                    "camera_shift_y": float(camera.data.shift_y),
                }

    bbox = projected_bbox(scene, camera, mesh_objects, clamp=True)
    if bbox is None:
        raise RuntimeError("Could not find a valid camera view for this model.")

    return bbox, {
        "camera_distance": float((camera.location - center).length),
        "camera_focal_mm": float(camera.data.lens),
        "camera_shift_x": float(camera.data.shift_x),
        "camera_shift_y": float(camera.data.shift_y),
        "camera_warning": "fallback_after_max_attempts",
    }
# ============================================================
#                          LABELS
# ============================================================

def bbox_to_yolo(bbox):
    """
    Blender NDC y=0 is bottom. YOLO y=0 is top, so invert Y center.
    """
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) * 0.5
    cy_blender = (y1 + y2) * 0.5
    cy = 1.0 - cy_blender

    width = x2 - x1
    height = y2 - y1

    return cx, cy, width, height


def write_yolo_label(label_path: Path, class_id: int, bbox):
    cx, cy, w, h = bbox_to_yolo(bbox)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(
            f"{class_id} "
            f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )


# ============================================================
#                         RENDERING
# ============================================================

def choose_render_engine(scene):
    # Prefer current EEVEE. Fall back if running an older Blender.
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine
            log(f"Render engine: {engine}")
            return
        except Exception:
            pass

    log("EEVEE unavailable, falling back to CYCLES.")
    scene.render.engine = "CYCLES"


def configure_scene():
    scene = bpy.context.scene

    choose_render_engine(scene)

    scene.render.resolution_x = IMAGE_WIDTH
    scene.render.resolution_y = IMAGE_HEIGHT
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = JPEG_QUALITY

    # No transparent background: keep the HDRI visible.
    scene.render.film_transparent = False

    # Better default color management for modern Blender.
    try:
        scene.view_settings.view_transform = 'AgX'
    except Exception:
        try:
            scene.view_settings.view_transform = 'Filmic'
        except Exception:
            pass

    # Slightly more deterministic output.
    scene.render.use_file_extension = True


def render_one(image_path: Path):
    scene = bpy.context.scene
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)


# ============================================================
#                         MAIN LOGIC
# ============================================================

def validate_assets(model_files, env_files):
    valid_models = []
    valid_envs = []

    log("=" * 65)
    log("VALIDATING 3D MODELS")
    log("=" * 65)

    for path in model_files:
        ok, info = validate_model(path)
        if ok:
            valid_models.append(path)
            log(f"[OK] MODEL: {path.name}  -> {info}")
        else:
            log(f"[BAD] MODEL: {path.name} -> {info}")

    log("=" * 65)
    log("VALIDATING ENVIRONMENTS")
    log("=" * 65)

    for path in env_files:
        ok, info = validate_environment(path)
        if ok:
            valid_envs.append(path)
            log(f"[OK] ENV:   {path.name}  -> {info}")
        else:
            log(f"[BAD] ENV:   {path.name} -> {info}")

    log("=" * 65)
    log(
        f"READY: {len(valid_models)} valid object(s), "
        f"{len(valid_envs)} valid environment(s)"
    )
    log("=" * 65)

    return valid_models, valid_envs


def write_classes_file(models):
    classes_path = OUTPUT_FOLDER / "classes.txt"
    with open(classes_path, "w", encoding="utf-8") as f:
        for model in models:
            f.write(safe_name(model) + "\n")
    return classes_path


def main():
    random.seed(RANDOM_SEED)

    if MAX_IMAGES_PER_ENV <= 0:
        raise ValueError("MAX_IMAGES_PER_ENV must be > 0")
    if IMAGES_PER_OBJECT <= 0:
        raise ValueError("IMAGES_PER_OBJECT must be > 0")
    if TOTAL_IMAGES <= 0:
        raise ValueError("TOTAL_IMAGES must be > 0")

    images_dir, labels_dir, metadata_dir = ensure_output_dirs()

    model_files = scan_files(MODEL_FOLDER, MODEL_EXTS)
    env_files = scan_files(ENV_FOLDER, ENV_EXTS)

    log(f"Found {len(model_files)} candidate model file(s).")
    log(f"Found {len(env_files)} candidate environment file(s).")

    if not model_files:
        raise RuntimeError(f"No model files found in: {MODEL_FOLDER}")
    if not env_files:
        raise RuntimeError(f"No EXR/HDR environment files found in: {ENV_FOLDER}")

    clean_previous_dataset_helpers()
    configure_scene()

    valid_models, valid_envs = validate_assets(model_files, env_files)

    if not valid_models:
        raise RuntimeError("No valid 3D models are available.")
    if not valid_envs:
        raise RuntimeError("No valid HDRI environments are available.")

    class_id_by_path = {str(p): i for i, p in enumerate(valid_models)}
    classes_path = write_classes_file(valid_models)

    log(f"Classes written to: {classes_path}")

    camera = ensure_camera()
    sun = ensure_sun()

    total_done = 0

    env_cursor = 0
    env_images_remaining = 0
    current_env = None
    env_meta = {}

    model_cursor = 0
    object_images_remaining = IMAGES_PER_OBJECT
    current_model = None
    subject_root = None
    mesh_objects = None

    # --------------------------------------------------------
    # Scheduling guarantees:
    # - Each environment is used for at most MAX_IMAGES_PER_ENV.
    # - Each model batch gets exactly IMAGES_PER_OBJECT,
    #   unless TOTAL_IMAGES ends in the middle of the final batch.
    # - If an environment ends during a model batch, the model is
    #   removed, the environment changes, then that same model batch
    #   continues in the next environment.
    # - Environments and models both cycle forever until TOTAL_IMAGES.
    # --------------------------------------------------------

    while total_done < TOTAL_IMAGES:

        # Need a new environment?
        if env_images_remaining <= 0:
            delete_subject()
            subject_root = None
            mesh_objects = None
            current_model = None

            current_env = valid_envs[env_cursor % len(valid_envs)]
            env_cursor += 1
            env_images_remaining = MAX_IMAGES_PER_ENV

            env_meta = load_environment(current_env)

            log(
                f"\nENV -> {current_env.name} "
                f"(up to {MAX_IMAGES_PER_ENV} image(s))"
            )

        # Need a model loaded?
        if subject_root is None:
            current_model = valid_models[model_cursor % len(valid_models)]
            subject_root, mesh_objects = load_subject(current_model)

            log(
                f"OBJECT -> {current_model.name} "
                f"({object_images_remaining} image(s) remaining in this object batch)"
            )

        # Randomize subject, camera and light for every image.
        pose_meta = randomize_subject_pose(subject_root)
        bbox, camera_meta = randomize_camera_for_subject(camera, mesh_objects)
        sun_meta = randomize_sun(sun)

        class_id = class_id_by_path[str(current_model)]
        class_name = safe_name(current_model)
        env_name = safe_name(current_env)

        image_index = total_done + 1
        stem = (
            f"{image_index:06d}"
            f"__{class_name}"
            f"__{env_name}"
        )

        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        metadata_path = metadata_dir / f"{stem}.json"

        render_one(image_path)

        if SAVE_YOLO_LABELS:
            write_yolo_label(label_path, class_id, bbox)

        if SAVE_METADATA_JSON:
            meta = {
                "image_index": image_index,
                "image_file": image_path.name,
                "model_file": str(current_model),
                "environment_file": str(current_env),
                "class_id": class_id,
                "class_name": class_name,
                "bbox_normalized_blender": [float(v) for v in bbox],
                "yolo_bbox": [float(v) for v in bbox_to_yolo(bbox)],
                **env_meta,
                **pose_meta,
                **camera_meta,
                **sun_meta,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        total_done += 1
        env_images_remaining -= 1
        object_images_remaining -= 1

        log(
            f"[{total_done:04d}/{TOTAL_IMAGES}] "
            f"saved {image_path.name}"
        )

        # Object batch completed: delete it and advance model.
        if object_images_remaining <= 0:
            delete_subject()
            subject_root = None
            mesh_objects = None
            current_model = None

            model_cursor = (model_cursor + 1) % len(valid_models)
            object_images_remaining = IMAGES_PER_OBJECT

        # If environment quota ended, the top of the loop will
        # delete any currently loaded subject and switch HDRI.
        # If a model batch was incomplete, the same model_cursor and
        # object_images_remaining are preserved and continue there.

    delete_subject()

    log("\n" + "=" * 65)
    log("DATASET COMPLETE")
    log(f"Generated images: {total_done}")
    log(f"Images:   {images_dir}")
    log(f"Labels:   {labels_dir}")
    log(f"Metadata: {metadata_dir}")
    log(f"Classes:  {classes_path}")
    log("=" * 65)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL ERROR")
        traceback.print_exc()
        raise
