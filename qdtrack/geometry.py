"""A2D2 rig -> Qwen-Drive lidar/ego/image frames.

Qwen-Drive expects a pinhole camera whose axes are OpenCV (x right, y down,
z forward) and an ego frame with x forward, y left, z up. A2D2 stores each
sensor as an origin plus x/y axes in the vehicle frame, and its camera x axis
is the optical axis (forward), y is left, z is up. Images in the release are
distorted; CamMatrix is the pinhole model of the undistorted image.
"""

from __future__ import annotations

import numpy as np

# nuScenes 6-camera prompt order. A2D2 does not have this angular layout:
# its three "front" cameras all look forward, and the side cameras sit near
# +/-90 deg rather than nuScenes' back-left / back-right. The mapping below
# is the closest slot for each physical camera. That mismatch is the point
# of the run: the released head was trained on a fixed rig.
CAM_SLOTS = (
    # qwen name, a2d2 key, filename tag, view tag
    ("CAM_FRONT", "front_center", "frontcenter", "<FRONT VIEW>"),
    ("CAM_FRONT_RIGHT", "front_right", "frontright", "<FRONT RIGHT VIEW>"),
    ("CAM_BACK_RIGHT", "side_right", "sideright", "<BACK RIGHT VIEW>"),
    ("CAM_BACK", "rear_center", "rearcenter", "<BACK VIEW>"),
    ("CAM_BACK_LEFT", "side_left", "sideleft", "<BACK LEFT VIEW>"),
    ("CAM_FRONT_LEFT", "front_left", "frontleft", "<FRONT LEFT VIEW>"),
)

# OpenCV point -> A2D2 camera point (x forward, y left, z up).
_CV_TO_A2D = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _axes(view: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(view["x-axis"], dtype=np.float64)
    y = np.asarray(view["y-axis"], dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=1)
    origin = np.asarray(view["origin"], dtype=np.float64)
    return rotation, origin


def sensor_to_ego(view: dict) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-ego rotation and translation for an OpenCV camera frame.

    P_ego = R @ P_cv + t
    """
    rotation_a2d, origin = _axes(view)
    rotation = rotation_a2d @ _CV_TO_A2D
    return rotation, origin


def build_lidar2img(intrinsic: np.ndarray, rotation_c2e: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Same composition as qwen_drive_perception.geometry.build_lidar2img.

    sensor2lidar_rotation / translation are camera-to-lidar. Here lidar == ego.
    """
    lidar2cam_r = np.linalg.inv(rotation_c2e)
    lidar2cam_t = translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = intrinsic
    return viewpad @ lidar2cam_rt.T


def scale_lidar2img(lidar2img: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
    out = np.array(lidar2img, dtype=np.float64, copy=True)
    scale = np.array([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]], dtype=np.float64)
    out[:3, :] = scale @ out[:3, :]
    return out


def undistort(image: np.ndarray, camera: dict):
    import cv2

    distorted_k = np.asarray(camera["CamMatrixOriginal"], dtype=np.float64)
    undistorted_k = np.asarray(camera["CamMatrix"], dtype=np.float64)
    distortion = np.asarray(camera["Distortion"], dtype=np.float64)
    if camera["Lens"] == "Fisheye":
        return cv2.fisheye.undistortImage(image, distorted_k, D=distortion, Knew=undistorted_k)
    return cv2.undistort(image, distorted_k, distortion.reshape(-1), newCameraMatrix=undistorted_k)


def pack_calibration(cameras: dict) -> dict:
    """Per-slot intrinsics and camera-to-ego, lidar frame identical to ego."""
    intrinsics, rotations, translations = [], [], []
    slots = []
    for qwen_name, a2d2_key, file_tag, view_tag in CAM_SLOTS:
        camera = cameras[a2d2_key]
        rotation, translation = sensor_to_ego(camera["view"])
        intrinsic = np.asarray(camera["CamMatrix"], dtype=np.float64)
        intrinsics.append(intrinsic)
        rotations.append(rotation)
        translations.append(translation)
        yaw = float(np.degrees(np.arctan2(camera["view"]["x-axis"][1], camera["view"]["x-axis"][0])))
        slots.append(
            {
                "qwen": qwen_name,
                "a2d2": a2d2_key,
                "file_tag": file_tag,
                "view_tag": view_tag,
                "lens": camera["Lens"],
                "optical_yaw_deg": round(yaw, 2),
                "resolution": camera["Resolution"],
            }
        )
    lidar2ego = np.eye(4, dtype=np.float64)
    return {
        "slots": slots,
        "cam_intrinsic": np.stack(intrinsics).astype(np.float32),
        "sensor2lidar_rotation": np.stack(rotations).astype(np.float32),
        "sensor2lidar_translation": np.stack(translations).astype(np.float32),
        "lidar2ego": lidar2ego.astype(np.float32),
    }


def frame_json(token: str) -> dict:
    content = []
    cam_order = []
    for qwen_name, _key, _tag, view_tag in CAM_SLOTS:
        content.append({"text": view_tag})
        content.append({"image": qwen_name})
        cam_order.append(qwen_name)
    content.append({"text": "Analyze the scene."})
    return {
        "dataset_type": "nuscenes",
        "cam_order": cam_order,
        "content": content,
        "token": token,
        "source": "a2d2",
    }
