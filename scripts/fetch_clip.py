#!/usr/bin/env python3
"""Download one A2D2 surround clip and pack it for Qwen-Drive + the eval UI.

Default scene is the Gaimersheim drive, a marked road at about 66 km/h.
The Munich window is the cobblestone square and stays available as --scene munich.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.geometry import CAM_SLOTS, build_lidar2img, pack_calibration, scale_lidar2img, undistort  # noqa: E402
from qdtrack.tracks import integrate_ego  # noqa: E402

S3 = "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com"
SCENES = {
    "gaimersheim": {
        "scene": "camera_lidar/20180810_150607",
        "stamp": "20180810150607",
        "city": "Gaimersheim",
        "title": "Gaimersheim road",
        "start": 12025,
        "blurb": (
            "About 66 km/h on a marked two-lane road: solid edge line, dashed lane line, "
            "cars ahead. This replaces the Munich cobblestone square."
        ),
    },
    "munich": {
        "scene": "camera_lidar/20190401_121727",
        "stamp": "20190401121727",
        "city": "Munich",
        "title": "Munich ring",
        "start": 8472,
        "blurb": (
            "The head was trained on a fixed nuScenes 6-camera ring and an OpenScene 8-camera ring. "
            "A2D2 is a different six-camera Audi rig."
        ),
    },
}
DEFAULT_COUNT = 20
DEFAULT_STRIDE = 6  # 5 Hz
CAM_DIR = {
    "frontcenter": "cam_front_center",
    "frontright": "cam_front_right",
    "frontleft": "cam_front_left",
    "sideright": "cam_side_right",
    "sideleft": "cam_side_left",
    "rearcenter": "cam_rear_center",
}


def frame_timestamp(frame_id: int, anchor_frame: int, anchor_ts: int) -> int:
    return int(anchor_ts + (frame_id - anchor_frame) * (1_000_000 / 30))


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, dest)


def load_signal(bus_path: Path, name: str) -> tuple[str, np.ndarray]:
    """Pull one signal out of a bus JSON, pretty-printed or compact."""
    decoder = json.JSONDecoder()
    unit = ""
    pairs: list[tuple[int, float]] = []
    capturing = False
    buf = ""
    key = f'"{name}"'
    with bus_path.open() as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk and not buf:
                break
            buf += chunk
            if not capturing:
                at = buf.find(key)
                if at < 0:
                    buf = buf[-len(key) :]
                    if not chunk:
                        break
                    continue
                buf = buf[at + len(key) :]
                capturing = True
            start = buf.find("{")
            if start < 0:
                if not chunk:
                    break
                continue
            try:
                obj, end = decoder.raw_decode(buf[start:])
            except json.JSONDecodeError:
                if not chunk:
                    raise
                continue
            unit = str(obj.get("unit", ""))
            for timestamp, value in obj["values"]:
                pairs.append((int(timestamp), float(value)))
            break
    if not pairs:
        raise SystemExit(f"bus signal {name} not found in {bus_path}")
    return unit, np.asarray(pairs, dtype=np.float64)


def interp(signal: np.ndarray, timestamp: int) -> float:
    times = signal[:, 0]
    index = int(np.searchsorted(times, timestamp))
    index = min(max(index, 1), len(times) - 1)
    t0, t1 = times[index - 1], times[index]
    w = 0.0 if t1 == t0 else (timestamp - t0) / (t1 - t0)
    return float(signal[index - 1, 1] * (1 - w) + signal[index, 1] * w)


def ensure_bus(cache: Path, scene: dict) -> Path:
    dest = cache / "bus.json"
    if not dest.exists():
        url = f"{S3}/{scene['scene']}/bus_signals_{scene['stamp']}.json"
        print(f"downloading bus signals ({url})", flush=True)
        download(url, dest)
    return dest


def anchor_timestamp(cache: Path, scene: dict, frame_id: int) -> int:
    dest = cache / f"anchor_{frame_id}.json"
    url = (
        f"{S3}/{scene['scene']}/camera/cam_front_center/"
        f"{scene['stamp']}_camera_frontcenter_{frame_id:09d}.json"
    )
    download(url, dest)
    return int(json.loads(dest.read_text())["cam_tstamp"])


def ensure_config(cache: Path) -> dict:
    dest = cache / "cams_lidars.json"
    if not dest.exists():
        download(f"{S3}/cams_lidars.json", dest)
    return json.loads(dest.read_text())


def fetch_images(frame_ids: list[int], cache: Path, scene: dict) -> None:
    jobs = []
    for frame_id in frame_ids:
        for _qwen, _key, tag, _view in CAM_SLOTS:
            name = f"{scene['stamp']}_camera_{tag}_{frame_id:09d}.png"
            url = f"{S3}/{scene['scene']}/camera/{CAM_DIR[tag]}/{name}"
            dest = cache / "raw" / f"{frame_id:09d}" / f"{tag}.png"
            jobs.append((url, dest))
    print(f"fetching {len(jobs)} images", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, url, dest) for url, dest in jobs]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 12 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENES), default="gaimersheim")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/a2d2-clip"))
    parser.add_argument("--ui-dir", type=Path, default=None, help="Write resized JPEGs + clip.json here")
    parser.add_argument("--pack-dir", type=Path, default=None, help="Write Qwen frame folders here")
    parser.add_argument("--display-width", type=int, default=960)
    args = parser.parse_args()
    scene = SCENES[args.scene]
    if args.start is None:
        args.start = scene["start"]

    frame_ids = [args.start + i * args.stride for i in range(args.count)]
    config = ensure_config(args.cache)
    bus = ensure_bus(args.cache, scene)
    anchor_ts = anchor_timestamp(args.cache, scene, frame_ids[0])
    print("reading vehicle_speed and yaw rate", flush=True)
    speed_unit, speed = load_signal(bus, "vehicle_speed")
    yaw_unit, yaw_rate = load_signal(bus, "angular_velocity_omega_z")
    assert "KiloMeter" in speed_unit, speed_unit
    assert "Degre" in yaw_unit, yaw_unit

    samples = []
    for frame_id in frame_ids:
        ts = frame_timestamp(frame_id, frame_ids[0], anchor_ts)
        kmh = interp(speed, ts)
        deg = interp(yaw_rate, ts)
        samples.append(
            {
                "frame_id": frame_id,
                "timestamp_us": ts,
                "t": (ts - anchor_ts) / 1e6,
                "speed_mps": kmh / 3.6,
                "yaw_rate": np.deg2rad(deg),
                "speed_kmh": kmh,
                "yaw_rate_deg": deg,
            }
        )
    poses = integrate_ego(samples)
    for sample, pose in zip(samples, poses):
        sample["pose"] = pose

    print(
        f"speed {samples[0]['speed_kmh']:.1f} -> {samples[-1]['speed_kmh']:.1f} km/h, "
        f"path {poses[-1]['x']:.1f} m",
        flush=True,
    )
    fetch_images(frame_ids, args.cache, scene)
    calib = pack_calibration(config["cameras"])

    if args.ui_dir:
        write_ui(args, samples, calib, config, scene)
    if args.pack_dir:
        write_pack(args, frame_ids, samples, calib, config)
    manifest = {
        "sequence": scene["scene"].split("/")[-1],
        "city": scene["city"],
        "start_frame": args.start,
        "stride": args.stride,
        "hz": 30 / args.stride,
        "frames": samples,
        "slots": calib["slots"],
        "license": "A2D2 CC BY-ND 4.0, Audi",
    }
    # Poses are not JSON-native until we drop numpy. samples already are.
    out = ROOT / "data" / "clip_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}", flush=True)


def write_ui(args, samples, calib, config, scene: dict) -> None:
    ui = args.ui_dir
    ui.mkdir(parents=True, exist_ok=True)
    frames = []
    for sample in samples:
        frame_id = sample["frame_id"]
        token = f"f{frame_id:09d}"
        frame_dir = ui / token
        frame_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        matrices = {}
        sizes = {}
        for slot_index, (qwen, _key, tag, _view) in enumerate(CAM_SLOTS):
            raw = cv2.imread(str(args.cache / "raw" / f"{frame_id:09d}" / f"{tag}.png"))
            if raw is None:
                raise SystemExit(f"missing {tag} frame {frame_id}")
            camera = config["cameras"][CAM_SLOTS[slot_index][1]]
            clean = undistort(raw, camera)
            scale = args.display_width / clean.shape[1]
            display = cv2.resize(clean, (args.display_width, int(round(clean.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            dest = frame_dir / f"{qwen}.jpg"
            cv2.imwrite(str(dest), display, [int(cv2.IMWRITE_JPEG_QUALITY), 74])
            intrinsic = np.asarray(camera["CamMatrix"], dtype=np.float64)
            rotation = calib["sensor2lidar_rotation"][slot_index].astype(np.float64)
            translation = calib["sensor2lidar_translation"][slot_index].astype(np.float64)
            matrix = scale_lidar2img(build_lidar2img(intrinsic, rotation, translation), scale, scale)
            images[qwen] = f"/clip/{token}/{qwen}.jpg"
            matrices[qwen] = np.round(matrix, 6).tolist()
            sizes[qwen] = [display.shape[1], display.shape[0]]
        frames.append(
            {
                "token": token,
                "frame_id": frame_id,
                "t": round(sample["t"], 4),
                "speed_mps": round(sample["speed_mps"], 3),
                "speed_kmh": round(sample["speed_kmh"], 2),
                "yaw_rate_deg": round(sample["yaw_rate_deg"], 3),
                "pose": {k: round(sample["pose"][k], 4) for k in ("x", "y", "yaw", "t")},
                "images": images,
                "lidar2img": matrices,
                "size": sizes,
            }
        )
    clip = {
        "title": scene["title"],
        "source": f"A2D2 sensor fusion, sequence {scene['scene'].split('/')[-1]}",
        "city": scene["city"],
        "license": "CC BY-ND 4.0, Audi Autonomous Driving Dataset",
        "model": "Qwen-Drive-1.0-4B perception",
        "dataset_type_slot": "nuscenes",
        "why_not_nuscenes": scene["blurb"],
        "hz": 30 / args.stride,
        "slots": calib["slots"],
        "classes": [
            "vehicle",
            "czone_sign",
            "bicycle",
            "generic_object",
            "pedestrian",
            "traffic_cone",
            "barrier",
        ],
        "map_classes": [
            "background",
            "driveable_surface",
            "road_line",
            "road_edge",
            "crosswalk",
            "walkway",
        ],
        "map_bounds": {"x": [-30, 30], "y": [-15, 15]},
        "frames": frames,
    }
    (ui / "clip.json").write_text(json.dumps(clip))
    print(f"wrote UI clip {ui / 'clip.json'} ({len(frames)} frames)", flush=True)


def write_pack(args, frame_ids, samples, calib, config) -> None:
    from qdtrack.geometry import frame_json

    pack = args.pack_dir
    for sample in samples:
        frame_id = sample["frame_id"]
        token = f"f{frame_id:09d}"
        folder = pack / token
        image_dir = folder / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for slot_index, (qwen, a2d2_key, tag, _view) in enumerate(CAM_SLOTS):
            raw = cv2.imread(str(args.cache / "raw" / f"{frame_id:09d}" / f"{tag}.png"))
            clean = undistort(raw, config["cameras"][a2d2_key])
            cv2.imwrite(str(image_dir / f"{qwen}.jpg"), clean, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        np.savez(
            folder / "calib.npz",
            cam_intrinsic=calib["cam_intrinsic"],
            sensor2lidar_rotation=calib["sensor2lidar_rotation"],
            sensor2lidar_translation=calib["sensor2lidar_translation"],
            lidar2ego=calib["lidar2ego"],
        )
        np.savez(folder / "gt.npz", boxes=np.zeros((0, 9), np.float32))
        meta = frame_json(token)
        meta["a2d2_frame"] = frame_id
        meta["pose"] = sample["pose"]
        meta["speed_mps"] = sample["speed_mps"]
        (folder / "frame.json").write_text(json.dumps(meta, indent=2))
    print(f"packed {len(samples)} frames into {pack}", flush=True)


if __name__ == "__main__":
    main()
