"""Optional RF-DETR Large object boxes for scripts/overlay.py (``--rfdetr``).

YOLOPv2 keeps the lane lines. RF-DETR Large (the ``rfdetr`` package,
Apache-2.0, COCO weights, 704 px, fp32) supplies the objects, each with its
own class: car, truck and bus, and every vulnerable road user COCO has
(person, bicycle, motorcycle and the animals). A person standing on a bicycle
or a motorcycle box is one rider: ``cyclist`` or ``motorcyclist``.

Nothing here runs unless ``--rfdetr`` is passed. The YOLOPv2 cache is not
touched; the RF-DETR boxes go to their own cache next to it.
"""

from pathlib import Path

import cv2
import numpy as np

# COCO name -> pipeline class. Anything else is dropped.
COCO_TO_KIND = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "dog": "dog",
    "cat": "cat",
    "horse": "horse",
    "cow": "cow",
    "sheep": "sheep",
    "bird": "bird",
    "elephant": "elephant",
    "bear": "bear",
    "zebra": "zebra",
    "giraffe": "giraffe",
}
ANIMALS = ("dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "zebra", "giraffe")
VEHICLES = ("car", "truck", "bus")
VRU = ("person", "cyclist", "bicycle", "motorcyclist", "motorcycle") + ANIMALS
KINDS = VEHICLES + VRU

# Tracker class (an existing qdtrack DET_CLASS_NAMES entry) for each kind.
# Car, truck and bus share the vehicle motion model, and every two-wheeler the
# bicycle gate; the kind is voted per track afterwards, so a car read as a
# truck in one frame does not break the track. Animals use generic_object.
TRACKER_NAME = {kind: "vehicle" for kind in VEHICLES + ("vehicle",)}
TRACKER_NAME.update({"person": "pedestrian", "cyclist": "bicycle", "bicycle": "bicycle",
                     "motorcyclist": "bicycle", "motorcycle": "bicycle"})
TRACKER_NAME.update({kind: "generic_object" for kind in ANIMALS})

# Ground footprint (width, length, height) in metres. The tracker moves the
# visible face to the centre by half the length.
SIZE = {"vehicle": (1.8, 4.2, 1.5), "car": (1.8, 4.2, 1.5), "truck": (2.5, 8.0, 3.2), "bus": (2.6, 11.0, 3.2),
        "person": (0.6, 0.6, 1.7), "cyclist": (0.6, 1.8, 1.7), "bicycle": (0.6, 1.8, 1.1),
        "motorcyclist": (0.8, 2.1, 1.6), "motorcycle": (0.8, 2.1, 1.2)}
for _kind in ANIMALS:
    SIZE[_kind] = (0.6, 1.0, 0.8)
for _kind in ("horse", "cow", "elephant", "bear", "zebra", "giraffe"):
    SIZE[_kind] = (0.8, 2.2, 1.6)

# BGR. Lane lines stay green and road edges orange, so no class uses those.
COLOUR = {
    "vehicle": (255, 150, 50),  # a YOLOPv2 box in --rfdetr-mode vru (no class)
    "car": (255, 150, 50),
    "truck": (255, 255, 0),
    "bus": (200, 60, 200),
    "person": (60, 60, 255),
    "cyclist": (0, 255, 255),
    "bicycle": (0, 170, 170),
    "motorcyclist": (255, 0, 255),
    "motorcycle": (150, 0, 150),
    "dog": (180, 255, 180),
    "cat": (200, 200, 255),
    "horse": (40, 100, 160),
    "cow": (255, 255, 255),
    "sheep": (210, 230, 240),
    "bird": (150, 150, 150),
    "elephant": (110, 110, 110),
    "bear": (20, 60, 110),
    "zebra": (230, 230, 230),
    "giraffe": (60, 180, 230),
}

STORE_MIN = 0.2  # scores kept in the cache, so the threshold can change without a rerun


def load_model(resolution=None, device="cpu"):
    from rfdetr import RFDETRLarge

    kwargs = {"device": device}
    if resolution:
        kwargs["resolution"] = int(resolution)
    try:
        return RFDETRLarge(**kwargs)
    except TypeError:
        return RFDETRLarge()


def detect(model, image_bgr, store_min=STORE_MIN):
    """One frame, fp32. Returns rows [x1, y1, x2, y2, score] and COCO names."""
    out = model.predict(np.ascontiguousarray(image_bgr[:, :, ::-1]), threshold=store_min)
    boxes = np.asarray(getattr(out, "xyxy", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(getattr(out, "confidence", np.zeros(len(boxes))), dtype=np.float32).reshape(-1)
    data = getattr(out, "data", None) or {}
    if "class_name" in data:
        names = [str(name) for name in list(data["class_name"])]
    else:
        from rfdetr.assets.coco_classes import COCO_CLASSES

        names = [COCO_CLASSES.get(int(c), str(int(c))) for c in getattr(out, "class_id", [])]
    return np.concatenate([boxes, scores[:, None]], 1), names


def run_clip(model, frames, cache):
    """frames: iterable of BGR images. Writes the RF-DETR cache (COCO classes, raw)."""
    rows, names = [], []
    for index, image in enumerate(frames):
        row, name = detect(model, image)
        rows.append(row.astype(np.float32))
        names.append(np.asarray(name, dtype=object))
        if index % 15 == 0:
            print("rfdetr", index, len(row), flush=True)
    np.savez_compressed(cache, rows=np.array(rows, dtype=object), names=np.array(names, dtype=object),
                        model=np.array("RFDETRLarge fp32 COCO"))
    return rows, names


def load_cache(cache):
    with np.load(cache, allow_pickle=True) as data:
        return list(data["rows"]), [list(n) for n in data["names"]]


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _rides(person, ride):
    """A person on a bicycle or motorcycle: the bike sits in the lower part of
    the person box, and they overlap sideways."""
    px1, py1, px2, py2 = person[:4]
    bx1, by1, bx2, by2 = ride[:4]
    overlap = max(0.0, min(px2, bx2) - max(px1, bx1))
    if overlap < 0.5 * min(px2 - px1, bx2 - bx1):
        return False
    height = max(py2 - py1, 1.0)
    return by1 < py2 and by2 > py1 + 0.4 * height and abs(by2 - py2) < 0.6 * height


def frame_objects(rows, names, threshold=0.5):
    """COCO boxes of one frame -> [(x1, y1, x2, y2, score, kind)], riders paired.

    Same-kind duplicates (IoU > 0.7) are dropped. A person on a bicycle is one
    cyclist box (the union), a person on a motorcycle one motorcyclist box.
    """
    items = []
    for row, name in zip(rows, names):
        kind = COCO_TO_KIND.get(str(name))
        if kind is None or float(row[4]) < threshold:
            continue
        items.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), kind])
    items.sort(key=lambda item: -item[4])
    kept = []
    for item in items:
        if any(other[5] == item[5] and _iou(item, other) > 0.7 for other in kept):
            continue
        kept.append(item)
    people = [item for item in kept if item[5] == "person"]
    rides = [item for item in kept if item[5] in ("bicycle", "motorcycle")]
    used_people, used_rides, out = set(), set(), []
    pairs = []
    for i, person in enumerate(people):
        for j, ride in enumerate(rides):
            if _rides(person, ride):
                pairs.append((-_iou(person, ride) - 0.01 * (person[4] + ride[4]), i, j))
    for _cost, i, j in sorted(pairs):
        if i in used_people or j in used_rides:
            continue
        used_people.add(i)
        used_rides.add(j)
        person, ride = people[i], rides[j]
        box = [min(person[0], ride[0]), min(person[1], ride[1]), max(person[2], ride[2]), max(person[3], ride[3])]
        kind = "cyclist" if ride[5] == "bicycle" else "motorcyclist"
        out.append(box + [max(person[4], ride[4]), kind])
    for i, person in enumerate(people):
        if i not in used_people:
            out.append(person)
    for j, ride in enumerate(rides):
        if j not in used_rides:
            out.append(ride)
    out.extend(item for item in kept if item[5] not in ("person", "bicycle", "motorcycle"))
    return out


def vote_kind(kinds):
    """Most frequent kind on a track, score-weighted."""
    totals = {}
    for kind, score in kinds:
        totals[kind] = totals.get(kind, 0.0) + score
    return max(totals, key=totals.get) if totals else None


def legend(panel, kinds, origin=(8, 20)):
    """Class colour key in the bird's-eye panel."""
    x, y = origin
    for kind in kinds:
        cv2.circle(panel, (x + 5, y - 4), 5, COLOUR[kind], -1)
        cv2.putText(panel, kind, (x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        y += 16
    return panel


def default_cache(cache):
    cache = Path(cache)
    return cache.with_name(cache.stem + "_rfdetr.npz")


# --rfdetr-mode label: YOLOPv2 keeps every vehicle box and track exactly as
# without the flag. A YOLOPv2 box only borrows the class of the RF-DETR box it
# overlaps most. These are the kinds a YOLOPv2 vehicle box may take.
LABEL_KINDS = VEHICLES + ("motorcyclist", "motorcycle", "cyclist", "bicycle")
LABEL_IOU = 0.3


def match_kinds(yolo_rows, objects, iou_min=LABEL_IOU):
    """One-to-one match of YOLOPv2 boxes to RF-DETR objects, highest IoU first.

    yolo_rows: [x1, y1, x2, y2, ...] image boxes. objects: frame_objects() output.
    Returns (kinds, used): kinds[i] is (kind, score) for yolo row i or None, and
    used is the set of object indices that labelled a YOLOPv2 box.
    """
    pairs = []
    for i, row in enumerate(yolo_rows):
        for j, item in enumerate(objects):
            if item[5] not in LABEL_KINDS:
                continue
            overlap = _iou(row, item)
            if overlap >= iou_min:
                pairs.append((-overlap, i, j))
    kinds = [None] * len(yolo_rows)
    used = set()
    for _cost, i, j in sorted(pairs):
        if kinds[i] is not None or j in used:
            continue
        kinds[i] = (objects[j][5], float(objects[j][4]))
        used.add(j)
    return kinds, used


def extra_objects(yolo_rows, objects, iou_min=LABEL_IOU):
    """RF-DETR objects that are not already a YOLOPv2 vehicle box.

    VRUs and animals that YOLOPv2 did not see, plus RF-DETR-only cars, trucks
    and buses (those become flagged rf_only tracks, never vehicle-tag input).
    """
    _kinds, used = match_kinds(yolo_rows, objects, iou_min)
    out = []
    for j, item in enumerate(objects):
        if j in used:
            continue
        if item[5] in LABEL_KINDS and any(_iou(row, item) >= iou_min for row in yolo_rows):
            continue  # a second RF box on an already labelled YOLOPv2 vehicle
        out.append(item)
    return out
