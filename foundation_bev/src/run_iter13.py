# Foundation BEV iter3 (draft spec, 23 Nov 2026): iter2 kernel + bev_gate.py (detection gates, track filters,
# static dots) + ego_motion.py (visual ego speed/yaw, curved ego path) + per-frame dets/depth/ego dump
# + ego-speed source comparison + optional YOLOPv2 baseline (measurement only).
# Previous header (iter2, based on GPU Bot's iter1 kernel):
# Changes: bash -o pipefail everywhere; WildDet3D submodules (lingbot_depth + sam3 code) init --recursive and
# pin-checked; import pre-flight + 3-frame WD3D pre-flight; stage 1 must print STAGE1 OK with >0 object boxes
# or the run stops (no silent fallback); stage 2 = iter2 foundation_bev.py (+ bev_smooth.py) with --require-dets;
# outputs *_foundation_bev_iter2.mp4 (never overwrites iter1 names); smoothness_metrics.py report at the end.
# Free Kaggle T4 only, $0. No TwinLiteNet, no YOLO, no Grounding DINO, no SAM 3.
# Stage 1: WildDet3D text prompts car/person/bicycle + lane/edge boxes.
# Stage 2: SAM 2.1 (box prompts, else lower-half fallback) + DAV2-S lift + BEV render.
# Outputs: *_foundation_bev.mp4, stills/, tracks/, polylines/, run_meta.json, README.md.
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

T0 = time.time()
OUT = os.environ.get("FBEV_OUT", "/kaggle/working")
TMP = os.environ.get("FBEV_TMP", "/kaggle/temp/w")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
os.chdir(OUT)
meta = {"rc": 1, "spend_usd": 0, "videos": {}, "stage": "start"}
PYBIN = sys.executable

WD3D_SHA = "1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a"
LINGBOT = "ff29de5bf433d1b1519cd851da00bc652668964a"
SAM3_SUB = "159490bb13108749dd5ceff68f42cd1792e60641"  # pin used by the working drive-videos-22nov WD3D kernel
HF_REV = "161edb26239f58d8973045517d9c8b179c515f73"
CKPT = "wilddet3d_alldata_all_prompt_v1.0.pt"
CKPT_SHA = "f8b6a9e548f733ba62625a0d2adc4b0f4fdb6007ee11d9927f9c1027010fee57"
SAM2_PIP = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DAV2_SHA = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
SAM_CKPT = "sam2.1_hiera_small.pt"
DAV2_CKPT = "depth_anything_v2_vits.pth"
PROMPTS = ["car", "person", "bicycle", "lane line", "lane marking", "road edge", "road curb"]
TAGS = [
    "kitti_highway_sample", "kitti_highway_exit_approach", "kitti_highway_late",
    "kitti_highway_final", "kitti_city_cyclist_candidate", "kitti_lane_split_candidate",
    "munich", "a2d2a", "a2d2b",
    "fc1_merge", "fc1_split",  # iter13: private 4 fps FC1 clips; run only when DS/<tag>.mp4 + cams_private.json exist (never in git)
]
PRIVATE = ("fc1_merge", "fc1_split")
BUDGET_S1 = float(os.environ.get("FBEV_BUDGET_S1", str(2.6 * 3600)))
WALL_CAP = float(os.environ.get("FBEV_WALL_CAP", str(3.8 * 3600)))
MAXF = int(os.environ.get("FBEV_MAXF", "0"))
SMOKE = os.environ.get("FBEV_SMOKE") == "1"
# iter8: stage-2-only rerun. FBEV_REUSE_DETS=<dir> holding stage-1 output wd3d_<tag>.npz (preferred) or
# <tag>_dets_ego.json (dets_dump); WildDet3D install/pre-flight/stage 1 are skipped. FBEV_TAGS = comma list of clips.
REUSE = os.environ.get("FBEV_REUSE_DETS", "")
if os.environ.get("FBEV_TAGS"):
    TAGS = [t for t in TAGS if t in os.environ["FBEV_TAGS"].split(",")]


def sh(c, **k):
    # iter2: pipefail, so `cmd | tee log` returns cmd's exit code (iter1 recorded tee's 0)
    print("+", c, flush=True)
    return subprocess.run(["bash", "-o", "pipefail", "-c", c], **k).returncode


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(**kw):
    meta.update(kw)
    meta["elapsed_s"] = round(time.time() - T0, 1)
    json.dump(meta, open(f"{OUT}/run_meta.json", "w"), indent=1)


def find_ds():
    hits = glob.glob("/kaggle/input/**/cams.json", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    # local smoke
    for c in (os.environ.get("FBEV_DS", ""), "/workspace/gpubot/foundation-bev-23nov/ds"):
        if c and os.path.exists(os.path.join(c, "cams.json")):
            return c
    return ""


DS = find_ds()
if not DS:
    save(rc=20, stage="no_dataset")
    sys.exit(0)
meta["dataset"] = DS

# copy scripts (Kaggle may have extracted src/)
SRC = os.path.join(DS, "src")
for fn in ("wd3d_infer.py", "foundation_bev.py", "bev_smooth.py", "bev_gate.py", "ego_motion.py", "lines_mt.py", "bev_ipm.py", "smoothness_metrics.py", "analyze_polylines.py"):
    src = os.path.join(SRC, fn)
    if not os.path.exists(src):
        src = os.path.join(DS, fn)
    if not os.path.exists(src):
        save(rc=20, stage=f"missing_{fn}")
        sys.exit(0)
    shutil.copy2(src, os.path.join(OUT, fn))
    meta.setdefault("script_sha256", {})[fn] = sha256(os.path.join(OUT, fn))

CAMS_JSON = next((os.path.join(DS, c) for c in ("cams_iter13.json", "cams_iter12.json", "cams_iter11.json", "cams_iter10.json", "cams_iter9.json", "cams_iter8.json", "cams_iter7.json", "cams_iter6.json", "cams_iter5b.json", "cams_iter5.json", "cams_iter3.json", "cams_iter2.json", "cams.json") if os.path.exists(os.path.join(DS, c))))
cams = json.load(open(CAMS_JSON))
meta["cams_json"] = os.path.basename(CAMS_JSON)
motion = json.load(open(os.path.join(DS, "motion_assumptions.json")))
# iter13: private clips (fc1_merge) ship their camera / motion entries next to the local video only
for _fn, _tab in (("cams_private.json", cams), ("motion_private.json", motion)):
    if os.path.exists(os.path.join(DS, _fn)):
        _tab.update(json.load(open(os.path.join(DS, _fn))))
for _t in PRIVATE:
    if _t in TAGS and not (os.path.exists(os.path.join(DS, f"{_t}.mp4")) and _t in cams):
        TAGS = [t for t in TAGS if t != _t]
        meta[_t] = "skipped (no private clip in FBEV_DS)"
if any(t in TAGS for t in PRIVATE):  # foundation_bev.py / the YOLO baseline read --cam-json: give them the merged table
    os.makedirs(OUT, exist_ok=True)
    CAMS_JSON = os.path.join(OUT, "cams_merged_private.json")
    json.dump(cams, open(CAMS_JSON, "w"), indent=1)
meta["assumptions"] = {t: {"cam": cams[t].get("note", ""), "motion": motion[t].get("note", ""),
                           "speed_mps": motion[t]["speed_mps"], "K": [cams[t]["fx"], cams[t]["fy"], cams[t]["cx"], cams[t]["cy"]],
                           "cam_height": cams[t]["height"]} for t in TAGS if t in cams}

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip()
meta["gpu"] = gpu or ("cpu-smoke" if SMOKE else "unknown")
save(stage="dataset_ok")

WROOT = f"{TMP}/wilddet3d"
WEIGHTS = f"{TMP}/weights"
os.makedirs(WEIGHTS, exist_ok=True)
ck = ""
stage1_s = 0
if not SMOKE and not REUSE:
    # --- install WildDet3D (same recipe as drive-videos-22nov-wilddet3d, rc=0) ---
    r = sh(
        f'{PYBIN} -m pip install -q --upgrade "setuptools<80" && '
        f'{PYBIN} -m pip install -q --no-deps einops timm "ftfy==6.1.1" iopath portalocker regex huggingface_hub pyquaternion scipy ml_collections && '
        f'{PYBIN} -m pip install -q --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git" && '
        f'({PYBIN} -c "import vis4d" 2>/dev/null || {PYBIN} -m pip install -q vis4d==1.0.0) && '
        f'({PYBIN} -c "import vis4d_cuda_ops" 2>/dev/null || {PYBIN} -m pip install -q "git+https://github.com/SysCV/vis4d_cuda_ops.git" --no-build-isolation --no-cache-dir) 2>&1 | tail -8 | tee log_install1.txt'
    )
    if r:
        save(rc=21, stage="install_stage1")
        sys.exit(0)
    r = sh(
        f"rm -rf {WROOT} && git clone -q https://github.com/allenai/WildDet3D.git {WROOT} && "
        f"git -C {WROOT} checkout -q {WD3D_SHA} && "
        f"git -C {WROOT} submodule update --init --recursive --quiet && "
        f"git -C {WROOT} rev-parse HEAD | tee commits.txt && "
        f"git -C {WROOT} submodule status --recursive | tee -a commits.txt"
    )
    if r:
        save(rc=21, stage="clone_wd3d")
        sys.exit(0)
    st = open("commits.txt").read()
    if LINGBOT not in st or SAM3_SUB not in st:
        save(rc=21, stage="submodule_pin_mismatch", submodules=st[-600:])
        sys.exit(0)
    # sam3 is initialised as CODE only: WildDet3D's model.py imports it. No SAM 3 weights are downloaded,
    # and SAM 3 is not run as a segmenter (WD3D's own checkpoint carries its weights).
    PRE_PATH = f"{WROOT}/third_party/sam3:{WROOT}/third_party/lingbot_depth:{WROOT}"
    r = sh(
        f"PYTHONPATH={PRE_PATH}:$PYTHONPATH {PYBIN} -c \"import sam3, wilddet3d; from wilddet3d import build_model, preprocess; "
        f"print('PREFLIGHT IMPORT OK')\" 2>&1 | tee log_preflight_import.txt"
    )
    if r or "PREFLIGHT IMPORT OK" not in open("log_preflight_import.txt").read():
        save(rc=26, stage="preflight_import")
        sys.exit(0)
    from huggingface_hub import hf_hub_download  # noqa: E402

    ck = hf_hub_download("allenai/WildDet3D", CKPT, revision=HF_REV, local_dir=f"{TMP}/ckpt")
    got = sha256(ck)
    open("commits.txt", "a").write(f"\n{CKPT}@{HF_REV} {got}\n")
    if got != CKPT_SHA:
        save(rc=22, stage="ckpt_sha", got=got)
        sys.exit(0)
    meta["wd3d_commit"] = WD3D_SHA
    meta["ckpt_sha256"] = got
    save(stage="stage1")
    tags = " ".join(TAGS)
    prompts = " ".join(f'"{p}"' for p in PROMPTS)
    mf = f" --max-frames {MAXF}" if MAXF else ""
    os.makedirs("dets", exist_ok=True)
    # pre-flight: 3 frames on one KITTI and one Munich clip must give >0 object boxes
    r0 = sh(
        f"{sys.executable} wd3d_infer.py --wd3d-root {WROOT} --ckpt {ck} --inputs {DS} "
        f"--tags kitti_highway_sample munich --outdir dets_preflight --device cuda:0 --fp16 1 --max-frames 3 "
        f"--prompts {prompts} 2>&1 | grep -v 'NMS DEBUG' | tee log_preflight_wd3d.txt"
    )
    if r0 or "STAGE1 OK" not in open("log_preflight_wd3d.txt").read():
        save(rc=27, stage="preflight_wd3d", preflight_rc=r0)
        sys.exit(0)
    t1 = time.time()
    r1 = sh(
        f"{sys.executable} wd3d_infer.py --wd3d-root {WROOT} --ckpt {ck} --inputs {DS} "
        f"--tags {tags} --outdir dets --device cuda:0 --fp16 1 --budget-s {BUDGET_S1}{mf} "
        f"--prompts {prompts} 2>&1 | grep -v 'NMS DEBUG' | tee log_stage1.txt"
    )
    stage1_s = round(time.time() - t1, 1)
    meta["stage1_rc"] = r1
    meta["stage1_s"] = stage1_s
    log1 = open("log_stage1.txt").read()
    if r1 or "STAGE1 OK" not in log1:
        save(rc=23, stage="stage1_failed")
        sys.exit(0)
    try:
        meta["stage1_summary"] = json.load(open("dets/wd3d_summary.json"))
    except Exception as e:  # noqa: BLE001
        meta["stage1_summary_err"] = repr(e)
if REUSE:  # iter8: link the kept stage-1 dets, stop if any requested clip is missing
    os.makedirs(f"{OUT}/dets", exist_ok=True)
    missing, used = [], {}
    for t in TAGS:
        for src_, dst in ((f"{REUSE}/wd3d_{t}.npz", f"{OUT}/dets/wd3d_{t}.npz"),
                          (f"{REUSE}/{t}_dets_ego.json", f"{OUT}/dets/{t}_dets_ego.json")):
            if os.path.exists(src_):
                if not os.path.exists(dst):
                    shutil.copy(src_, dst)
                used[t] = os.path.basename(dst); break
        else:
            missing.append(t)
    meta["reuse_dets"] = {"dir": REUSE, "used": used, "missing": missing}
    print("REUSE DETS", json.dumps(meta["reuse_dets"]), flush=True)
    if missing:
        save(rc=28, stage="reuse_dets_missing")
        sys.exit(0)
    if not SMOKE:
        sh(f"{PYBIN} -c 'import scipy' 2>/dev/null || {PYBIN} -m pip install -q scipy")
if not SMOKE:
    # free WD3D before SAM/DAV2
    sh(f"rm -rf {WROOT} {TMP}/ckpt")
    save(stage="install_stage2")
    r = sh(
        "pip install -q hydra-core iopath opencv-python-headless huggingface_hub scikit-image && "
        f"SAM2_BUILD_CUDA=0 pip install -q --no-deps 'git+https://github.com/facebookresearch/sam2.git@{SAM2_PIP}' && "
        f"rm -rf {TMP}/dav2 && git clone -q https://github.com/DepthAnything/Depth-Anything-V2.git {TMP}/dav2 && "
        f"git -C {TMP}/dav2 checkout -q {DAV2_SHA} && "
        f"{sys.executable} -c \"from huggingface_hub import hf_hub_download as d; d('depth-anything/Depth-Anything-V2-Small','{DAV2_CKPT}',local_dir='{WEIGHTS}'); d('facebook/sam2.1-hiera-small','{SAM_CKPT}',local_dir='{WEIGHTS}')\""
    )
    if r:
        save(rc=25, stage="install_stage2")
        sys.exit(0)
    meta["weights_sha256"] = {os.path.basename(p): sha256(p) for p in glob.glob(f"{WEIGHTS}/*")}
    meta["sam2_pip"] = SAM2_PIP
    meta["dav2_commit"] = DAV2_SHA
    save(stage="stage2")
else:
    save(stage="smoke_stage2")

os.makedirs(f"{OUT}/results", exist_ok=True)
os.makedirs(f"{OUT}/stills", exist_ok=True)
os.makedirs(f"{OUT}/tracks", exist_ok=True)
os.makedirs(f"{OUT}/polylines", exist_ok=True)
os.makedirs(f"{OUT}/dets_dump", exist_ok=True)

env = dict(os.environ)
if not SMOKE:
    env["PYTHONPATH"] = f"{TMP}/dav2:" + env.get("PYTHONPATH", "")

fails = []
t2 = time.time()
for tag in TAGS:
    if time.time() - T0 > WALL_CAP:
        meta["stopped_for_time"] = True
        meta["stopped_before"] = tag
        break
    v = os.path.join(DS, f"{tag}.mp4")
    if not os.path.exists(v):
        fails.append(tag)
        meta["videos"][tag] = {"rc": 26, "error": "missing video"}
        save()
        continue
    o = f"{OUT}/results/{tag}_foundation_bev_iter13.mp4"
    dets = f"{OUT}/dets/wd3d_{tag}.npz"
    if not os.path.exists(dets) and os.path.exists(f"{OUT}/dets/{tag}_dets_ego.json"):
        dets = f"{OUT}/dets/{tag}_dets_ego.json"  # iter8: reused dets_dump
    if SMOKE:
        dets = dets if os.path.exists(dets) else ""
    elif not os.path.exists(dets):
        fails.append(tag)
        meta["videos"][tag] = {"rc": 7, "error": "missing WD3D dets (iter3 does not fall back)"}
        save()
        continue
    spd = motion.get(tag, {}).get("speed_mps", 12.0)
    mf = f" --max-frames {MAXF}" if MAXF else ""
    smoke_flag = " --smoke" if SMOKE else ""
    device = "cpu" if SMOKE else "cuda:0"
    cmd = (
        f"{sys.executable} {OUT}/foundation_bev.py --video {v} --output {o} --cam-json {CAMS_JSON} --require-dets {0 if SMOKE else 1} "
        f"--tag {tag} --speed {spd} --device {device} --weights {WEIGHTS} "
        f"--stills {OUT}/stills --tracks-csv {OUT}/tracks/{tag}_tracks.csv "
        f"--polylines-json {OUT}/polylines/{tag}_polylines.json "
        f"--clip-meta {OUT}/results/{tag}.json --dets-json {OUT}/dets_dump/{tag}_dets_ego.json "
        + (f"--dets {dets} " if dets and os.path.exists(dets) else "")
        + mf + smoke_flag
    )
    t = time.time()
    with open(f"{OUT}/log_{tag}.txt", "w") as lf:
        try:
            rc = subprocess.run(cmd, shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=3600).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    rec = {"rc": rc, "sec": round(time.time() - t, 1)}
    jp = f"{OUT}/results/{tag}.json"
    if os.path.exists(jp):
        rec.update(json.load(open(jp)))
        rec["rc"] = rc if rc else rec.get("rc", 0)
    if os.path.exists(o):
        rec["sha256"] = sha256(o)
        rec["bytes"] = os.path.getsize(o)
    else:
        rec.setdefault("rc", rc or 5)
        fails.append(tag)
    if rec.get("rc"):
        if tag not in fails:
            fails.append(tag)
    meta["videos"][tag] = rec
    save()

meta["stage2_s"] = round(time.time() - t2, 1)

# iter3: ego-speed source comparison per clip (no KITTI oxts available: fallback oxts > flow if consistent > scene > prior)
print("EGO SPEED COMPARISON (m/s): clip | prior | flow median (valid) | scene closing-rate median (valid) | BEV phase (valid, repeats) | oxts | selected source -> speed", flush=True)
ego_tab = []
for tag in TAGS:
    e = (meta["videos"].get(tag) or {}).get("ego_speed") or {}
    if e:
        row = (f"{tag} | {e.get('prior_mps')} | {e.get('flow_median_mps', float('nan')):.1f} ({e.get('flow_valid_frac', 0):.2f}) | "
               f"{e.get('scene_median_mps', float('nan')):.1f} ({e.get('scene_valid_frac', 0):.2f}) | "
               f"{e.get('bev_phase_mps', float('nan')):.1f} ({e.get('bev_phase_valid', 0):.2f}, {e.get('n_repeat_frames', 0)}) | {e.get('oxts_mps')} | "
               f"{e.get('source')} -> {e.get('speed_median_mps', float('nan')):.1f}")
        print("EGO", row, flush=True)
        ego_tab.append(row)
meta["ego_speed_table"] = ego_tab

# iter3: optional YOLOPv2 baseline on the KITTI clips (measurement only; never part of the foundation pipeline).
# Enabled when FBEV_YOLOPV2 (TorchScript weights) and FBEV_YOLO_REPO (sam-bev-22nov repo dir or .tar.gz) exist.
YW, YR = os.environ.get("FBEV_YOLOPV2", ""), os.environ.get("FBEV_YOLO_REPO", "")
if YW and YR and os.path.exists(YW) and os.path.exists(YR) and os.path.exists(os.path.join(DS, "yolo_baseline", "run_yolo_baseline.sh")):
    ty = time.time()
    if any(t in TAGS for t in PRIVATE):  # iter13: YOLOPv2 baseline also on the private 4 fps clips (GT comparison)
        os.environ["CLIPS"] = " ".join(["kitti_highway_sample", "kitti_highway_exit_approach", "kitti_highway_late",
                                        "kitti_highway_final", "kitti_city_cyclist_candidate", "kitti_lane_split_candidate"]
                                       + [t for t in PRIVATE if t in TAGS])
    ry = sh(f"bash {DS}/yolo_baseline/run_yolo_baseline.sh {YR} {YW} {DS} {OUT}/yolo_baseline {'cpu' if SMOKE else 'cuda'} "
            f"{CAMS_JSON} {OUT} 2>&1 | tee {OUT}/log_yolo_baseline.txt")
    meta["yolo_baseline"] = {"rc": ry, "sec": round(time.time() - ty, 1)}
else:
    meta["yolo_baseline"] = "skipped (set FBEV_YOLOPV2 + FBEV_YOLO_REPO)"
meta["stage1_s"] = stage1_s
meta["fails"] = fails or "none"
shas = [r.get("sha256") for r in meta["videos"].values()]
ok_n = sum(1 for t in TAGS if t in meta["videos"] and meta["videos"][t].get("rc") == 0 and meta["videos"][t].get("sha256"))
all_done = ok_n == len(TAGS) and not fails and None not in shas and len(set(shas)) == len([s for s in shas if s])
meta["rc"] = 0 if all_done else (3 if fails else 4)

# README
lines = []
lines.append("# Foundation BEV run (WildDet3D + SAM 2.1 + Depth Anything V2 Small)")
lines.append("")
lines.append("Free Kaggle T4 only, $0. No TwinLiteNet, no YOLO, no Grounding DINO, no SAM 3.")
lines.append("")
lines.append("## Script / model versions")
lines.append("")
for k, v in (meta.get("script_sha256") or {}).items():
    lines.append(f"- `{k}` sha256 `{v}`")
lines.append(f"- WildDet3D commit `{WD3D_SHA}` (lingbot_depth `{LINGBOT}`, sam3 code submodule `{SAM3_SUB}`: imported by WD3D only)")
lines.append(f"- WildDet3D checkpoint `{CKPT}` @ `{HF_REV}` sha256 `{CKPT_SHA}`")
lines.append(f"- sam2 pip git `{SAM2_PIP}`; weights `facebook/sam2.1-hiera-small` / `{SAM_CKPT}`")
lines.append(f"- Depth Anything V2 Small code `{DAV2_SHA}`; weights `{DAV2_CKPT}`")
lines.append(f"- GPU: {meta.get('gpu')}")
lines.append(f"- wall_s: {meta.get('elapsed_s')}; stage1_s: {stage1_s}; stage2_s: {meta.get('stage2_s')}; rc: {meta['rc']}")
lines.append("")
lines.append("## iter3 per clip")
lines.append("")
lines.append("Lanes: WD3D lane boxes -> SAM masks (skeleton) + top-hat thin stripes in the SAM road -> BEV ->")
lines.append("lateral clusters -> RANSAC x(z) poly2 -> EMA 0.3, 0.5 m gate, hold 15, centred 7-frame window.")
lines.append("Edges: SAM drivable-area lower envelope around the ego corridor, object boxes masked, rails rejected.")
lines.append("Tracks: two-stage IoU+BEV association, CV Kalman + RTS, re-link <= 1.5 m, min length 5.")
lines.append("")
for t in TAGS:
    rec = meta.get("videos", {}).get(t, {})
    g = rec.get("ground", {}) or {}
    lines.append(f"- **{t}**: rc={rec.get('rc')}, frames {rec.get('frames_out', '?')}, obj boxes {rec.get('n_obj_dets_npz')}, "
                 f"tracks {rec.get('n_tracks')}, relinks {rec.get('n_relink')}, pitch {g.get('pitch_deg')} deg ({g.get('pitch_mode')}), "
                 f"lane width check {rec.get('lane_width_check')}")
lines.append("")
lines.append("## Assumptions")
lines.append("")
for t in TAGS:
    a = meta.get("assumptions", {}).get(t, {})
    lines.append(f"### {t}")
    lines.append(f"- Intrinsics fx,fy,cx,cy = {a.get('K')}, height {a.get('cam_height')} m")
    lines.append(f"- Ego motion: {a.get('motion')}")
    lines.append(f"- Camera: {a.get('cam')}")
    lines.append("")
lines.append("## Outputs")
lines.append("")
lines.append("- `results/<clip>_foundation_bev_iter13.mp4` — side-by-side front + BEV, H.264 when ffmpeg is available")
lines.append("- `stills/<clip>_fNNNN.png` — 2–3 mid / busy / early frames")
lines.append("- `tracks/<clip>_tracks.csv` — frame, track_id, class, score, bev_x, bev_z, x, y, z")
lines.append("- `polylines/<clip>_polylines.json` — per-frame lane/edge poly2 coefficients + sampled BEV/pixel curves")
lines.append("- `dets_dump/<clip>_dets_ego.json` — per-frame WD3D boxes/scores/3D, DAV2 scale, box depth, billboard flag, ego-speed candidates (offline replay)")
lines.append("- `metrics.md` / `metrics.json` — smoothness_metrics.py report (accel RMS, ID-switch proxy, line jitter)")
sh(f"cd {OUT} && {sys.executable} smoothness_metrics.py --run {OUT} --label iter13 --out {OUT}/metrics.md --json {OUT}/metrics.json > log_metrics.txt 2>&1")
if os.path.exists(f"{OUT}/metrics.md"):
    lines += ["", open(f"{OUT}/metrics.md").read()]
open(f"{OUT}/README.md", "w").write("\n".join(lines) + "\n")

sh(f"find {OUT}/results {OUT}/stills {OUT}/tracks {OUT}/polylines {OUT}/dets_dump {OUT}/run_meta.json {OUT}/README.md {OUT}/metrics.md -type f 2>/dev/null | sort | xargs -r sha256sum > {OUT}/SHA256SUMS.txt")
save(stage="end")
print("RC", meta["rc"], "fails", fails or "none", flush=True)
