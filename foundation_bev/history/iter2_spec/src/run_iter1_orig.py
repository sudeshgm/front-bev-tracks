# GPU Bot, 23 Nov 2026: foundation BEV (WildDet3D + SAM 2.1 hiera-small + Depth Anything V2 Small).
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
]
BUDGET_S1 = float(os.environ.get("FBEV_BUDGET_S1", str(2.6 * 3600)))
WALL_CAP = float(os.environ.get("FBEV_WALL_CAP", str(3.8 * 3600)))
MAXF = int(os.environ.get("FBEV_MAXF", "0"))
SMOKE = os.environ.get("FBEV_SMOKE") == "1"


def sh(c, **k):
    print("+", c, flush=True)
    return subprocess.run(c, shell=True, **k).returncode


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
for fn in ("wd3d_infer.py", "foundation_bev.py"):
    src = os.path.join(SRC, fn)
    if not os.path.exists(src):
        src = os.path.join(DS, fn)
    if not os.path.exists(src):
        save(rc=20, stage=f"missing_{fn}")
        sys.exit(0)
    shutil.copy2(src, os.path.join(OUT, fn))
    meta.setdefault("script_sha256", {})[fn] = sha256(os.path.join(OUT, fn))

cams = json.load(open(os.path.join(DS, "cams.json")))
motion = json.load(open(os.path.join(DS, "motion_assumptions.json")))
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
if not SMOKE:
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
        f"git -C {WROOT} submodule update --init --quiet -- third_party/lingbot_depth && "
        f"git -C {WROOT}/third_party/lingbot_depth checkout -q {LINGBOT} && "
        f"git -C {WROOT} rev-parse HEAD | tee commits.txt && "
        f"git -C {WROOT}/third_party/lingbot_depth rev-parse HEAD | tee -a commits.txt"
    )
    if r:
        save(rc=21, stage="clone_wd3d")
        sys.exit(0)
    # Do NOT init third_party/sam3 (gated weights, unused; this job must not run SAM 3).
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
    t1 = time.time()
    r1 = sh(
        f"{sys.executable} wd3d_infer.py --wd3d-root {WROOT} --ckpt {ck} --inputs {DS} "
        f"--tags {tags} --outdir dets --device cuda:0 --fp16 1 --budget-s {BUDGET_S1}{mf} "
        f"--prompts {prompts} 2>&1 | tee log_stage1.txt"
    )
    stage1_s = round(time.time() - t1, 1)
    meta["stage1_rc"] = r1
    meta["stage1_s"] = stage1_s
    if r1:
        save(rc=23, stage="stage1_failed")
        sys.exit(0)
    # free WD3D before SAM/DAV2
    sh(f"rm -rf {WROOT} {TMP}/ckpt")
    save(stage="install_stage2")
    r = sh(
        "pip install -q hydra-core iopath opencv-python-headless huggingface_hub && "
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
    o = f"{OUT}/results/{tag}_foundation_bev.mp4"
    dets = f"{OUT}/dets/wd3d_{tag}.npz"
    if SMOKE:
        dets = dets if os.path.exists(dets) else ""
    spd = motion.get(tag, {}).get("speed_mps", 12.0)
    mf = f" --max-frames {MAXF}" if MAXF else ""
    smoke_flag = " --smoke" if SMOKE else ""
    device = "cpu" if SMOKE else "cuda:0"
    cmd = (
        f"{sys.executable} {OUT}/foundation_bev.py --video {v} --output {o} --cam-json {DS}/cams.json "
        f"--tag {tag} --speed {spd} --device {device} --weights {WEIGHTS} "
        f"--stills {OUT}/stills --tracks-csv {OUT}/tracks/{tag}_tracks.csv "
        f"--polylines-json {OUT}/polylines/{tag}_polylines.json "
        f"--clip-meta {OUT}/results/{tag}.json "
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
lines.append(f"- WildDet3D commit `{WD3D_SHA}` (lingbot_depth `{LINGBOT}`; SAM 3 submodule not initialised)")
lines.append(f"- WildDet3D checkpoint `{CKPT}` @ `{HF_REV}` sha256 `{CKPT_SHA}`")
lines.append(f"- sam2 pip git `{SAM2_PIP}`; weights `facebook/sam2.1-hiera-small` / `{SAM_CKPT}`")
lines.append(f"- Depth Anything V2 Small code `{DAV2_SHA}`; weights `{DAV2_CKPT}`")
lines.append(f"- GPU: {meta.get('gpu')}")
lines.append(f"- wall_s: {meta.get('elapsed_s')}; stage1_s: {stage1_s}; stage2_s: {meta.get('stage2_s')}; rc: {meta['rc']}")
lines.append("")
lines.append("## Prompting route per clip")
lines.append("")
lines.append("SAM 2.1 has no native text prompt. First route: WildDet3D open-vocab 2D boxes for")
lines.append("`lane line` / `lane marking` and `road edge` / `road curb`, used as SAM box prompts.")
lines.append("Fallback: SAM road-surface points in the lower road band, road edges from the left/right")
lines.append("mask boundary, lane centre-lines from thin bright (white/yellow) stripes inside the road mask.")
lines.append("")
for t in TAGS:
    rec = meta.get("videos", {}).get(t, {})
    lines.append(f"- **{t}**: route `{rec.get('prompt_route', 'n/a')}` "
                 f"(sample hits {rec.get('sample_hits', '?')}/{rec.get('n_sample', '?')}), "
                 f"rc={rec.get('rc')}, frames {rec.get('frames_out', rec.get('frames_in', '?'))}")
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
lines.append("- `results/<clip>_foundation_bev.mp4` — side-by-side front + BEV, H.264 when ffmpeg is available")
lines.append("- `stills/<clip>_fNNNN.png` — 2–3 mid / busy / early frames")
lines.append("- `tracks/<clip>_tracks.csv` — frame, track_id, class, score, bev_x, bev_z, x, y, z")
lines.append("- `polylines/<clip>_polylines.json` — per-frame lane and edge polylines in pixels and BEV metres")
open(f"{OUT}/README.md", "w").write("\n".join(lines) + "\n")

sh(f"find {OUT}/results {OUT}/stills {OUT}/tracks {OUT}/polylines {OUT}/run_meta.json {OUT}/README.md -type f 2>/dev/null | sort | xargs -r sha256sum > {OUT}/SHA256SUMS.txt")
save(stage="end")
print("RC", meta["rc"], "fails", fails or "none", flush=True)
