# app.py
# ---------------------------------------------------------------------------------------
# QR + Face Verification (LBPH, MobileNetV2, Swin-T, ORB) with per-algo pipeline previews
# ORB is rewritten to match your legacy code (gamma+CLAHE+ROI, affine RANSAC, kp1/kp2).
#
# Storage:
#   data/
#     student.json                       # your roster (QR payload == student_id)
#     faces/
#       LBPH/<id>/{model.xml,tau.json}
#       MobileNetV2/<id>/{emb.npy,tau.json}
#       SwinT/<id>/{emb.npy,tau.json}
#       ORB/<id>/{ref.png,ref_orb.npz,tau.json}
#
# Run:
#   streamlit run app.py
# Reqs:
#   pip install streamlit opencv-contrib-python numpy
#   (optional barcodes) pip install pyzbar
#   (cnn) pip install tensorflow
#   (swin) pip install torch torchvision timm
# ---------------------------------------------------------------------------------------

import os, json, time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# optional barcodes
try:
    from pyzbar.pyzbar import decode as zbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False

# ============================= Paths / Config =============================
DATA_DIR        = Path("data")
ROSTER_JSON     = DATA_DIR / "student.json"
FACES_DIR       = DATA_DIR / "faces"
ALGOS           = ["LBPH", "MobileNetV2", "SwinT", "ORB"]

# ---- LBPH config ----
FACE_SIZE_LBPH  = (112, 112)
USE_SOFT_MASK   = True
SHRINK_BOX      = 0.90
BLUR_VAR_MIN    = 80.0
BRIGHT_MINMAX   = (40, 210)
HAAR_NORMAL     = dict(scale=1.20, neighbors=5, minsize=100, upscale=1.0)
HAAR_SMALL      = dict(scale=1.05, neighbors=3,  minsize=40,  upscale=1.5)
GRID_XY         = (8, 8)
GLOBAL_TAU_LBPH = 60.0

# ---- CNN/Swin shared preprocess (round mask only) ----
IMSIZE_CNN      = 224
KERAS_MNET_PATH = "output/checkpoints_mnet/best_mnetv2.keras"   # <- change if needed
SWIN_CKPT_PATH  = "output/checkpoints_transformer/best.pt"      # <- change if needed
GLOBAL_TAU_MNET = 0.90
GLOBAL_TAU_SWIN = 0.85

# ---- ORB config (legacy) ----
cv2.setRNGSeed(12345)  # deterministic RANSAC
ORB_SIZE        = 160
RATIO_TEST      = 0.78
RANSAC_REPROJ   = 4.0
RANSAC_ITERS    = 2000
RANSAC_CONF     = 0.99
GLOBAL_TAU_ORB  = {"tau_inliers": 10, "tau_goods": 18}   # your defaults

# ============================= Roster =============================
def load_roster():
    if ROSTER_JSON.exists():
        try:
            obj = json.load(open(ROSTER_JSON, "r", encoding="utf-8"))
            if isinstance(obj, dict) and "students" in obj:
                return {s["student_id"]: s for s in obj["students"]}
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}
ROSTER = load_roster()
def roster_get(uid: str): return ROSTER.get(uid, {})

# ============================= QR / Barcode =============================
def decode_qr_or_barcode(bgr):
    if _HAS_PYZBAR:
        try:
            res = zbar_decode(bgr)
            if res:
                return res[0].data.decode("utf-8", errors="ignore")
        except Exception:
            pass
    try:
        det = cv2.QRCodeDetector()
        data, pts, _ = det.detectAndDecode(bgr)
        if data: return data
    except Exception:
        pass
    return None

def _draw_qr_overlays(vis_bgr):
    shown = False
    if _HAS_PYZBAR:
        try:
            for r in zbar_decode(vis_bgr):
                pts = r.polygon
                if pts and len(pts) >= 4:
                    pts = np.array([(p.x, p.y) for p in pts], dtype=np.int32)
                    cv2.polylines(vis_bgr, [pts], True, (0, 255, 0), 2)
                    shown = True
        except Exception:
            pass
    try:
        det = cv2.QRCodeDetector()
        data, pts, _ = det.detectAndDecode(vis_bgr)
        if pts is not None and len(pts):
            pts = pts.astype(int).reshape(-1,1,2)
            cv2.polylines(vis_bgr, [pts], True, (0,255,0), 2)
            shown = True
    except Exception:
        pass
    return shown

# ============================= Face detect (shared) =============================
def _haar():
    c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if c.empty(): raise RuntimeError("Failed to load Haar cascade.")
    return c
_FACE_CASCADE = _haar()

def detect_largest_face_bgr(bgr_img, scaleFactor=1.1, minNeighbors=5):
    if bgr_img is None: return None, None
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor, minNeighbors)
    if len(faces) == 0: return None, None
    (x, y, w, h) = max(faces, key=lambda r: r[2]*r[3])
    return (x, y, w, h), bgr_img[y:y+h, x:x+w]

def draw_overlays(bgr, algo, small_face=False):
    vis = bgr.copy()
    try:
        box, _ = detect_largest_face_bgr(bgr)
        if box is not None:
            x,y,w,h = box
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,200,255), 2)
    except Exception:
        pass
    _draw_qr_overlays(vis)
    return vis

# ============================= LBPH backend =============================
def _ensure_cv2_face():
    if not hasattr(cv2, "face"):
        raise ImportError("cv2.face not found. Install: pip install opencv-contrib-python")
def _lbph_user_dir(uid): return FACES_DIR / "LBPH" / uid
def _lbph_model(uid):    return _lbph_user_dir(uid) / "model.xml"
def _lbph_tau(uid):      return _lbph_user_dir(uid) / "tau.json"
def _lbph_get_tau(uid):
    p = _lbph_tau(uid)
    if p.exists():
        try: return float(json.load(open(p,"r")).get("tau", GLOBAL_TAU_LBPH))
        except Exception: return GLOBAL_TAU_LBPH
    return GLOBAL_TAU_LBPH
def _lbph_set_tau(uid, tau):
    p = _lbph_tau(uid); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": float(tau)}, open(p,"w"), indent=2)
def _variance_of_laplacian(gray): return cv2.Laplacian(gray, cv2.CV_64F).var()
def _detect_largest(gray, *, small_face=False):
    p = HAAR_SMALL if small_face else HAAR_NORMAL
    c = _haar()
    up = cv2.resize(gray, None, fx=p["upscale"], fy=p["upscale"], interpolation=cv2.INTER_LINEAR) \
         if p["upscale"] != 1.0 else gray
    faces = c.detectMultiScale(up, p["scale"], p["neighbors"], flags=cv2.CASCADE_SCALE_IMAGE,
                               minSize=(p["minsize"], p["minsize"]))
    if len(faces)==0: return None
    x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
    if p["upscale"] != 1.0:
        x = int(x/p["upscale"]); y = int(y/p["upscale"]); w = int(w/p["upscale"]); h = int(h/p["upscale"])
    return (x,y,w,h)
def _shrink_box(x,y,w,h,shrink=SHRINK_BOX):
    cx, cy = x+w//2, y+h//2
    nw, nh = max(1,int(w*shrink)), max(1,int(h*shrink))
    nx, ny = cx-nw//2, cy-nh//2
    return nx, ny, nw, nh
def ellipse_alpha_center(shape_or_gray, scale=0.92):
    if isinstance(shape_or_gray, tuple): h,w = shape_or_gray
    else: h,w = shape_or_gray.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w/2.0, h/2.0
    rx, ry = (w*scale)/2.0, (h*scale)/2.0
    d = ((X - cx)/rx)**2 + ((Y - cy)/ry)**2
    return np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)
def _apply_soft_ellipse(gray):
    a = ellipse_alpha_center(gray, scale=0.92)
    return (gray.astype(np.float32) * a).astype(np.uint8)
def _apply_hard_ellipse(gray):
    mask = np.zeros_like(gray, np.uint8)
    h, w = gray.shape
    cv2.ellipse(mask, (w//2, h//2), (int(w*0.45), int(h*0.60)), 0,0,360,255,-1)
    return cv2.bitwise_and(gray, mask)
def preprocess_face_img_lbph(bgr_or_gray, detect=True, small_face=False, use_soft_mask=True, return_info=True):
    gray0 = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY) if bgr_or_gray.ndim==3 else bgr_or_gray
    face_box = None
    if detect: face_box = _detect_largest(gray0, small_face=small_face)
    gray = gray0
    if face_box is not None:
        x,y,w,h = _shrink_box(*face_box, shrink=SHRINK_BOX); x, y = max(0,x), max(0,y)
        gray = gray0[y:y+h, x:x+w]
    resized = cv2.resize(gray, FACE_SIZE_LBPH, interpolation=cv2.INTER_AREA)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(resized)
    masked = _apply_soft_ellipse(eq) if use_soft_mask else _apply_hard_ellipse(eq)
    info=None
    if return_info:
        blur = _variance_of_laplacian(masked); mean=float(np.mean(masked))
        ok = (blur >= BLUR_VAR_MIN) and (BRIGHT_MINMAX[0] <= mean <= BRIGHT_MINMAX[1])
        info = dict(face_box=face_box, blur_mask=blur, mean_mask=mean, ok=bool(ok),
                    small_face=bool(small_face), use_soft_mask=bool(use_soft_mask))
    return masked, info
def _make_lbph():
    _ensure_cv2_face()
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=GRID_XY[0], grid_y=GRID_XY[1])
def enroll_lbph(uid, frames_bgr, small_face=False, use_soft_mask=True, tau_init=GLOBAL_TAU_LBPH, min_keep=3):
    kept=[]
    for bgr in frames_bgr:
        proc, info = preprocess_face_img_lbph(bgr, detect=True, small_face=small_face, use_soft_mask=use_soft_mask, return_info=True)
        if info["ok"]: kept.append(proc)
    if len(kept) < min_keep: return False, f"Not enough good shots ({len(kept)}/{min_keep})."
    model = _make_lbph(); labels = np.zeros((len(kept),), dtype=np.int32); model.train(kept, labels)
    out = _lbph_model(uid); out.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out)); _lbph_set_tau(uid, tau_init)
    return True, f"LBPH enrolled {uid} with {len(kept)} shots."
def verify_lbph(uid, frame_bgr, small_face=False, use_soft_mask=True):
    mp = _lbph_model(uid)
    if not mp.exists(): return False, float("inf"), _lbph_get_tau(uid), {"error":"No LBPH model.xml for this ID."}
    model = _make_lbph(); model.read(str(mp))
    probe, info = preprocess_face_img_lbph(frame_bgr, detect=True, small_face=small_face, use_soft_mask=use_soft_mask, return_info=True)
    tau = _lbph_get_tau(uid)
    if not info["ok"]: return False, float("inf"), tau, {"quality_fail": info}
    label, dist = model.predict(probe); accept = (label==0) and (dist <= tau)
    dbg = {"label":int(label), "distance":float(dist), "tau":float(tau), **info}
    return bool(accept), float(dist), float(tau), dbg

# ============================= CNN/Swin shared preprocess (round mask only) =============================
def auto_gamma(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); m = float(gray.mean())
    gamma = np.clip(1.0 + (128.0 - m)/300.0, 0.7, 1.3)
    inv = 1.0/max(1e-6, gamma)
    table = (np.linspace(0,1,256, dtype=np.float32) ** inv * 255.0).astype(np.uint8)
    return cv2.LUT(bgr, table)
def clahe_bgr(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); L,A,B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    return cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2BGR)
def bilateral_bgr(bgr, d=9, sigmaColor=75, sigmaSpace=75): return cv2.bilateralFilter(bgr, d, sigmaColor, sigmaSpace)
def unsharp(bgr, amt=0.7, rad=1.5):
    blur = cv2.GaussianBlur(bgr, (0,0), rad)
    return cv2.addWeighted(bgr, 1+amt, blur, -amt, 0)
def preprocess_face_bgr_for_embed(bgr, out_size=IMSIZE_CNN):
    box, crop = detect_largest_face_bgr(bgr); 
    if crop is None: return None
    x = auto_gamma(crop); x = clahe_bgr(x); x = bilateral_bgr(x, 9, 75, 75); x = unsharp(x, 0.7, 1.5)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    alpha = ellipse_alpha_center(g, scale=0.92); g_m = (g.astype(np.float32)*alpha).astype(np.uint8)
    g3 = cv2.cvtColor(g_m, cv2.COLOR_GRAY2BGR)
    return cv2.resize(g3, (out_size, out_size), interpolation=cv2.INTER_AREA)

# ============================= MobileNetV2 backend =============================
def _mnet_user_dir(uid): return FACES_DIR / "MobileNetV2" / uid
def _mnet_emb(uid):      return _mnet_user_dir(uid) / "emb.npy"
def _mnet_tau(uid):      return _mnet_user_dir(uid) / "tau.json"
def _mnet_get_tau(uid):
    p = _mnet_tau(uid)
    if p.exists():
        try: return float(json.load(open(p,"r")).get("tau", GLOBAL_TAU_MNET))
        except Exception: return GLOBAL_TAU_MNET
    return GLOBAL_TAU_MNET
def _mnet_set_tau(uid, tau):
    p = _mnet_tau(uid); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": float(tau)}, open(p,"w"), indent=2)
_MNET_CACHE = {"model": None}
def _load_mnet_embedder():
    if _MNET_CACHE["model"] is not None: return _MNET_CACHE["model"]
    import tensorflow as tf
    from tensorflow import keras
    if not Path(KERAS_MNET_PATH).exists():
        raise FileNotFoundError(f"MobileNetV2 .keras not found at {KERAS_MNET_PATH}")
    clf = keras.models.load_model(KERAS_MNET_PATH, compile=False)
    feat = None
    try: feat = clf.get_layer("embedding").output
    except Exception:
        for L in reversed(clf.layers):
            if isinstance(L, (keras.layers.GlobalAveragePooling2D, keras.layers.GlobalMaxPooling2D)):
                feat = L.output; break
        if feat is None: feat = clf.layers[-2].output
    try: out = keras.layers.UnitNormalization(axis=-1, name="l2norm_eval")(feat)
    except Exception:
        out = keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1), name="l2norm_eval")(feat)
    emb = keras.Model(clf.input, out, name="mnet_embedder_eval"); emb.trainable=False
    _MNET_CACHE["model"] = emb; return emb
def _mnet_embed(bgr_tile):
    import tensorflow as tf
    emb = _load_mnet_embedder()
    rgb = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    z = emb.predict(np.expand_dims(rgb,0), verbose=0)[0].astype(np.float32)
    return z / (np.linalg.norm(z)+1e-9)
def enroll_mnet(uid, frames_bgr, shots_min=1):
    vecs=[]
    for bgr in frames_bgr:
        tile = preprocess_face_bgr_for_embed(bgr, IMSIZE_CNN)
        if tile is None: continue
        vecs.append(_mnet_embed(tile))
    if len(vecs) < shots_min: return False, f"Not enough usable shots ({len(vecs)}/{shots_min})."
    v = np.mean(np.stack(vecs,0),0); v = v/(np.linalg.norm(v)+1e-9)
    out = _mnet_emb(uid); out.parent.mkdir(parents=True, exist_ok=True); np.save(str(out), v.astype(np.float32))
    _mnet_set_tau(uid, GLOBAL_TAU_MNET); return True, f"MobileNetV2 enrolled {uid} with {len(vecs)} shots."
def verify_mnet(uid, frame_bgr):
    p = _mnet_emb(uid)
    if not p.exists(): return False, -1.0, _mnet_get_tau(uid), {"error":"No MobileNetV2 enrollment for this ID."}
    ref = np.load(str(p))
    tile = preprocess_face_bgr_for_embed(frame_bgr, IMSIZE_CNN)
    if tile is None: return False, -1.0, _mnet_get_tau(uid), {"quality_fail":"no face"}
    probe = _mnet_embed(tile); tau = _mnet_get_tau(uid)
    sim = float(np.dot(ref, probe) / (np.linalg.norm(ref)*np.linalg.norm(probe) + 1e-9))
    return (sim>=tau), sim, tau, {}

# ============================= Swin-T backend =============================
def _swin_user_dir(uid): return FACES_DIR / "SwinT" / uid
def _swin_emb(uid):      return _swin_user_dir(uid) / "emb.npy"
def _swin_tau(uid):      return _swin_user_dir(uid) / "tau.json"
def _swin_get_tau(uid):
    p=_swin_tau(uid)
    if p.exists():
        try: return float(json.load(open(p,"r")).get("tau", GLOBAL_TAU_SWIN))
        except Exception: return GLOBAL_TAU_SWIN
    return GLOBAL_TAU_SWIN
def _swin_set_tau(uid, tau):
    p=_swin_tau(uid); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": float(tau)}, open(p,"w"), indent=2)
_SWIN_CACHE = {"model": None}
def _load_swin_embedder():
    if _SWIN_CACHE["model"] is not None: return _SWIN_CACHE
    import torch, torch.nn as nn, timm
    class L2Norm(nn.Module):
        def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
        def forward(self,x): return torch.nn.functional.normalize(x, dim=1, eps=self.eps)
    class SwinTinyEmbedder(nn.Module):
        def __init__(self, embed_dim=256):
            super().__init__()
            self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0, global_pool="avg")
            in_feats = self.backbone.num_features
            self.proj = nn.Linear(in_feats, embed_dim, bias=False)
            self.l2 = L2Norm()
        def forward(self,x):
            f = self.backbone(x); e = self.proj(f); return self.l2(e)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTinyEmbedder(embed_dim=256).to(device).eval()
    if not Path(SWIN_CKPT_PATH).exists(): raise FileNotFoundError(f"Swin checkpoint not found: {SWIN_CKPT_PATH}")
    ckpt = torch.load(str(SWIN_CKPT_PATH), map_location="cpu")
    state=None
    for k in ("embedder_state","model_state_dict","state_dict","model_state"):
        if k in ckpt and isinstance(ckpt[k], dict): state = ckpt[k]; break
    if state is None and isinstance(ckpt, dict):
        state = ckpt
    if state is not None:
        state = { (k.split("embedder.",1)[-1] if k.startswith("embedder.") else k): v for k,v in state.items() }
        model.load_state_dict(state, strict=False)
    _SWIN_CACHE.update({"model":model, "device":device}); return _SWIN_CACHE
def _swin_embed(bgr_tile):
    import torch
    cache = _load_swin_embedder(); model, device = cache["model"], cache["device"]
    rgb = cv2.cvtColor(bgr_tile, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.transpose(rgb,(2,0,1))[None,...]; x = torch.from_numpy(x).to(device)
    with torch.no_grad(): z = model(x).detach().cpu().numpy()[0].astype(np.float32)
    return z/(np.linalg.norm(z)+1e-9)
def enroll_swin(uid, frames_bgr, shots_min=1):
    vecs=[]
    for bgr in frames_bgr:
        tile = preprocess_face_bgr_for_embed(bgr, IMSIZE_CNN)
        if tile is None: continue
        vecs.append(_swin_embed(tile))
    if len(vecs) < shots_min: return False, f"Not enough usable shots ({len(vecs)}/{shots_min})."
    v = np.mean(np.stack(vecs,0),0); v = v/(np.linalg.norm(v)+1e-9)
    out = _swin_emb(uid); out.parent.mkdir(parents=True, exist_ok=True); np.save(str(out), v.astype(np.float32))
    _swin_set_tau(uid, GLOBAL_TAU_SWIN); return True, f"Swin-T enrolled {uid} with {len(vecs)} shots."
def verify_swin(uid, frame_bgr):
    p = _swin_emb(uid)
    if not p.exists(): return False, -1.0, _swin_get_tau(uid), {"error":"No Swin-T enrollment for this ID."}
    ref = np.load(str(p))
    tile = preprocess_face_bgr_for_embed(frame_bgr, IMSIZE_CNN)
    if tile is None: return False, -1.0, _swin_get_tau(uid), {"quality_fail":"no face"}
    probe = _swin_embed(tile); tau = _swin_get_tau(uid)
    sim = float(np.dot(ref, probe) / (np.linalg.norm(ref)*np.linalg.norm(probe) + 1e-9))
    return (sim>=tau), sim, tau, {}

# ============================= ORB backend (LEGACY) =============================
def percentile_clip(gray, p1=1, p99=99):
    v1, v99 = np.percentile(gray, (p1, p99))
    if v99 <= v1: return gray
    return np.clip((gray - v1) * (255.0/(v99 - v1)), 0, 255).astype(np.uint8)
def apply_gamma(gray, gamma=1.0):
    if gamma <= 0: gamma = 1.0
    inv = 1.0/gamma
    table = np.arange(256, dtype=np.float32)/255.0
    table = np.power(table, inv) * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)[gray]
def choose_interpolation(src_shape, out_size):
    h,w = src_shape[:2]; oh,ow = out_size
    return cv2.INTER_AREA if (oh<h or ow<w) else cv2.INTER_CUBIC
def ellipse_mask(shape, scale=0.92):
    h,w = shape[:2]
    mask = np.zeros((h,w), np.uint8)
    cv2.ellipse(mask, (w//2, h//2), (int(w*scale/2), int(h*scale/2)), 0, 0, 360, 255, -1)
    return mask
def add_replicate_border(img, border=2):
    return cv2.copyMakeBorder(img, border,border,border,border, cv2.BORDER_REPLICATE)
def preprocess_face_orb(bgr_face, out_size=(ORB_SIZE, ORB_SIZE)):
    gray0 = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
    clipped = percentile_clip(gray0, 1, 99)
    mean_int = float(clipped.mean())
    gamma = np.clip(1.0 + (128.0 - mean_int) / 300.0, 0.7, 1.3)
    gamma_applied = apply_gamma(clipped, gamma)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gamma_applied)
    interp = choose_interpolation(eq.shape, out_size)
    resized = cv2.resize(eq, out_size, interpolation=interp)
    final = add_replicate_border(resized, border=2)
    return final  # gray uint8

# global kp lists (legacy style)
kp1 = None
kp2 = None

def extract_orb_features(gray_img, nfeatures=2000, mask=None):
    mean_int = float(gray_img.mean())
    fast_t = 5 if mean_int < 90 else 7
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=4,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=fast_t
    )
    kps, des = orb.detectAndCompute(gray_img, mask)
    return kps, des

def match_and_score(des1, des2, ratio=RATIO_TEST, require_model=True, model="affine"):
    if des1 is None or des2 is None or len(des1)<2 or len(des2)<2:
        return 0, [], None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m,n in knn if m.distance < ratio*n.distance]
    inlier_mask = None
    if require_model and len(good) >= 6 and kp1 is not None and kp2 is not None:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        if model == "affine":
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                                  ransacReprojThreshold=RANSAC_REPROJ,
                                                  maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            inlier_mask = mask.ravel().tolist()
            return int(np.sum(mask)), good, inlier_mask
    return len(good), good, inlier_mask

def verify_faces_orb(ref_gray, cap_gray, threshold_inliers=10, threshold_goods=18, force_model="affine"):
    global kp1, kp2
    ref_mask = ellipse_mask(ref_gray.shape, scale=0.92)
    cap_mask = ellipse_mask(cap_gray.shape, scale=0.92)
    kp1, des1 = extract_orb_features(ref_gray, mask=ref_mask)
    kp2, des2 = extract_orb_features(cap_gray, mask=cap_mask)
    score, good_matches, inlier_mask = match_and_score(des1, des2, ratio=RATIO_TEST, require_model=True, model=force_model)
    used_metric = "inliers" if inlier_mask is not None else "good_matches"
    threshold = threshold_inliers if used_metric=="inliers" else threshold_goods
    decision = score >= threshold
    inlier_ratio = (float(score)/max(1,len(good_matches))) if (inlier_mask is not None) else None
    dbg = {
        "used_metric": used_metric,
        "threshold": threshold,
        "score": int(score),
        "num_kp_ref": 0 if kp1 is None else len(kp1),
        "num_kp_cap": 0 if kp2 is None else len(kp2),
        "good_matches": len(good_matches),
        "inlier_mask": inlier_mask,
        "good_matches_list": good_matches,
        "inlier_ratio": inlier_ratio
    }
    return decision, score, dbg

def draw_matches_orb(ref_gray, cap_gray, matches, inlier_mask=None):
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    if inlier_mask is not None:
        inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
        return cv2.drawMatches(ref_gray, kp1, cap_gray, kp2, inlier_matches, None, flags=flags)
    return cv2.drawMatches(ref_gray, kp1, cap_gray, kp2, matches, None, flags=flags)

# persistence for ORB
def _orb_user_dir(uid): return FACES_DIR / "ORB" / uid
def _orb_ref_img(uid):  return _orb_user_dir(uid) / "ref.png"
def _orb_ref_npz(uid):  return _orb_user_dir(uid) / "ref_orb.npz"
def _orb_tau(uid):      return _orb_user_dir(uid) / "tau.json"
def _orb_get_tau(uid):
    p = _orb_tau(uid)
    if p.exists():
        try:
            j = json.load(open(p,"r"))
            return {"tau_inliers": int(j.get("tau_inliers", GLOBAL_TAU_ORB["tau_inliers"])),
                    "tau_goods":   int(j.get("tau_goods",   GLOBAL_TAU_ORB["tau_goods"]))}
        except Exception: pass
    return dict(GLOBAL_TAU_ORB)
def _orb_set_tau(uid, tau_inliers, tau_goods):
    p = _orb_tau(uid); p.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau_inliers": int(tau_inliers), "tau_goods": int(tau_goods)}, open(p,"w"), indent=2)

def enroll_orb(uid, frames_bgr, shots_min=1):
    # pick the shot with most keypoints after preprocessing
    best = None; best_k = -1
    for bgr in frames_bgr:
        box, crop = detect_largest_face_bgr(bgr)
        if crop is None: continue
        g = preprocess_face_orb(crop, out_size=(ORB_SIZE, ORB_SIZE))
        roi = ellipse_mask(g.shape, scale=0.92)
        kps, des = extract_orb_features(g, mask=roi)
        if kps is None or des is None or len(kps)==0: continue
        if len(kps) > best_k:
            best_k = len(kps); best = (g, kps, des)
    if best is None: return False, f"Not enough usable shots (0/{shots_min})."
    g, kps, des = best
    outd = _orb_user_dir(uid); outd.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_orb_ref_img(uid)), g)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    np.savez_compressed(str(_orb_ref_npz(uid)), kpts=pts, des=des.astype(np.uint8))
    t = _orb_get_tau(uid); _orb_set_tau(uid, t["tau_inliers"], t["tau_goods"])
    return True, f"ORB enrolled {uid} (ref with {best_k} keypoints)."

def verify_orb(uid, frame_bgr):
    npz = _orb_ref_npz(uid); ref_img_p = _orb_ref_img(uid)
    tau = _orb_get_tau(uid)
    if not (npz.exists() and ref_img_p.exists()):
        return False, 0, (tau["tau_inliers"], tau["tau_goods"]), {"error":"No ORB enrollment."}
    data = np.load(str(npz))
    ref_kpts_xy = data["kpts"].astype(np.float32)
    ref_des     = data["des"].astype(np.uint8)
    ref_gray    = cv2.imread(str(ref_img_p), cv2.IMREAD_GRAYSCALE)

    # build global kp1 from saved coords
    global kp1, kp2
    kp1 = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in ref_kpts_xy]

    box, crop = detect_largest_face_bgr(frame_bgr)
    if crop is None:
        return False, 0, (tau["tau_inliers"], tau["tau_goods"]), {"quality_fail":"no face"}
    probe_g = preprocess_face_orb(crop, out_size=(ORB_SIZE, ORB_SIZE))
    roi = ellipse_mask(probe_g.shape, scale=0.92)
    kp2, des2 = extract_orb_features(probe_g, mask=roi)
    score, good, inlier_mask = match_and_score(ref_des, des2, ratio=RATIO_TEST, require_model=True, model="affine")
    used = "inliers" if inlier_mask is not None else "good_matches"
    th   = tau["tau_inliers"] if used=="inliers" else tau["tau_goods"]
    ok   = score >= th
    dbg = {"used_metric":used, "threshold":int(th), "score":int(score),
           "good_matches":len(good), "num_kp_ref":len(kp1), "num_kp_cap":0 if kp2 is None else len(kp2),
           "face_box": box, "inlier_mask": inlier_mask, "good_matches_list": good}
    return bool(ok), int(score), (tau["tau_inliers"], tau["tau_goods"]), dbg

# ============================= Enrollment presence =============================
def has_enrollment(algo, uid):
    d = FACES_DIR / algo / uid
    if algo == "LBPH":
        return (d / "model.xml").exists()
    elif algo == "MobileNetV2":
        return (d / "emb.npy").exists()
    elif algo == "SwinT":
        return (d / "emb.npy").exists()
    elif algo == "ORB":
        return (d / "ref.png").exists() and (d / "ref_orb.npz").exists()
    return False

# ============================= Pipeline previews (per algo) =============================
def _rgb(img):
    if img is None: return None
    if img.ndim==2: return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preview_lbph_steps(bgr, small_face=False, use_soft=True):
    # original + box
    vis = draw_overlays(bgr, "LBPH", small_face=small_face)

    # detect & crop in gray
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    box = _detect_largest(gray0, small_face=small_face)
    if box is not None:
        x,y,w,h = _shrink_box(*box, shrink=SHRINK_BOX)
        x,y = max(0,x), max(0,y)
        crop = gray0[y:y+h, x:x+w]
    else:
        crop = gray0

    resized = cv2.resize(crop, FACE_SIZE_LBPH, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8)).apply(resized)
    soft_mask = _apply_soft_ellipse(clahe)
    hard_mask = _apply_hard_ellipse(clahe)
    lap = cv2.Laplacian(soft_mask, cv2.CV_64F)
    lap_vis = np.uint8(np.clip(np.abs(lap)/np.max(np.abs(lap)+1e-6)*255, 0, 255))

    # stats tile
    canvas = np.full((112, 220, 3), 255, np.uint8)
    blur = _variance_of_laplacian(soft_mask)
    mean = float(np.mean(soft_mask))
    lines = [
        f"Small-face: {'ON' if small_face else 'OFF'}",
        f"Mask: {'Soft' if use_soft else 'Hard'}",
        f"blur σ²: {blur:.0f}",
        f"mean: {mean:.0f}",
    ]
    y0=18
    for t in lines:
        cv2.putText(canvas, t, (6,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
        y0+=18

    return [
        ("original (boxed)", _rgb(vis)),
        ("raw crop(gray)", _rgb(crop)),
        ("resized 112×112", _rgb(resized)),
        ("CLAHE", _rgb(clahe)),
        ("mask (soft)", _rgb(soft_mask)),
        ("mask (hard)", _rgb(hard_mask)),
        ("Laplacian (edges)", _rgb(lap_vis)),
        ("stats", _rgb(canvas)),
    ]

def preview_cnn_steps(bgr):
    vis = draw_overlays(bgr, "MobileNetV2")
    box, crop = detect_largest_face_bgr(bgr)
    if crop is None: return [("original (boxed)", _rgb(vis))]
    step1 = auto_gamma(crop)
    step2 = clahe_bgr(step1)
    step3 = bilateral_bgr(step2, 9, 75, 75)
    step4 = unsharp(step3, 0.7, 1.5)
    gray  = cv2.cvtColor(step4, cv2.COLOR_BGR2GRAY)
    alpha = ellipse_alpha_center(gray, 0.92)
    gmask = (gray.astype(np.float32)*alpha).astype(np.uint8)
    g3    = cv2.cvtColor(gmask, cv2.COLOR_GRAY2BGR)
    final = cv2.resize(g3, (IMSIZE_CNN, IMSIZE_CNN), interpolation=cv2.INTER_AREA)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    lap  = np.uint8(np.clip(np.abs(lap)/np.max(np.abs(lap)+1e-6)*255, 0, 255))
    return [
        ("original (boxed)", _rgb(vis)),
        ("crop", _rgb(crop)),
        ("auto-gamma", _rgb(step1)),
        ("CLAHE", _rgb(step2)),
        ("bilateral", _rgb(step3)),
        ("unsharp", _rgb(step4)),
        ("gray+round mask", _rgb(gmask)),
        (f"final {IMSIZE_CNN}×{IMSIZE_CNN}", _rgb(final)),
    ]

def preview_orb_steps(bgr):
    vis = draw_overlays(bgr, "ORB")
    box, crop = detect_largest_face_bgr(bgr)
    if crop is None: return [("original (boxed)", _rgb(vis))]
    gray0 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clipped = percentile_clip(gray0, 1, 99)
    mean_int = float(clipped.mean()); gamma = np.clip(1.0 + (128.0 - mean_int)/300.0, 0.7, 1.3)
    gamma_applied = apply_gamma(clipped, gamma)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gamma_applied)
    interp = choose_interpolation(eq.shape, (ORB_SIZE, ORB_SIZE))
    resized = cv2.resize(eq, (ORB_SIZE, ORB_SIZE), interpolation=interp)
    final = add_replicate_border(resized, 2)
    roi = ellipse_mask(final.shape, 0.92)
    roi_vis = cv2.bitwise_and(final, final, mask=roi)
    return [
        ("original (boxed)", _rgb(vis)),
        ("crop(gray)", _rgb(gray0)),
        ("percentile clip", _rgb(clipped)),
        (f"gamma {gamma:.2f}", _rgb(gamma_applied)),
        ("CLAHE", _rgb(eq)),
        (f"resized {ORB_SIZE}×{ORB_SIZE}", _rgb(resized)),
        ("replicate border", _rgb(final)),
        ("ROI ellipse", _rgb(roi_vis)),
    ]

def show_pipeline(title, steps):
    st.markdown(f"**{title}**")
    cols_per_row = 4
    row = []
    for i, (name, img) in enumerate(steps):
        if i % cols_per_row == 0:
            row = st.columns(cols_per_row)
        with row[i % cols_per_row]:
            if img is not None:
                st.image(img, caption=name, use_container_width=True)

# ===========================Access UI===================================
from PIL import Image

def _safe_img(path):
    """Open an image if it exists; otherwise return None."""
    try:
        p = Path(path) if path else None
        if p and p.exists():
            return Image.open(p)
    except Exception:
        pass
    return None

def _get_ref_image(stu, uid):
    """Use roster ref_photo, or fall back to data/faces/Original/<uid>.png."""
    return (_safe_img(stu.get("ref_photo")) or
            _safe_img(f"data/faces/Original/{uid}.png"))

def render_accept_ui(stu: dict, uid: str, tag: str, det_vis_bgr=None):
    """Simple, clean verification card with ref photo + QR + details."""
    # minimal CSS
    st.markdown("""
    <style>
      .badge{display:inline-block;padding:4px 10px;border-radius:999px;
             background:#16a34a20;color:#166534;border:1px solid #16a34a;font-weight:600}
      .card{border:1px solid #e5e7eb;border-radius:14px;padding:14px;background:#fff}
      .muted{color:#6b7280}
    </style>
    """, unsafe_allow_html=True)

    # header line
    st.markdown(f"<span class='badge'>✅ VERIFIED</span> &nbsp; <span class='muted'>{tag}</span>", unsafe_allow_html=True)

# show detections overlay if provided
    if det_vis_bgr is not None:
        st.image(cv2.cvtColor(det_vis_bgr, cv2.COLOR_BGR2RGB), caption="Detections (QR + face)", use_container_width=True)

    # content card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        ref_img = _get_ref_image(stu, uid) if stu else None
        if ref_img:
            st.image(ref_img, caption="Reference", use_container_width=True)

    with c2:
        name = (stu or {}).get("name", "-")
        st.markdown(f"### {name}")
        st.write(f"ID: {uid}")
        st.write(f"Programme: {(stu or {}).get('programme_title','-')} ({(stu or {}).get('programme_code','-')})")
        st.write(f"Faculty: {(stu or {}).get('faculty','-')}")
        tg = (stu or {}).get("tutorial_group")
        if tg: st.write(f"Group: {tg}")
        seat = (stu or {}).get("seat")
        if seat: st.write(f"Seat: {seat}")
        st.caption(time.strftime("Time: %Y-%m-%d %H:%M:%S"))

    with c3:
        qr_img = _safe_img((stu or {}).get("qr_png"))
        if qr_img:
            st.image(qr_img, caption="QR", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================= Streamlit UI =============================
st.set_page_config(page_title="QR + Face Verification (Multi-Alg)", page_icon="🪪", layout="wide")
st.title("🪪 QR + Face Verification — LBPH / MobileNetV2 / Swin-T / ORB")

with st.sidebar:
    st.header("General")
    algo = st.selectbox("Algorithm", ALGOS, index=0)
    small_face = False
    use_soft = True
    if algo == "LBPH":
        st.header("LBPH")
        small_face = st.toggle("Small-face Haar (more sensitive)", value=False)
        use_soft   = st.toggle("Soft ellipse mask (vs hard)", value=True)
    st.caption("After you capture a face (Enroll or Verify), the algorithm’s preprocessing pipeline is shown.")

# 1) Scan QR / enter ID
st.subheader("1) Scan QR / Barcode")
frame = st.camera_input("Show your QR/barcode to the camera")
uid = None
last_captured_for_preview = None  # used for pipeline preview

if frame is not None:
    arr = np.frombuffer(frame.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    det_vis = draw_overlays(bgr, algo, small_face=small_face)
    st.image(cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB), caption="Detections (QR + face)", use_container_width=True)
    uid = decode_qr_or_barcode(bgr)

uid = st.text_input("or type ID manually", value=(uid or ""))

if not uid:
    st.stop()

stu = roster_get(uid)
if stu:
    st.success(f"ID: {uid} — {stu.get('name','')}")
else:
    st.warning(f"ID: {uid} (not found in roster)")

enrolled = has_enrollment(algo, uid)
st.write(f"Enrollment status for **{algo}**: {'✅ Found' if enrolled else '❌ Missing'}")

# Per-user thresholds
with st.expander("Per-user threshold(s)"):
    if algo == "LBPH":
        cur = _lbph_get_tau(uid)
        new = st.number_input("τ (LBPH — accept if distance ≤ τ)", value=float(cur), step=1.0)
        if st.button("Save τ (LBPH)"):
            _lbph_set_tau(uid, float(new)); st.success(f"Saved τ={new:.2f}")
    elif algo == "MobileNetV2":
        cur = _mnet_get_tau(uid)
        new = st.number_input("τ (MobileNetV2 — accept if cosine ≥ τ)", value=float(cur), step=0.01, min_value=0.0, max_value=1.0)
        if st.button("Save τ (MobileNetV2)"):
            _mnet_set_tau(uid, float(new)); st.success(f"Saved τ={new:.2f}")
    elif algo == "SwinT":
        cur = _swin_get_tau(uid)
        new = st.number_input("τ (Swin-T — accept if cosine ≥ τ)", value=float(cur), step=0.01, min_value=0.0, max_value=1.0)
        if st.button("Save τ (Swin-T)"):
            _swin_set_tau(uid, float(new)); st.success(f"Saved τ={new:.2f}")
    elif algo == "ORB":
        cur = _orb_get_tau(uid)
        t_inl = st.number_input("τ_inliers (primary)", value=int(cur["tau_inliers"]), step=1, min_value=0)
        t_gd  = st.number_input("τ_goods (fallback)", value=int(cur["tau_goods"]), step=1, min_value=0)
        if st.button("Save τ (ORB)"):
            _orb_set_tau(uid, int(t_inl), int(t_gd)); st.success(f"Saved ORB τ_inliers={t_inl}, τ_goods={t_gd}")

# 2) Action (no duplicates)
default_idx = 1 if enrolled else 0
mode = st.radio("2) Action", ["Enroll", "Verify"], index=default_idx, horizontal=True)

# 3) Enroll
if mode == "Enroll":
    st.subheader("Enrollment")
    shots = st.slider("Number of shots", 1, 20, 10 if algo=="LBPH" else 3)
    frames_bgr = []
    for i in range(shots):
        fi = st.camera_input(f"Enroll shot #{i+1}")
        if fi:
            arr = np.frombuffer(fi.read(), np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            frames_bgr.append(bgr)
            last_captured_for_preview = bgr
            ov = draw_overlays(bgr, algo, small_face=small_face)
            st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption=f"Shot #{i+1} (QR + face)", use_container_width=True)
    if st.button("Enroll now", type="primary", disabled=(len(frames_bgr)==0)):
        if algo == "LBPH":
            ok, msg = enroll_lbph(uid, frames_bgr, small_face=small_face, use_soft_mask=use_soft, tau_init=_lbph_get_tau(uid))
        elif algo == "MobileNetV2":
            ok, msg = enroll_mnet(uid, frames_bgr, shots_min=1)
        elif algo == "SwinT":
            ok, msg = enroll_swin(uid, frames_bgr, shots_min=1)
        elif algo == "ORB":
            ok, msg = enroll_orb(uid, frames_bgr, shots_min=1)
        st.success(msg) if ok else st.error(msg)

# 4) Verify
if mode == "Verify":
    st.subheader("Verification")
    v = st.camera_input("Capture verification frame")
    if v and st.button("Verify now", type="primary"):
        arr = np.frombuffer(v.read(), np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        last_captured_for_preview = bgr

        if algo == "LBPH":
            ok, score, tau, dbg = verify_lbph(uid, bgr, small_face=small_face, use_soft_mask=use_soft)
            tag = f"Distance = {score:.2f}  |  τ = {tau:.2f}  (accept if distance ≤ τ)"
            probe_proc, _info = preprocess_face_img_lbph(bgr, detect=True, small_face=small_face, use_soft_mask=use_soft, return_info=True)
            st.image(probe_proc, caption="Preprocessed (LBPH: CLAHE + ellipse)", use_container_width=True)
        elif algo == "MobileNetV2":
            ok, sim, tau, dbg = verify_mnet(uid, bgr)
            score = sim; tag = f"Cosine = {sim:.3f}  |  τ = {tau:.3f}  (accept if cosine ≥ τ)"
            tile = preprocess_face_bgr_for_embed(bgr, out_size=IMSIZE_CNN)
            if tile is not None: st.image(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), caption="Preprocessed (CNN: round mask)", use_container_width=True)
        elif algo == "SwinT":
            ok, sim, tau, dbg = verify_swin(uid, bgr)
            score = sim; tag = f"Cosine = {sim:.3f}  |  τ = {tau:.3f}  (accept if cosine ≥ τ)"
            tile = preprocess_face_bgr_for_embed(bgr, out_size=IMSIZE_CNN)
            if tile is not None: st.image(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), caption="Preprocessed (Swin: round mask)", use_container_width=True)
        elif algo == "ORB":
            ok, score, taus, dbg = verify_orb(uid, bgr)
            t_inl, t_gd = taus
            tag = f"Score = {score}  |  τ_inliers={t_inl} / τ_goods={t_gd} (accept if score ≥ τ)"
            # matches visualization
            npz = _orb_ref_npz(uid); ref_img_p = _orb_ref_img(uid)
            if npz.exists() and ref_img_p.exists() and "good_matches_list" in dbg:
                ref_gray = cv2.imread(str(ref_img_p), cv2.IMREAD_GRAYSCALE)
                box, crop = detect_largest_face_bgr(bgr)
                if crop is not None:
                    probe_g = preprocess_face_orb(crop, out_size=(ORB_SIZE, ORB_SIZE))
                    vis = draw_matches_orb(ref_gray, probe_g, dbg["good_matches_list"], dbg.get("inlier_mask"))
                    st.image(_rgb(vis), caption="ORB: Matches / Inliers", use_container_width=True)
        else:
            ok, score, tag, dbg = False, 0, "Unknown algo", {}

        if ok:
            det_vis = draw_overlays(bgr, algo, small_face=small_face)
            render_accept_ui(stu, uid, tag, det_vis_bgr=det_vis)
        else:
            st.markdown("Result: ❌ REJECT")
            st.write(tag)
            det_vis = draw_overlays(bgr, algo, small_face=small_face)
            st.image(cv2.cvtColor(det_vis, cv2.COLOR_BGR2RGB), caption="Detections (QR + face)", use_container_width=True)

# ---------- Pipeline preview (from last captured face frame) ----------
if last_captured_for_preview is not None:
    st.markdown("### Pipeline Preview")
    if algo == "LBPH":
        steps = preview_lbph_steps(last_captured_for_preview, small_face=small_face, use_soft=use_soft)
        show_pipeline("LBPH — resized → CLAHE → ellipse (+edges/stats)", steps)
    elif algo == "MobileNetV2":
        steps = preview_cnn_steps(last_captured_for_preview)
        show_pipeline("MobileNetV2 — gamma → CLAHE → bilateral → unsharp → gray+mask → tile", steps)
    elif algo == "SwinT":
        steps = preview_cnn_steps(last_captured_for_preview)
        show_pipeline("Swin-T — gamma → CLAHE → bilateral → unsharp → gray+mask → tile", steps)
    elif algo == "ORB":
        steps = preview_orb_steps(last_captured_for_preview)
        show_pipeline("ORB — clip → gamma → CLAHE → resize → border → ROI", steps)
