# app.py
import os
import io
import math
import json
import difflib
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from PIL import Image
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import torch
from transformers import CLIPProcessor, CLIPModel
import requests

# ---------------------------
# Config & constants
# ---------------------------
app = FastAPI(title="Threat Modeling (STRIDE) - PoC with Gemini")

CANDIDATE_CLASSES = [
    "user", "internet", "load balancer", "application server", "api gateway",
    "database", "cache", "queue", "file storage", "kms", "key management",
    "waf", "firewall", "monitoring", "external service", "auth service"
]

# STRIDE rules map (component -> list of threat strings)
STRIDE_RULES = {
    "database": ["Information Disclosure (I)", "Tampering (T)", "Elevation of Privilege (E)"],
    "api gateway": ["Denial of Service (D)", "Tampering (T)"],
    "load balancer": ["Denial of Service (D)"],
    "application server": ["Tampering (T)", "Information Disclosure (I)", "Repudiation (R)"],
    "user": ["Spoofing (S)", "Repudiation (R)"],
    "auth service": ["Spoofing (S)", "Elevation of Privilege (E)"],
    "kms": ["Information Disclosure (I)", "Elevation of Privilege (E)"],
    "waf": ["Tampering (T)"],
    "firewall": ["Tampering (T)", "Denial of Service (D)"],
    "monitoring": ["Repudiation (R)"],
    "external service": ["Information Disclosure (I)", "Spoofing (S)"],
    "cache": ["Information Disclosure (I)"],
    "queue": ["Tampering (T)", "Denial of Service (D)"],
    "file storage": ["Information Disclosure (I)", "Tampering (T)"]
}

CANONICAL_SYNONYMS = {
    "db": "database", "rds": "database", "postgres": "database", "mysql": "database",
    "s3": "file storage", "bucket": "file storage",
    "alb": "load balancer", "loadbalancer": "load balancer",
    "api": "api gateway", "gateway": "api gateway",
    "auth": "auth service", "kms": "kms",
    "waf": "waf", "firewall": "firewall",
    "cache": "cache", "redis": "cache", "memcached": "cache",
    "queue": "queue", "rabbitmq": "queue", "sqs": "queue",
    "monitor": "monitoring", "cloudwatch": "monitoring", "logging": "monitoring"
}

# CLIp model (zero-shot) loaded once
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Loading CLIP model {CLIP_MODEL_NAME} on {device} (this may take a while)...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
print("CLIP loaded.")

# ---------------------------
# Utility helpers
# ---------------------------
def severity_for(threat: str, exposed: bool = False) -> str:
    base = {"I": "Medium", "T": "High", "D": "High", "S": "High", "E": "High", "R": "Medium"}
    key = threat.split("(")[-1].strip(")")
    sev = base.get(key, "Medium")
    if exposed and key in ["I", "D", "S"]:
        if sev == "Medium":
            return "High"
        if sev == "High":
            return "Critical"
    return sev

def fuzzy_to_canonical(text: str):
    if not text:
        return None
    t = text.lower().strip()
    for k, v in CANONICAL_SYNONYMS.items():
        if k in t:
            return v
    best = difflib.get_close_matches(t, CANDIDATE_CLASSES, n=1, cutoff=0.6)
    if best:
        return best[0]
    return None

def read_imagefile(file_bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# ---------------------------
# OCR & detection helpers
# ---------------------------
def detect_texts_and_labels(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    texts = []
    for i, txt in enumerate(data['text']):
        if not isinstance(txt, str) or txt.strip() == "":
            continue
        conf_raw = data['conf'][i]
        try:
            conf = float(conf_raw)
        except:
            conf = -1.0
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        texts.append({"text": txt.strip(), "conf": conf, "bbox": (x, y, w, h)})
    return texts

def find_candidate_boxes(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive thresholding and morphology for box-like shapes
    _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    close = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h_img, w_img = img.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 500 or w < 20 or h < 20:
            continue
        if w > 0.9 * w_img and h > 0.9 * h_img:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        rows, cols = 4, 6
        rw, rh = w_img // cols, h_img // rows
        for r in range(rows):
            for c in range(cols):
                x = c * rw; y = r * rh
                boxes.append((x, y, rw, rh))
    return boxes

def crop_image(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def clip_classify_crop(pil_img: Image.Image, candidate_labels=CANDIDATE_CLASSES):
    inputs = clip_processor(text=candidate_labels, images=pil_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return candidate_labels[idx], float(probs[idx])

def detect_lines_connections(img: np.ndarray, boxes: List[Tuple[int,int,int,int]]):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, threshold=60, minLineLength=30, maxLineGap=15)
    connections = []
    if lines is None:
        return connections
    centers = []
    for i, (x, y, w, h) in enumerate(boxes):
        centers.append((i, (x + w/2, y + h/2)))
    def nearest_box(px, py):
        best_i = None; best_d = 1e9
        for i,(cx,cy) in centers:
            d = (cx-px)**2 + (cy-py)**2
            if d < best_d:
                best_d = d; best_i = i
        return best_i
    for l in lines:
        x1, y1, x2, y2 = l[0]
        a = nearest_box(x1, y1)
        b = nearest_box(x2, y2)
        if a is not None and b is not None and a != b:
            pair = tuple(sorted((a, b)))
            if pair not in connections:
                connections.append(pair)
    return connections

# ---------------------------
# Gemini integration
# ---------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}"

def call_gemini_polish(findings_json: dict, timeout: int = 30) -> str:
    """
    Call Gemini generateContent to polish/generate a textual STRIDE report.
    Returns string text (or raises exception).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    url = GEMINI_URL_TEMPLATE.format(GEMINI_API_KEY)
    prompt = (
        "Você é um especialista em segurança da informação. Gere um relatório STRIDE conciso e técnico "
        "com base no JSON de evidências abaixo. Para cada componente, inclua: (1) resumo da ameaça STRIDE, "
        "(2) justificativa curta baseada nas evidências, (3) severidade (Low/Medium/High/Critical), "
        "(4) recomendações práticas (quick wins + config examples), e (5) referências técnicas (CWE/OWASP/NIST) "
        "quando aplicável. Seja objetivo e use linguagem técnica.\n\nEVIDENCES_JSON:\n"
        + json.dumps(findings_json, indent=2, ensure_ascii=False)
    )
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "text/plain",
            # you can add other controls here if desired
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    # resilient extraction: check likely keys
    text_out = ""
    if not data:
        raise RuntimeError("Empty response from Gemini")
    # Common pattern: 'candidates' list with 'output' or 'content' or 'content'->'parts'
    candidates = data.get("candidates") or data.get("candidates", [])
    if isinstance(candidates, list) and len(candidates) > 0:
        c = candidates[0]
        # try multiple fields
        if isinstance(c.get("output"), str) and c.get("output").strip():
            text_out = c["output"].strip()
        elif isinstance(c.get("content"), str) and c.get("content").strip():
            text_out = c["content"].strip()
        else:
            # sometimes 'output' is nested differently; fallback to join parts
            # try c.get("content", []) as list
            content_field = c.get("content")
            if isinstance(content_field, list):
                parts = []
                for p in content_field:
                    if isinstance(p, dict):
                        parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        parts.append(p)
                text_out = "\n".join([p for p in parts if p])
            else:
                # ultimate fallback: stringify candidate
                text_out = json.dumps(c, ensure_ascii=False)
    else:
        # other possible shape: top-level 'candidates' absent but top-level 'output' present
        if isinstance(data.get("output"), str):
            text_out = data.get("output")
        else:
            text_out = json.dumps(data, ensure_ascii=False)
    return text_out

# ---------------------------
# Core analysis pipeline
# ---------------------------
def analyze_image_bytes(file_bytes: bytes) -> Tuple[str, dict]:
    """
    Runs detection heuristics and returns:
      - local plain text report (fallback / immediate)
      - a findings_json dictionary (structured) suitable to send to Gemini
    """
    img = read_imagefile(file_bytes)
    h_img, w_img = img.shape[:2]

    ocr_texts = detect_texts_and_labels(img)
    boxes = find_candidate_boxes(img)
    components = []

    for idx, bbox in enumerate(boxes):
        crop = crop_image(img, bbox)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        clip_label, clip_score = clip_classify_crop(pil)
        found_label = None
        for t in ocr_texts:
            tx, ty, tw, th = t['bbox']
            if tx >= bbox[0] and ty >= bbox[1] and (tx+tw) <= (bbox[0]+bbox[2]) and (ty+th) <= (bbox[1]+bbox[3]):
                found_label = t['text']
                break
        canonical = None
        exposed = False
        if found_label:
            possible = fuzzy_to_canonical(found_label)
            if possible:
                canonical = possible
            else:
                canonical = clip_label
            if "internet" in found_label.lower() or "public" in found_label.lower() or "external" in found_label.lower():
                exposed = True
        else:
            canonical = clip_label
            x, y, w, h = bbox
            pad = 0.05
            if x < w_img*pad or y < h_img*pad or (x+w) > w_img*(1-pad) or (y+h) > h_img*(1-pad):
                exposed = True

        components.append({
            "id": idx,
            "bbox": bbox,
            "label_raw": found_label or "",
            "label_clip": clip_label,
            "clip_score": float(clip_score),
            "type": (canonical.lower() if isinstance(canonical, str) else canonical),
            "exposed": bool(exposed)
        })

    connections = detect_lines_connections(img, boxes)

    findings = []
    for c in components:
        ctype = c.get("type", "")
        if not ctype:
            continue
        mapped = fuzzy_to_canonical(ctype) or ctype
        threats = STRIDE_RULES.get(mapped, [])
        if threats:
            findings.append({
                "component_id": c['id'],
                "component_type": mapped,
                "label_raw": c['label_raw'],
                "evidence": {
                    "clip": {"label": c['label_clip'], "score": c['clip_score']},
                    "ocr": c['label_raw'],
                    "bbox": c['bbox'],
                    "exposed": c['exposed']
                },
                "threats": threats
            })

    # Create a local plain text report (fallback)
    lines = []
    lines.append("Threat modeling (STRIDE) - automatic PoC (local heuristics)\n")
    lines.append("Detected components:")
    for c in components:
        lines.append(f"- id={c['id']} type={c['type']} exposed={c['exposed']} clip={c['label_clip']} score={c['clip_score']:.2f} ocr='{c['label_raw']}'")
    lines.append("\nInferred threats and short recommendations:")
    if not findings:
        lines.append("No clear components detected or no rules fired. Try a clearer diagram or different crop.")
    else:
        for f in findings:
            lines.append(f"\nComponent id={f['component_id']} ({f['component_type']})")
            lines.append(f" Evidence: clip={f['evidence']['clip']['label']}({f['evidence']['clip']['score']:.2f}) ocr='{f['evidence']['ocr']}'")
            lines.append(f" Exposed: {f['evidence']['exposed']}")
            for t in f['threats']:
                sev = severity_for(t, exposed=f['evidence']['exposed'])
                code = t.split("(")[-1].strip(")")
                if code == "I":
                    recs = ["Enable encryption at rest/in-transit; restrict network access; enable audit logging; use KMS."]
                elif code == "T":
                    recs = ["Input validation; RBAC; restrict write access; backups and integrity checks."]
                elif code == "D":
                    recs = ["WAF/rate limiting; autoscaling; cloud DDoS protections; circuit breakers."]
                elif code == "S":
                    recs = ["Strong auth (MFA, OAuth2), centralized IAM, identity proofing."]
                elif code == "E":
                    recs = ["Least privilege, strict KMS policies, workload isolation."]
                elif code == "R":
                    recs = ["Centralized tamper-proof logging (SIEM), retention and auditing."]
                else:
                    recs = ["Apply standard security hardening and monitoring."]
                lines.append(f"  - Threat: {t} | Severity: {sev}")
                lines.append(f"    Recommendations: {'; '.join(recs)}")

    local_report_text = "\n".join(lines)

    # Compose findings_json for Gemini
    findings_json = {
        "summary": "Auto-detected components and STRIDE candidate threats (heuristic).",
        "components": components,
        "connections": [{"a": a, "b": b} for (a, b) in connections],
        "findings": findings
    }

    return local_report_text, findings_json

# ---------------------------
# FastAPI endpoint
# ---------------------------
@app.post("/analyze", response_class=PlainTextResponse)
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        fallback_text, findings_json = analyze_image_bytes(content)

        # Try to call Gemini if key present
        if GEMINI_API_KEY:
            try:
                polished = call_gemini_polish(findings_json, timeout=60)
                # return polished text (if not empty)
                if polished and polished.strip():
                    return PlainTextResponse(content=polished, media_type="text/plain")
                else:
                    # fallback
                    return PlainTextResponse(content=fallback_text, media_type="text/plain")
            except Exception as e:
                # If Gemini call fails, return fallback plus error note (do not leak stack)
                note = "\n\n---\nGemini polishing failed or not available; returning local heuristic report. Error: " + str(e)
                return PlainTextResponse(content=(fallback_text + note), media_type="text/plain")
        else:
            # No API key configured: return local report
            note = "\n\n---\nNo GEMINI_API_KEY configured. Set GEMINI_API_KEY env var to enable polishing using Gemini."
            return PlainTextResponse(content=(fallback_text + note), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
