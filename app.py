import os
import io
import math
import json
import re
import difflib
from typing import List, Tuple, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from PIL import Image
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import torch
from transformers import CLIPProcessor, CLIPModel
import requests

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------
# Config & constants
# ---------------------------
app = FastAPI(title="Threat Modeling (STRIDE) - PoC with Gemini (formatted output)")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=" * 80)
print("CORS middleware configured successfully!")
print("Allowed origins: *")
print("=" * 80)

CANDIDATE_CLASSES = [
    "user", "internet", "load balancer", "application server", "api gateway",
    "database", "cache", "queue", "file storage", "kms", "key management",
    "waf", "firewall", "monitoring", "external service", "auth service"
]

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

device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
print(f"Loading CLIP model {CLIP_MODEL_NAME} on {device} (this may take a while)...")
try:
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("CLIP loaded successfully!")
except Exception as exc:
    print("Warning: failed to load CLIP model, using fallback. Error:", str(exc))
    clip_model = None
    clip_processor = None

# ---------------------------
# Helper utilities
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

# Mapeia sinônimos e abreviações para nomes canônicos.
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

# Converte bytes da imagem em array NumPy para processamento com OpenCV.
def read_imagefile(file_bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
    if clip_model is None or clip_processor is None:
        try:
            txt = pytesseract.image_to_string(pil_img).strip().lower()
            if txt:
                best = difflib.get_close_matches(txt, candidate_labels, n=1, cutoff=0.4)
                if best:
                    return best[0], 0.6
        except Exception:
            pass
        return "application server", 0.35

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={}"


def call_gemini_polish_raw(findings_json: dict, timeout: int = 30) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    url = GEMINI_URL_TEMPLATE.format(GEMINI_API_KEY)

    prompt = (
        "Você é um especialista sênior em segurança da informação e arquitetura de sistemas. "
        "Analise o JSON de evidências abaixo e gere um relatório STRIDE COMPLETO e TÉCNICO.\n\n"

        "Para CADA componente identificado, você DEVE incluir:\n\n"

        "1. RESUMO DA AMEAÇA: Descrição concisa da ameaça STRIDE identificada\n"
        "2. JUSTIFICATIVA TÉCNICA: Explique POR QUE essa ameaça existe neste contexto específico, "
        "baseando-se nas evidências (tipo de componente, exposição, conexões)\n"
        "3. SEVERIDADE: Classifique como Low/Medium/High/Critical considerando:\n"
        "   - Exposição à internet/rede pública\n"
        "   - Tipo de dado processado\n"
        "   - Impacto potencial no negócio\n"
        "4. VETOR DE ATAQUE: Descreva como um atacante poderia explorar essa vulnerabilidade\n"
        "5. CENÁRIO DE EXPLORAÇÃO: Exemplo concreto de ataque (2-3 linhas)\n"
        "6. PROPOSTAS DE CORREÇÃO (OBRIGATÓRIO - mínimo 3 itens):\n"
        "   a) Quick Wins: Ações imediatas que podem ser implementadas em 1-2 dias\n"
        "      - Seja ESPECÍFICO: cite ferramentas, serviços cloud, configurações exatas\n"
        "      - Exemplo: 'Ativar AWS WAF com regra OWASP Top 10' ao invés de 'usar WAF'\n"
        "   b) Configurações Técnicas: Exemplos de código/config prontos para uso\n"
        "      - Inclua snippets de configuração (JSON, YAML, código)\n"
        "      - Exemplo: policy IAM, regra de firewall, header de segurança\n"
        "   c) Melhorias Arquiteturais: Mudanças estruturais de médio prazo\n"
        "      - Sugira padrões como Zero Trust, Defense in Depth, etc\n"
        "   d) Controles Compensatórios: Se a correção completa não for viável\n"
        "7. REFERÊNCIAS TÉCNICAS:\n"
        "   - CWE (Common Weakness Enumeration) relevante\n"
        "   - OWASP Top 10 categoria\n"
        "   - NIST guidelines aplicáveis\n"
        "   - MITRE ATT&CK técnicas relacionadas\n"
        "8. PRIORIZAÇÃO: Ordene as correções por impacto vs esforço\n"
        "9. MÉTRICAS DE SUCESSO: Como validar que a correção foi efetiva\n\n"

        "REQUISITOS CRÍTICOS:\n"
        "- Use linguagem técnica precisa, sem jargões genéricos\n"
        "- Todas as recomendações devem ser ACIONÁVEIS e ESPECÍFICAS\n"
        "- Inclua exemplos de código/configuração quando aplicável\n"
        "- Considere o contexto cloud (AWS/Azure/GCP) nas recomendações\n"
        "- Para componentes expostos à internet, seja EXTRA específico nas correções\n"
        "- Priorize correções baseadas em frameworks modernos (NIST CSF, ISO 27001)\n\n"

        "FORMATO DE SAÍDA:\n"
        "Retorne um JSON estruturado com a seguinte hierarquia:\n"
        "{\n"
        '  "STRIDE_Report": {\n'
        '    "description": "Relatório de Threat Modeling STRIDE",\n'
        '    "analysis_date": "data atual",\n'
        '    "components": [\n'
        "      {\n"
        '        "component_id": int,\n'
        '        "component_type": string,\n'
        '        "exposure_level": "internal/external/internet-facing",\n'
        '        "threats": [\n'
        "          {\n"
        '            "type": "STRIDE category (ex: Spoofing)",\n'
        '            "summary": "resumo da ameaça",\n'
        '            "justification": "por que essa ameaça existe aqui",\n'
        '            "attack_vector": "como explorar",\n'
        '            "exploitation_scenario": "exemplo de ataque",\n'
        '            "severity": "Low/Medium/High/Critical",\n'
        '            "remediation": {\n'
        '              "quick_wins": ["ação 1", "ação 2", "ação 3"],\n'
        '              "technical_configs": [\n'
        "                {\n"
        '                  "description": "o que fazer",\n'
        '                  "example_config": "código/config exemplo",\n'
        '                  "platform": "AWS/Azure/GCP/On-prem"\n'
        "                }\n"
        "              ],\n"
        '              "architectural_improvements": ["melhoria 1", "melhoria 2"],\n'
        '              "compensating_controls": ["controle 1", "controle 2"]\n'
        "            },\n"
        '            "prioritization": {\n'
        '              "priority": "P0/P1/P2/P3",\n'
        '              "effort": "baixo/médio/alto",\n'
        '              "impact": "baixo/médio/alto/crítico",\n'
        '              "recommended_timeline": "imediato/1-2 semanas/1-3 meses"\n'
        "            },\n"
        '            "validation": ["como testar correção 1", "como testar 2"],\n'
        '            "references": ["CWE-XXX", "OWASP A0X", "NIST guideline", "MITRE TXXXX"]\n'
        "          }\n"
        "        ]\n"
        "      }\n"
        "    ],\n"
        '    "executive_summary": {\n'
        '      "total_threats": int,\n'
        '      "critical_count": int,\n'
        '      "high_count": int,\n'
        '      "top_3_priorities": ["prioridade 1", "prioridade 2", "prioridade 3"],\n'
        '      "estimated_remediation_time": "tempo estimado total"\n'
        "    }\n"
        "  }\n"
        "}\n\n"

        "EVIDÊNCIAS DO DIAGRAMA:\n"
        + json.dumps(findings_json, indent=2, ensure_ascii=False)
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "text/plain",
            "temperature": 0.4,
            "maxOutputTokens": 8192
        }
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_json_from_gemini_text(gemini_text: str) -> Tuple[Optional[dict], str]:
    if not gemini_text:
        return None, ""

    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", gemini_text, flags=re.IGNORECASE)
    json_text = None
    cleaned = gemini_text
    if m:
        json_text = m.group(1)
        cleaned = gemini_text[:m.start()] + gemini_text[m.end():]
    else:
        m2 = re.search(r"(\{[\s\S]{100,}\})", gemini_text)
        if m2:
            json_text = m2.group(1)
            cleaned = gemini_text.replace(json_text, "")

    parsed = None
    if json_text:
        try:
            parsed = json.loads(json_text)
        except Exception:
            try:
                fixed = re.sub(r",\s*\}\s*", "}", json_text)
                fixed = fixed.replace("'", '"')
                parsed = json.loads(fixed)
            except Exception:
                parsed = None
    return parsed, cleaned.strip()


def pretty_json_from_gemini(gemini_text: str) -> str:
    parsed, leftover = extract_json_from_gemini_text(gemini_text)
    if parsed:
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    else:
        return gemini_text.strip()


def human_report_from_gemini(gemini_text: str) -> str:
    parsed, _ = extract_json_from_gemini_text(gemini_text)
    if not parsed:
        return gemini_text.strip()

    root = parsed
    if "STRIDE_Report" in parsed:
        root = parsed["STRIDE_Report"]
    lines = []
    title = root.get("description", "STRIDE Report")
    lines.append(title)
    lines.append("=" * len(title))
    comps = root.get("components", root.get("components", []))

    for comp in comps:
        cid = comp.get("component_id", "N/A")
        ctype = comp.get("component_type", comp.get("type", "Unknown"))
        lines.append(f"\nComponente {cid} — {ctype}")
        lines.append("-" * (20 + len(str(cid))))
        threats = comp.get("threats", [])
        if not threats:
            lines.append("  (nenhuma ameaça listada)")
            continue
        for t in threats:
            ttype = t.get("type", "Unknown")
            summary = t.get("summary", "")
            just = t.get("justification", "")
            sev = t.get("severity", "")
            recs = t.get("recommendations", [])
            refs = t.get("references", [])
            lines.append(f"• Ameaça: {ttype} (Severidade: {sev})")
            if summary:
                lines.append(f"  Resumo: {summary}")
            if just:
                lines.append(f"  Justificativa: {just}")
            if recs:
                lines.append("  Recomendações:")
                for r in recs:
                    lines.append(f"    - {r}")
            if refs:
                lines.append("  Referências: " + ", ".join(refs))
            lines.append("")

    return "\n".join(lines)


def analyze_image_bytes(file_bytes: bytes) -> Tuple[str, dict]:
    print("\n[DEBUG] Starting image analysis...")
    img = read_imagefile(file_bytes)
    h_img, w_img = img.shape[:2]
    print(f"[DEBUG] Image dimensions: {w_img}x{h_img}")

    print("[DEBUG] Running OCR...")
    ocr_texts = detect_texts_and_labels(img)
    print(f"[DEBUG] Found {len(ocr_texts)} text regions")

    print("[DEBUG] Finding candidate boxes...")
    boxes = find_candidate_boxes(img)
    print(f"[DEBUG] Found {len(boxes)} boxes")

    components = []

    for idx, bbox in enumerate(boxes):
        print(f"[DEBUG] Processing box {idx}: {bbox}")
        crop = crop_image(img, bbox)
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        clip_label, clip_score = clip_classify_crop(pil)
        print(f"[DEBUG] CLIP classification: {clip_label} (score: {clip_score:.2f})")

        found_label = None
        for t in ocr_texts:
            tx, ty, tw, th = t['bbox']
            if tx >= bbox[0] and ty >= bbox[1] and (tx+tw) <= (bbox[0]+bbox[2]) and (ty+th) <= (bbox[1]+bbox[3]):
                found_label = t['text']
                break

        if found_label:
            print(f"[DEBUG] OCR label found: {found_label}")

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

    print("[DEBUG] Detecting connections...")
    connections = detect_lines_connections(img, boxes)
    print(f"[DEBUG] Found {len(connections)} connections")

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

    print(f"[DEBUG] Generated {len(findings)} findings")

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

    findings_json = {
        "summary": "Auto-detected components and STRIDE candidate threats (heuristic).",
        "components": components,
        "connections": [{"a": a, "b": b} for (a, b) in connections],
        "findings": findings
    }

    print("[DEBUG] Image analysis completed successfully")
    return local_report_text, findings_json


@app.post("/analyze", response_class=PlainTextResponse)
async def analyze(file: UploadFile = File(...)):
    print("\n" + "="*80)
    print(f"[REQUEST] New analysis request received")
    print(f"[REQUEST] Filename: {file.filename}")
    print(f"[REQUEST] Content-Type: {file.content_type}")
    print("="*80)

    content = await file.read()
    print(f"[DEBUG] File size: {len(content)} bytes")

    if not content:
        print("[ERROR] No file content!")
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        print("[DEBUG] Calling analyze_image_bytes...")
        fallback_text, findings_json = analyze_image_bytes(content)
        print("[DEBUG] Analysis completed, checking Gemini...")

        if GEMINI_API_KEY:
            try:
                print("[DEBUG] Calling Gemini API...")
                raw_resp = call_gemini_polish_raw(findings_json, timeout=240)
                print("[DEBUG] Gemini API responded successfully")

                gemini_text = ""
                candidates = raw_resp.get("candidates") or raw_resp.get("candidates", [])
                if isinstance(candidates, list) and len(candidates) > 0:
                    c = candidates[0]
                    gemini_text = c.get("output") or c.get("content") or json.dumps(c)
                else:
                    gemini_text = json.dumps(raw_resp)

                print(f"[DEBUG] Gemini response length: {len(gemini_text)} chars")

                pretty_json = pretty_json_from_gemini(gemini_text)
                human_text = human_report_from_gemini(gemini_text)

                if human_text and human_text.strip():
                    print("[DEBUG] Returning human-readable report")
                    return PlainTextResponse(content=human_text, media_type="text/plain")
                elif pretty_json:
                    print("[DEBUG] Returning JSON report")
                    return PlainTextResponse(content=pretty_json, media_type="application/json")
                else:
                    print("[DEBUG] Returning fallback report")
                    return PlainTextResponse(content=fallback_text, media_type="text/plain")

            except Exception as e:
                print(f"[ERROR] Gemini error: {str(e)}")
                note = f"\n\n---\nGemini polishing failed or not available; returning local heuristic report. Error: {str(e)}"
                return PlainTextResponse(content=(fallback_text + note), media_type="text/plain")
        else:
            print("[DEBUG] No Gemini API key, returning fallback")
            note = "\n\n---\nNo GEMINI_API_KEY configured. Set GEMINI_API_KEY env var to enable polishing using Gemini."
            return PlainTextResponse(content=(fallback_text + note), media_type="text/plain")

    except Exception as e:
        print(f"[ERROR] Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting STRIDE Threat Analyzer API")
    print("Server will run on: http://0.0.0.0:8000")
    print("CORS enabled for all origins")
    print("="*80 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
