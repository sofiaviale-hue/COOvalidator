import io, re, json, zipfile, time
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import cv2
import easyocr

# ---------- Config ----------
TRUSTED_DOMAINS = {"ecosys.gov.vn"}
URL_PARAM_KEYS  = {
    "ecosys.gov.vn": ["CertificateNumber","certificateNumber","ReferenceNo","RefNo","CertNo"],
    "_default":      ["CertificateNumber","ReferenceNo","RefNo","CertNo","No","no"]
}
PDF_RENDER_DPI = 300
MAX_FILES_PER_RUN = 50
# ----------------------------

def normalize_ref(s: str) -> str:
    s = (s or "").upper().replace("—","-").replace("–","-")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z0-9/\-]", "", s)
    return s

@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

def render_pdf_first_page(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def ocr_reference_no(img: Image.Image) -> str:
    reader = get_ocr_reader()
    text = " ".join(reader.readtext(np.array(img), detail=0, paragraph=True)).upper()
    text = text.replace("REFERENCE N0", "REFERENCE NO")  # 0/O fix
    m = re.search(r"REFERENCE\s*NO\.?\s*[:\-]?\s*([A-Z0-9/\-]{6,})", text)
    if m: return normalize_ref(m.group(1))
    m = re.search(r"VN[-/]?CN[0-9/]{6,}", text)
    return normalize_ref(m.group(0)) if m else ""

def decode_qr_from_image(img: Image.Image) -> str | None:
    det = cv2.QRCodeDetector()
    for angle in (0, 90, 180, 270):
        rotated = img.rotate(angle, expand=True) if angle else img
        data, _ = det.detectAndDecode(cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR))
        if data:
            return data.strip()
    return None

def extract_qr_url(pdf_bytes: bytes) -> str | None:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_to_try = [0] + ([1] if doc.page_count > 1 else [])
    url = None
    for i in pages_to_try:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)  # ~216 DPI
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        url = decode_qr_from_image(img)
        if url: break
    doc.close()
    return url

def extract_ref_from_url(url: str) -> str:
    if not url: return ""
    up = urlparse(url); host = up.netloc.lower()
    keys = URL_PARAM_KEYS.get(host, URL_PARAM_KEYS["_default"])
    q = parse_qs(up.query)
    for k in keys:
        if q.get(k): return normalize_ref(unquote(q[k][0]))
    m = re.search(r"VN[-/]?CN[0-9/]{6,}", unquote(url).upper())
    return normalize_ref(m.group(0)) if m else ""

def is_trusted(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return any(host.endswith(d) for d in TRUSTED_DOMAINS)
    except Exception:
        return False

def check_coo(file_name: str, pdf_bytes: bytes) -> dict:
    page_img = render_pdf_first_page(pdf_bytes)
    pdf_ref = ocr_reference_no(page_img)
    qr_url  = extract_qr_url(pdf_bytes)
    url_ref = extract_ref_from_url(qr_url) if qr_url else ""

    if not pdf_ref: status, reason = "Error", "No Reference No in PDF"
    elif not qr_url: status, reason = "Error", "No QR found"
    elif not url_ref: status, reason = "Error", "No Reference No in QR URL"
    elif not is_trusted(qr_url): status, reason = "FraudSuspected", "Untrusted domain"
    else:
        status = "Valid" if pdf_ref == url_ref else "FraudSuspected"
        reason = "Reference No match" if status=="Valid" else "Reference mismatch"

    evidence = {
        "file": file_name,
        "pdf_reference": pdf_ref or None,
        "qr_url": qr_url,
        "url_reference": url_ref or None,
        "trusted_domain": is_trusted(qr_url) if qr_url else False,
    }
    return {
        "file": file_name, "status": status, "reason": reason,
        "PDF_reference": pdf_ref or None, "URL_reference": url_ref or None,
        "evidence": evidence
    }

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="e-COO Verifier", layout="wide")
st.title("e-COO Verifier (Streamlit Cloud)")

st.write("Upload PDF certificates. We compare the **Reference No.** printed in the PDF with the **Reference No.** encoded in the QR page URL.")
uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
colA, colB = st.columns([1,3])
with colA:
    trusted_show = st.multiselect("Trusted domains", options=list(TRUSTED_DOMAINS), default=list(TRUSTED_DOMAINS))
with colB:
    st.caption("Tip: keep batches ≤ 50 PDFs per run on the free tier.")

if st.button("Run validation", type="primary", disabled=not uploaded):
    start = time.time()
    if not uploaded:
        st.warning("Please upload at least one PDF."); st.stop()
    global TRUSTED_DOMAINS
    TRUSTED_DOMAINS = set(trusted_show)

    files = uploaded[:MAX_FILES_PER_RUN]
    rows = [check_coo(f.name, f.read()) for f in files]
    df = pd.DataFrame(rows)

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv_bytes, file_name="verification_register.csv", mime="text/csv")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rows:
            j = json.dumps(r["evidence"], ensure_ascii=False, indent=2).encode("utf-8")
            z.writestr(f"evidence/{Path(r['file']).stem}_evidence.json", j)
    buf.seek(0)
    st.download_button("Download Evidence ZIP", data=buf, file_name="evidence_pack.zip", mime="application/zip")

    st.caption(f"Processed {len(files)} file(s) in {time.time()-start:.1f}s.")
