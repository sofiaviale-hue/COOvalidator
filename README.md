# e-COO Verifier (Streamlit Cloud)

Validates Certificates of Origin by comparing the **Reference No.** printed in the PDF with the **Reference No.** encoded in the QR page URL.

**Stack (no system binaries):** Streamlit + PyMuPDF (PDF render) + EasyOCR (OCR) + OpenCV (QR).

## Deploy
1) Push this repo to GitHub.
2) Go to https://share.streamlit.io → New app → pick this repo and `app.py`.
3) Deploy. You’ll get a public URL.

## Notes
- Keep batches ≤ 50 PDFs per run on the free tier.
- Trusted domains default to `ecosys.gov.vn` (editable in the UI).
