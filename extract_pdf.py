import sys, os
from pypdf import PdfReader

src = sys.argv[1]
dst = sys.argv[2]
reader = PdfReader(src)
with open(dst, "w", encoding="utf-8") as f:
    f.write(f"PAGES: {len(reader.pages)}\n")
    for i, p in enumerate(reader.pages, 1):
        try:
            t = p.extract_text() or ""
        except Exception as e:
            t = f"[extract error: {e}]"
        f.write(f"\n\n===== PAGE {i} =====\n{t}")
print(len(reader.pages))
