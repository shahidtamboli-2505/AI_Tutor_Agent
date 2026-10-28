"""Stage 2: read a document, check it has real text, cut it into numbered passages."""
import re
from typing import Dict, List

try:
    import pymupdf
except ImportError:  # older PyMuPDF
    import fitz as pymupdf

import config
from backend.schemas import Span


class DocumentError(Exception):
    """Friendly message that is safe to show to the user."""


SAMPLE_TEXT = """How Plants Make Food: Photosynthesis

Plants are called producers because they make their own food. They do this with a process called photosynthesis. In photosynthesis, a plant uses light energy to turn water and carbon dioxide into glucose, a type of sugar, and oxygen. The glucose gives the plant energy to live and grow, and the oxygen is released into the air.

Light and chlorophyll

Photosynthesis happens mostly in the leaves, inside tiny structures called chloroplasts. Chloroplasts contain a green pigment called chlorophyll. Chlorophyll absorbs light, mainly red and blue light, and reflects green light, which is why leaves look green. The absorbed light energy is what powers the whole process. Without chlorophyll, a plant could not capture the energy it needs.

Raw materials: water and carbon dioxide

A plant needs two raw materials for photosynthesis. Water is taken up from the soil by the roots and carried to the leaves through tubes called xylem. Carbon dioxide enters the leaf from the air through tiny holes called stomata, which are mostly found on the underside of the leaf. Each stoma can open and close, which helps the plant control how much water it loses.

The two stages of photosynthesis

Photosynthesis has two connected stages. In the first stage, called the light-dependent reactions, chlorophyll absorbs light and the energy is used to split water molecules. This releases oxygen and stores energy in small carrier molecules. In the second stage, called the Calvin cycle, the stored energy is used to turn carbon dioxide into glucose. The Calvin cycle does not use light directly, but it depends on the energy carriers made in the first stage.

What changes the speed of photosynthesis

Several factors change how fast photosynthesis happens. More light makes the process faster until the plant cannot use any more light. Higher carbon dioxide levels also speed it up, up to a limit. Temperature matters too: the reactions are controlled by enzymes, which work best at a moderate temperature and slow down or stop if it becomes too hot. The factor that is in shortest supply is called the limiting factor, because it limits how fast the whole process can go.

What the plant does with glucose

The glucose made in photosynthesis has several uses. Some is used right away in respiration to release energy. Some is changed into starch for storage, and some is used to build cellulose for cell walls. Because plants store extra glucose as starch, we can test a leaf for starch to show that photosynthesis has been happening.
"""


def read_document(file_bytes: bytes, filename: str) -> List[Dict]:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            pages = [{"page": i + 1, "text": p.get_text()} for i, p in enumerate(doc)]
        except Exception:
            raise DocumentError("I could not open this PDF. It may be damaged or password protected.")
    elif name.endswith(".txt"):
        pages = [{"page": 1, "text": file_bytes.decode("utf-8", errors="replace")}]
    else:
        raise DocumentError("Please upload a PDF or a .txt file.")
    if not pages:
        raise DocumentError("This file has no pages.")
    return pages


def check_quality(pages: List[Dict]) -> None:
    total = sum(len(p["text"].strip()) for p in pages)
    if total < config.MIN_TOTAL_CHARS or total / max(len(pages), 1) < config.MIN_AVG_CHARS:
        raise DocumentError(
            "I could not find readable text in this file. It might be a scanned PDF (pictures of pages). "
            "Please try a PDF where you can select the text, or a .txt file.")


def _split_long(text: str, max_len: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        while len(s) > max_len:                       # no sentence end found: hard cut
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:max_len])
            s = s[max_len:]
        if cur and len(cur) + len(s) + 1 > max_len:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def make_spans(pages: List[Dict], target: int = config.SPAN_TARGET, max_len: int = config.SPAN_MAX) -> List[Span]:
    spans: List[Span] = []

    def add(page: int, text: str) -> None:
        if text.strip():
            spans.append(Span(id=f"S{len(spans) + 1}", page=page, text=text.strip()))

    for p in pages:
        paras = [re.sub(r"\s*\n\s*", " ", x).strip() for x in re.split(r"\n\s*\n", p["text"])]
        buf = ""
        for para in filter(None, paras):
            pieces = _split_long(para, max_len) if len(para) > max_len else [para]
            for piece in pieces:
                if buf and len(buf) + len(piece) + 2 > target:
                    add(p["page"], buf)
                    buf = piece
                else:
                    buf = f"{buf}\n\n{piece}".strip()
        add(p["page"], buf)
    return spans
