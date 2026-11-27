"""Technique 1 – Prompt optimisation.

Rule-based compression: strips politeness/filler, rewrites wordy phrases,
removes duplicated sentences. Returns a word-level diff for the UI.
"""
import difflib
import re

# (pattern, replacement) – applied in order, case-insensitive
REWRITES = [
    (r"\bi (?:was|am) wondering (?:if|whether) (?:you )?(?:could|can|would)\b", ""),
    (r"\b(?:could|can|would|will) you (?:please |kindly )*(?:help me (?:to )?)?", ""),
    (r"\bi (?:would|'d) like (?:you )?to\b", ""),
    (r"\bi want (?:you )?to\b", ""),
    (r"\bi need you to\b", ""),
    (r"\b(?:hi|hello|hey|dear (?:assistant|ai))\b[ ,!.]*", ""),
    (r"\bthank(?:s| you)(?: (?:so|very) much)?(?: in advance)?[ !.]*", ""),
    (r"\bplease\b,?", ""),
    (r"\bkindly\b,?", ""),
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bas well as\b", "and"),
    (r"\bwith regard(?:s)? to\b", "about"),
    (r"\ba (?:large|great|good) number of\b", "many"),
    (r"\bit is important to note that\b", ""),
    (r"\b(?:basically|actually|really|very|just|simply|quite|literally|definitely)\b", ""),
    (r"\bso much\b", ""),
    (r"\bgive me a (?:detailed|long|thorough) (?:explanation|answer) (?:of|about|on)\b", "explain"),
]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", text)
    text = re.sub(r"^[\s,;:.!-]+", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = text.strip()
    # re-capitalise sentence starts
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def _dedupe_sentences(text: str) -> str:
    seen, out = set(), []
    for s in re.split(r"(?<=[.!?])\s+", text):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(s)
    return " ".join(out)


def optimize_prompt(text: str) -> str:
    out = text
    for pat, rep in REWRITES:
        out = re.sub(pat, rep, out, flags=re.I)
    out = _clean(out)
    out = _dedupe_sentences(out)
    return out or text.strip()


_TOK = re.compile(r"\s+|\w+|[^\w\s]")


def diff_segments(before: str, after: str):
    """Word-level diff -> [{"t": text, "op": "keep" | "del"}]."""
    a, b = _TOK.findall(before), _TOK.findall(after)
    sm = difflib.SequenceMatcher(a=[x.lower() for x in a], b=[x.lower() for x in b], autojunk=False)
    segs = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "equal":
            segs.append({"t": "".join(a[i1:i2]), "op": "keep"})
        elif tag in ("delete", "replace"):
            segs.append({"t": "".join(a[i1:i2]), "op": "del"})
    return segs
