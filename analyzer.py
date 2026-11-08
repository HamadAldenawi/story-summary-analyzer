"""Detect which evaluation-grid criteria a child's (transcribed) oral summary covers.
Method 1 - keyword baseline.  Method 2 - retrieval-augmented semantic matching (RAG-style):
each criterion is enriched with the most relevant sentences of the source story, then compared to
sliding windows of the summary with character n-gram TF-IDF (robust to spelling/ASR errors)."""
import json, re, unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn"))

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def windows(text, size=8, step=4):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + 1), step)] or [text]

def keyword_detect(summary, grid):
    s = norm(summary)
    return {c: any(norm(k) in s for k in v["keys"]) for c, v in grid.items()}

class RagMatcher:
    def __init__(self, story, k=1):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.src = sentences(norm(story["text"]))
        self.vec.fit(self.src + [norm(v["desc"]) for v in story["grid"].values()])
        S = self.vec.transform(self.src)
        self.crit = {}
        for c, v in story["grid"].items():
            q = norm(v["desc"] + " " + " ".join(v["keys"]))
            sims = (self.vec.transform([q]) @ S.T).toarray()[0]
            ctx = [self.src[i] for i in np.argsort(-sims)[:k]]      # retrieved source context
            self.crit[c] = q + " " + " ".join(ctx)

    def scores(self, summary):
        W = self.vec.transform(windows(norm(summary)))
        return {c: float((self.vec.transform([t]) @ W.T).max()) for c, t in self.crit.items()}

def load():
    return json.load(open("data/stories.json")), json.load(open("data/summaries.json"))
