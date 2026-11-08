"""Leave-one-story-out evaluation: the RAG matcher threshold is tuned on two stories and tested on the third.
Also measures robustness to simulated ASR errors."""
import json, numpy as np
from analyzer import load, keyword_detect, RagMatcher
from asr_noise import add_noise

def prf(pairs):
    tp = sum(p and g for p, g in pairs); fp = sum(p and not g for p, g in pairs); fn = sum(g and not p for p, g in pairs)
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    return round(P, 3), round(R, 3), round(2 * P * R / (P + R) if P + R else 0, 3)

def run(noise=0.0):
    stories, sums = load(); S = {s["id"]: s for s in stories}
    M = {i: RagMatcher(s) for i, s in S.items()}
    kw, rag = [], []
    for test in S:
        train = [x for x in sums if x["story"] != test]
        best = max(np.arange(0.05, 0.6, 0.01), key=lambda t: prf([(M[x["story"]].scores(add_noise(x["text"], noise, j))[c] >= t, c in x["labels"])
                     for j, x in enumerate(train) for c in S[x["story"]]["grid"]])[2])
        for j, x in enumerate(sums):
            if x["story"] != test: continue
            t = add_noise(x["text"], noise, j); g = S[test]["grid"]
            k = keyword_detect(t, g); r = M[test].scores(t)
            kw += [(k[c], c in x["labels"]) for c in g]; rag += [(r[c] >= best, c in x["labels"]) for c in g]
    return {"keyword_baseline": prf(kw), "rag_char_ngram": prf(rag)}

if __name__ == "__main__":
    res = {f"asr_noise_{int(n*100)}%": run(n) for n in [0.0, 0.1, 0.2, 0.3]}
    json.dump(res, open("results.json", "w"), indent=2)
    print("(precision, recall, F1)"); print(json.dumps(res, indent=1))
