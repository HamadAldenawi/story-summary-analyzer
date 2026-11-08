"""Simulate speech-recognition errors on a transcript: word deletions and character substitutions."""
import random
def add_noise(text, rate, seed=0):
    rnd, out = random.Random(seed), []
    for w in text.split():
        r = rnd.random()
        if r < rate / 3: continue                                   # deleted word
        if r < rate and len(w) > 3:                                  # misrecognised word
            i = rnd.randrange(len(w)); w = w[:i] + rnd.choice("aeioumnrlst") + w[i + 1:]
        out.append(w)
    return " ".join(out)
