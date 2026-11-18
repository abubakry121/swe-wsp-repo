"""Pure-python metrics (no torch needed)."""
import re

def words(t):
    return re.findall(r"\w+", t.lower())

def rep_ratio(text, n=4):
    """Fraction of word n-grams that are repeats of an earlier n-gram. 0 = no repetition."""
    w = words(text)
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1 - len(set(grams)) / len(grams)

def is_loop(text, thr=0.3):
    return rep_ratio(text) >= thr

SV = set("och i att det som en på är av för med till den har de inte om ett men var jag sig från så kan man när år efter över under också vid eller sin sina hade blev bland mellan".split())
EN = set("the and of to in is that it was for on are as with his they be at one have this from or had by but not what all were when we there can an your which their said if do will each about how up out them then she many some so these would other into has more her two like him see time could no make than first been its who now people my made over did down only way find use may water long little very after words called just where most know".split())

def sv_frac(text):
    """Share of Swedish function words among Swedish+English function words (crude language-ID)."""
    w = words(text)
    s = sum(x in SV for x in w); e = sum(x in EN for x in w)
    return s / (s + e) if (s + e) else 0.0

def bootstrap_ci(xs, ys=None, n=2000, seed=0):
    """95% CI of mean(xs) or paired mean(xs - ys)."""
    import random
    r = random.Random(seed)
    d = list(xs) if ys is None else [a - b for a, b in zip(xs, ys)]
    m = [sum(r.choices(d, k=len(d))) / len(d) for _ in range(n)]
    m.sort()
    return sum(d) / len(d), m[int(.025 * n)], m[int(.975 * n)]
