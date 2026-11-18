"""One-shot, LLM-free direction-ablation edit vs controls for Swedish repetition loops.
Usage: python run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --revision <commit> --out results/
NOTE: written without access to a GPU/network; run `--smoke` first (tiny sizes)."""
import argparse, json, os, random, time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from metrics import rep_ratio, is_loop, sv_frac, bootstrap_ci

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--revision", default=None, help="pin a commit hash for reproducibility")
p.add_argument("--out", default="results")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_dev", type=int, default=48)
p.add_argument("--n_test", type=int, default=128)
p.add_argument("--n_ppl", type=int, default=100)
p.add_argument("--max_new", type=int, default=200)
p.add_argument("--n_random", type=int, default=5)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
if a.smoke: a.n_dev, a.n_test, a.n_ppl, a.max_new, a.n_random = 8, 8, 8, 40, 1
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- data: Swedish Wikipedia (CC BY-SA), disjoint dev / test / ppl splits ----------
need = a.n_dev + a.n_test + a.n_ppl
ds = load_dataset("wikimedia/wikipedia", "20231101.sv", split="train", streaming=True).shuffle(seed=a.seed, buffer_size=10000)
paras = []
for ex in ds:
    ps = [x for x in ex["text"].split("\n") if len(x) > 600]
    if ps: paras.append(ps[0])
    if len(paras) >= need: break
prefix = lambda s: " ".join(s.split()[:20])           # 20-word prefix to continue
dev_p = [prefix(s) for s in paras[:a.n_dev]]
test_p = [prefix(s) for s in paras[a.n_dev:a.n_dev + a.n_test]]
ppl_t = paras[a.n_dev + a.n_test:]
json.dump({"dev": dev_p, "test": test_p, "n_ppl": len(ppl_t)}, open(f"{a.out}/prompts.json", "w"), ensure_ascii=False, indent=1)

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.float32).to(dev).eval()
layers = model.model.layers
H = model.config.hidden_size

@torch.no_grad()
def generate(prompts, bs=16, **kw):
    out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id, **kw)
        out += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return out

@torch.no_grad()
def ppl(texts):
    nll = n = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
        loss = model(ids, labels=ids).loss.item()
        nll += loss * (ids.shape[1] - 1); n += ids.shape[1] - 1
    return float(np.exp(nll / n))

def score(prompts, **kw):
    gens = generate(prompts, **kw)
    r = [rep_ratio(g) for g in gens]
    return {"loop_rate": float(np.mean([is_loop(g) for g in gens])), "rep4": float(np.mean(r)),
            "sv_frac": float(np.mean([sv_frac(g) for g in gens])), "_gens": gens, "_loop": [int(is_loop(g)) for g in gens]}

# ---------- edit machinery: project direction d out of a layer's MLP output ----------
orig = {}
def ablate(layer, d, alpha):
    W = layers[layer].mlp.down_proj.weight           # [H, inter]; MLP out = W @ act
    orig.setdefault(layer, W.data.clone())
    W.data -= alpha * torch.outer(d, d @ W.data)      # W' = (I - alpha d d^T) W
def restore():
    for l, W in orig.items(): layers[l].mlp.down_proj.weight.data.copy_(W)
    orig.clear()

# ---------- find candidate direction per layer (dev split only, no LLM, no gradients) ----------
@torch.no_grad()
def find_dirs(dev_gens, dev_prompts):
    caps = {}
    hooks = [l.mlp.register_forward_hook(lambda m, i, o, k=k: caps.__setitem__(k, o[0])) for k, l in enumerate(layers)]
    S = {k: [torch.zeros(H, device=dev), 0, torch.zeros(H, device=dev), 0] for k in range(len(layers))}
    for pr, g in zip(dev_prompts, dev_gens):
        ids = tok(pr + g, return_tensors="pt").input_ids.to(dev)[0]
        npr = len(tok(pr).input_ids)
        seen, lab = set(), []
        for i in range(npr, len(ids)):                # label generated tokens: inside a repeated 4-gram?
            gram = tuple(ids[max(0, i - 3):i + 1].tolist()); lab.append(gram in seen); seen.add(gram)
        lab = torch.tensor(lab, device=dev)
        if lab.sum() == 0 or (~lab).sum() == 0: continue
        model(ids[None])
        for k in caps:
            o = caps[k][npr:]
            S[k][0] += o[lab].sum(0); S[k][1] += int(lab.sum()); S[k][2] += o[~lab].sum(0); S[k][3] += int((~lab).sum())
    for h in hooks: h.remove()
    dirs = {}
    for k, (sl, nl, sn, nn) in S.items():
        if nl and nn:
            diff = sl / nl - sn / nn; dirs[k] = (diff / diff.norm(), float(diff.norm()))
    return dirs

# ---------- run ----------
res, t0 = {"args": vars(a), "torch": torch.__version__}, time.time()
base_dev = score(dev_p); res["dev_base"] = {k: v for k, v in base_dev.items() if k[0] != "_"}
ppl_base = ppl(ppl_t); res["ppl_base"] = ppl_base
dirs = find_dirs(base_dev["_gens"], dev_p)
cand = sorted(dirs, key=lambda k: -dirs[k][1])[:6]     # top-6 layers by contrast magnitude
grid = []
for L in cand:
    for alpha in (0.5, 1.0):
        ablate(L, dirs[L][0], alpha)
        s = score(dev_p); pp = ppl(ppl_t[:30])
        grid.append({"layer": L, "alpha": alpha, "loop_rate": s["loop_rate"], "ppl": pp}); restore()
ppl30_base = ppl(ppl_t[:30])
ok = [g for g in grid if g["ppl"] <= 1.05 * ppl30_base] or grid
best = min(ok, key=lambda g: (g["loop_rate"], g["ppl"]))
res["dev_grid"], res["chosen"] = grid, best
L, alpha = best["layer"], best["alpha"]

def evaluate(name, **kw):
    s = score(test_p, **kw); s["ppl"] = ppl(ppl_t); return name, s
conds = {}
n, s = evaluate("unchanged"); conds[n] = s
n, s = evaluate("repetition_penalty_1.1", repetition_penalty=1.1); conds[n] = s
n, s = evaluate("no_repeat_ngram_4", no_repeat_ngram_size=4); conds[n] = s
ablate(L, dirs[L][0], alpha); n, s = evaluate("targeted_edit"); conds[n] = s; restore()
for r in range(a.n_random):                            # random unit directions, same layer & alpha
    g = torch.Generator(device="cpu").manual_seed(1000 + r)
    d = torch.randn(H, generator=g).to(dev); d /= d.norm()
    ablate(L, d, alpha); n, s = evaluate(f"random_edit_{r}"); conds[n] = s; restore()

base_loop = conds["unchanged"]["_loop"]
res["test"] = {}
for k, v in conds.items():
    m, lo, hi = bootstrap_ci(base_loop, v["_loop"])   # paired: baseline minus condition (positive = fewer loops)
    res["test"][k] = {**{kk: vv for kk, vv in v.items() if kk[0] != "_"}, "loop_reduction_vs_base": [m, lo, hi]}
json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
json.dump({k: v["_gens"][:10] for k, v in conds.items()}, open(f"{a.out}/examples.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res["test"], indent=1)); print("chosen", best, "elapsed", time.time() - t0)
