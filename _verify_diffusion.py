import json
import os

base = r"c:\Users\Gopi.Battineni\OneDrive - University of Limerick\Desktop\SYNTH\SYNTH_BENCHMARK\Diffusion GANs"
issues = []
for f in sorted(os.listdir(base)):
    if not f.endswith(".ipynb"):
        continue
    nb = json.load(open(os.path.join(base, f), encoding="utf-8"))
    text = "".join("".join(c.get("source", [])) for c in nb["cells"])
    for bad in [
        "CTABGAN",
        "WGAN_GP",
        "synthetic_ctabgan",
        "synthetic_wgan_gp",
        "class Generator",
        "ctabgan.fit",
    ]:
        if bad in text:
            issues.append((f, bad, text.count(bad)))
    if "train_tabddpm" not in text:
        issues.append((f, "missing train", 0))
    if 'model_order = ["TabDDPM", "CoDi", "GOGGLE", "ForestDiffusion"]' not in text:
        issues.append((f, "bad model_order", 0))
print("issues", len(issues))
for i in issues:
    print(i)
