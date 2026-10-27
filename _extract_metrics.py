import json
import re

path = r"c:\Users\Gopi.Battineni\OneDrive - University of Limerick\Desktop\SYNTH\SYNTH_BENCHMARK\Other GANS\1. Cancer_other GAN.ipynb"
nb = json.load(open(path, encoding="utf-8"))
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

patterns = {
    "classifiers": re.findall(r'"(Logistic[^"]+|SVM[^"]*|KNN[^"]*|Naive[^"]*|Decision[^"]*|Random[^"]*|Extra[^"]*|Gradient[^"]*|Ada[^"]*|MLP[^"]*)"', text),
    "metrics_utility": re.findall(r"(Accuracy|F1|AUC|Precision|Recall|R2|MSE|RMSE|MAE)[_ ]?(Drop|Score)?", text),
    "evaluate_models": "evaluate_models" in text,
}

# Extract function defs
funcs = re.findall(r"def (\w+)\([^)]*\):", text)
print("FUNCTIONS:", sorted(set(funcs)))
print()

# Key constants
for name in ["METRIC_SAMPLE_SIZE", "METRIC_RANDOM_STATE", "test_size", "model_order", "bins"]:
    m = re.search(rf"{name}\s*=\s*([^\n]+)", text)
    if m:
        print(f"{name} = {m.group(1)}")

# Print utility section snippet
idx = text.find("# 4 Utility")
if idx >= 0:
    print("\n--- UTILITY SNIPPET ---")
    print(text[idx:idx+2500])

idx = text.find("# 8. Privacy")
if idx >= 0:
    print("\n--- PRIVACY SNIPPET ---")
    print(text[idx:idx+2000])

idx = text.find("Bivariate")
if idx >= 0:
    print("\n--- BIVARIATE SNIPPET ---")
    print(text[idx:idx+1500])

idx = text.find("Multivariate")
if idx >= 0:
    print("\n--- MULTIVARIATE SNIPPET ---")
    print(text[idx:idx+1500])
