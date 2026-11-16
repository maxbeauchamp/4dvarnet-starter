"""(rebuttal) MAX_TIME parameter for the SIC notebooks: wall-clock budget of trainer.fit (Lightning max_time,
'DD:HH:MM:SS'), so that the SIC runs (more and larger training windows) take about as long as the SSH runs.
None = no limit (MAX_EPOCHS only). Idempotent."""
import json, glob, sys
PARAM = "MAX_TIME = None          # (rebuttal) wall-clock budget of training, 'DD:HH:MM:SS' (None: MAX_EPOCHS only)\n"
for f in sys.argv[1:] or sorted(glob.glob("Notebooks/Notebooks_SIC/*.ipynb")):
    nb = json.load(open(f)); changed = False
    for c in nb["cells"]:
        src = "".join(c["source"])
        if src.startswith("# --- parameters (patch_notebook_params.py) ---") and "MAX_TIME" not in src:
            src = src.rstrip("\n") + "\n" + PARAM; changed = True
        if "max_epochs=MAX_EPOCHS," in src and "max_time=MAX_TIME" not in src:
            src = src.replace("max_epochs=MAX_EPOCHS,", "max_epochs=MAX_EPOCHS, max_time=MAX_TIME,"); changed = True
        c["source"] = src.splitlines(keepends=True)
    if changed:
        json.dump(nb, open(f, "w"), indent=1, ensure_ascii=False); open(f, "a").write("\n"); print("patched", f)
