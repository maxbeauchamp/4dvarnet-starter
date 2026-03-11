import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# -----------------------------
# Configuration
# -----------------------------
ncfile = "/data/users/maxb/PREPROC/preproc_CROSCIM_x50.nc"

variables = ["asip_sic","cimr_SIC","cimr_SIT","cristal_SIT","cristal_SSH"]
ntime = 15
t_input_end = 11  # tfinal-3
record = 0
sample = 0

out = Path("figures")
slice_dir = out / "slices"
slice_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load NetCDF
# -----------------------------
ds = xr.open_dataset(ncfile)

# -----------------------------
# Export PNGs for each variable and time step
# -----------------------------
print("Exporting slices as PNGs...")
for var in variables:
    arr = ds[var].isel(record=record, sample=sample)
    for t in range(ntime):
        field = arr.isel(time=t).values
        plt.figure(figsize=(2,2))
        plt.imshow(field)
        plt.axis("off")
        fname = slice_dir / f"{var}_t{t}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
print("All slices exported.")

# -----------------------------
# Generate TikZ code
# -----------------------------
print("Generating TikZ file...")

tikz = []
tikz.append(r"\documentclass[tikz,border=2mm]{standalone}")
tikz.append(r"\usepackage{graphicx}")
tikz.append(r"\begin{document}")
tikz.append(r"\begin{tikzpicture}["
             r"forecast/.style={draw,dashed,minimum width=2cm,minimum height=2cm},"
             r"nn/.style={draw,fill=green!20,minimum width=3cm,minimum height=1.8cm},"
             r"stackshift/.style={xshift=0.15cm,yshift=0.15cm}]")

xspace = 2.6
yspace = 2.2

# Stacks for each variable and time
for iv, var in enumerate(variables):
    y = -iv * yspace
    var_tex = var.replace("_", r"\_")  # escape underscores
    tikz.append(rf"\node[left] at (-1,{y}) {{{var_tex}}};")

    for t in range(ntime):
        x = t * xspace
        if t <= t_input_end:
            # Stack 3D with small shifts
            tikz.append(rf"""
\node at ({x},{y}) {{\includegraphics[width=2cm]{{slices/{var}_t{t}.png}}}};
\node[stackshift] at ({x},{y}) {{\includegraphics[width=2cm]{{slices/cimr_SIC_t{t}.png}}}};
\node[stackshift] at ({x},{y}) {{\includegraphics[width=2cm]{{slices/cimr_SIT_t{t}.png}}}};
\node[stackshift] at ({x},{y}) {{\includegraphics[width=2cm]{{slices/cristal_SIT_t{t}.png}}}};
\node[stackshift] at ({x},{y}) {{\includegraphics[width=2cm]{{slices/cristal_SSH_t{t}.png}}}};
""")
        else:
            tikz.append(rf"\node[forecast] at ({x},{y}) {{}};")

# Timeline labels
for t in range(ntime):
    x = t * xspace
    tikz.append(rf"\node[below] at ({x},1) {{$t_{t}$}};")

# Neural Network block
nn_x = 11*xspace + 3
nn_y = -2.5
tikz.append(rf"\node[nn] (nn) at ({nn_x},{nn_y}) {{Spatio-temporal\\Neural Network}};")

# Horizontal arrows
tikz.append(rf"\draw[->,thick] ({11*xspace},{nn_y}) -- (nn);")
tikz.append(rf"\draw[->,thick] (nn) -- ({14*xspace+1},{nn_y});")
tikz.append(rf"\node at ({14*xspace+1.5},{nn_y}) {{Forecast}};")

tikz.append(r"\end{tikzpicture}")
tikz.append(r"\end{document}")

# Save TikZ to file
tikz_file = out / "tensor_pipeline_3D.tex"
with open(tikz_file, "w") as f:
    f.write("\n".join(tikz))

print(f"TikZ file written → {tikz_file}")

# -----------------------------
# Compile to PDF
# -----------------------------
print("Compiling TikZ to PDF...")
try:
    subprocess.run(
        ["pdflatex", "-output-directory", str(out), str(tikz_file)],
        check=True
    )
    print("PDF compilation complete.")
except Exception as e:
    print("Error during PDF compilation:", e)