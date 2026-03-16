import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess

# -----------------------------
# Configuration
# -----------------------------
ncfile = "/data/users/maxb/PREPROC/preproc_CROSCIM_x50.nc"

variables = [
    "asip_sic",
    "cimr_SIC",
    "cimr_SIT",
    "cristal_SIT",
    "cristal_SSH"
]

ntime = 15
t_input_end = 11   # tfinal - 3
record = 0
sample = 0

outdir = Path("figures")
slicedir = outdir / "slices"

outdir.mkdir(exist_ok=True)
slicedir.mkdir(exist_ok=True)

# -----------------------------
# Load NetCDF
# -----------------------------
print("Loading NetCDF...")
ds = xr.open_dataset(ncfile)

# -----------------------------
# Export PNG slices
# -----------------------------
print("Exporting images...")

for var in variables:

    arr = ds[var].isel(record=record, sample=sample)

    for t in range(ntime):

        field = arr.isel(time=t).values

        plt.figure(figsize=(2,2))
        plt.imshow(field, cmap="viridis")
        plt.axis("off")

        fname = slicedir / f"{var}_t{t}.png"

        plt.savefig(
            fname,
            dpi=200,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white"
        )

        plt.close()

print("Images exported.")


# -----------------------------
# Generate TikZ
# -----------------------------
print("Generating TikZ...")

tikz = []

tikz.append(r"\documentclass[tikz,border=3mm]{standalone}")
tikz.append(r"\usepackage{graphicx}")
tikz.append(r"\graphicspath{{slices/}}")

tikz.append(r"\begin{document}")

tikz.append(r"""
\begin{tikzpicture}[
forecast/.style={draw,dashed,minimum width=2cm,minimum height=2cm},
nn/.style={draw,fill=green!20,minimum width=3.5cm,minimum height=2cm}
]
""")

xspace = 2.8
yspace = 2.3

stack_shift = 0.18

# -----------------------------
# Draw stacks
# -----------------------------
for iv, var in enumerate(variables):

    y = -iv * yspace
    var_tex = var.replace("_", r"\_")

    tikz.append(
        rf"\node[left] at (-1.5,{y}) {{{var_tex}}};"
    )

    for t in range(ntime):

        x = t * xspace

        if t <= t_input_end:

            # 3D stack using all variables
            for k, svar in enumerate(variables):

                shift = k * stack_shift

                tikz.append(
rf"""
\node at ({x+shift},{y+shift})
{{\includegraphics[width=2cm]{{{svar}_t{t}.png}}}};
"""
                )

        else:

            tikz.append(
                rf"\node[forecast] at ({x},{y}) {{}};"
            )

# -----------------------------
# Timeline labels
# -----------------------------
for t in range(ntime):

    x = t * xspace

    tikz.append(
        rf"\node[below] at ({x},1) {{$t_{{{t}}}$}};"
    )

# -----------------------------
# Neural Network block
# -----------------------------
nn_x = (t_input_end + 1.5) * xspace
nn_y = -2.5

tikz.append(
rf"\node[nn] (nn) at ({nn_x},{nn_y}) {{Spatio-temporal\\Neural Network}};"
)

# Arrow from inputs to NN
tikz.append(
rf"\draw[->,thick] ({t_input_end*xspace},{nn_y}) -- (nn);"
)

# Arrow from NN to forecast
tikz.append(
rf"\draw[->,thick] (nn) -- ({(ntime-1)*xspace+1},{nn_y});"
)

tikz.append(
rf"\node at ({(ntime-1)*xspace+1.5},{nn_y}) {{Forecast}};"
)

tikz.append(r"\end{tikzpicture}")
tikz.append(r"\end{document}")

# -----------------------------
# Save tex
# -----------------------------
texfile = outdir / "spatiotemporal_pipeline.tex"

with open(texfile, "w") as f:
    f.write("\n".join(tikz))

print("TikZ file written:", texfile)


# -----------------------------
# Compile PDF
# -----------------------------
print("Compiling PDF...")

subprocess.run(
    [
        "pdflatex",
        "-interaction=nonstopmode",
        "-output-directory",
        str(outdir),
        str(texfile)
    ]
)

print("Done.")
print("Output PDF:", outdir / "spatiotemporal_pipeline.pdf")