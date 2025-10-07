import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Find the valsPerLoop file (assumes only one matching file in ./Data/)
data_dir = "./Data/"
file_pattern = os.path.join(data_dir, "valsPerLoop*.txt")
files = glob.glob(file_pattern)
if not files:
    raise FileNotFoundError("No valsPerLoop*.txt file found in ./Data/")
vals_file = files[0]

# Read the file, skipping the header
with open(vals_file, "r") as f:
    lines = f.readlines()

header = lines[0].strip().split()
try:
    ndw_col = header.index("NDW")
except ValueError:
    raise ValueError("NDW column not found in header.")

ndw_vals = []
timesteps = []
for i, line in enumerate(lines[1:]):
    if not line.strip():
        continue
    cols = line.strip().split()
    ndw = float(cols[ndw_col])
    timestep = i  # Each row is its own timestep, starting from 0
    timesteps.append(timestep)
    ndw_vals.append(ndw)

plt.figure(figsize=(8,5))
plt.plot(timesteps, ndw_vals, marker='o', linestyle='-', color='royalblue', markersize=4, linewidth=2, label='NDW')

# Add -1 gradient reference line
x_ref = np.array([timesteps[1], timesteps[-1]])
y_ref = ndw_vals[1] * (x_ref / x_ref[0])**-1
plt.plot(x_ref, y_ref, 'k--', linewidth=2, label='Gradient = -1')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Timestep", fontsize=14)
plt.ylabel("Number of Domain Walls", fontsize=14)
plt.title("Domain Wall Number Evolution (log-log)", fontsize=16)
plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("./Plots/number_DW_evolution.png", dpi=200)