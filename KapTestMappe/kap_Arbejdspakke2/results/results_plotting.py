import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
filename = "energies.csv"
n = 2 # Number of bodies (for plot title)
# ---------------------

# Resolve the absolute path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, filename)

# Load the CSV data (assuming order: tspan, PE, KE, TE)
data = np.loadtxt(file_path, delimiter=',')
tspan, PE, KE, TE = data[:, 0], data[:, 1], data[:, 2], data[:, 3]


plt.figure(figsize=(10, 6))

# Plot each component
plt.plot(tspan, KE, label='Kinetic Energy (KE)')
plt.plot(tspan, PE, label='Potential Energy (PE)')
plt.plot(tspan, TE, label='Total Energy (TE)', linestyle='--', color='black')

# Formatting
plt.title(f"Energy of the System with n={n} bodies")
plt.xlabel("Time [s]")
plt.ylabel("Energy [J]")
plt.legend()
plt.grid(True, alpha=0.5)

plt.show()