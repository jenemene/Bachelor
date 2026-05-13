import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_all_energies(filename):
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
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.show()

def plot_TE(filename):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data = np.loadtxt(file_path, delimiter=',')
    tspan, PE, KE, TE = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    TE_Delta = TE - TE[0] # Normalize total energy to start at 0 for easier comparison of drift over time

    plt.figure(figsize=(10, 6))

    plt.plot(tspan, TE_Delta, color='black')

    # Formatting
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta$ Total Energy [J]")
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

def compare_TE(filename1,filename2):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(script_dir, filename1)
    file_path2 = os.path.join(script_dir, filename2)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data1 = np.loadtxt(file_path1, delimiter=',')
    tspan1, TE1 = data1[:, 0], data1[:, 3]
    dt1 = tspan1[1] - tspan1[0]

    data2 = np.loadtxt(file_path2, delimiter=',')
    tspan2, TE2 = data2[:, 0], data2[:, 3]
    dt2 = tspan2[1] - tspan2[0]

    TE_Delta1 = TE1 - TE1[0] # Normalize total energy to start at 0
    TE_Delta2 = TE2 - TE2[0]

    plt.figure(figsize=(10, 6))

    plt.plot(tspan1, TE_Delta1, color='black', label=fr'$\Delta t = {dt1:.2e}$')
    plt.plot(tspan2, TE_Delta2, color='red', linestyle='--', label=fr'$\Delta t = {dt2:.2e}$')

    # Formatting
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\Delta$ Total Energy [J]")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.show()

### CONFIG ###
filename1 = "energies.csv"
filename2 = "energies_half_ts.csv"

plot_all_energies(filename1)

plot_TE(filename2)

compare_TE(filename1,filename2)