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

def adams_comp_pend(filename, adams_file_path=None, plot_diff=False):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

    # Determine number of bodies dynamically (1 column for time, 3 per body)
    n_bodies = (data.shape[1] - 1) // 3

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 4 * n_bodies), sharex=True)
    if n_bodies == 1:
        axes = [axes]
        
    # Load Adams Data if provided
    adams_data = []
    if adams_file_path:
        if os.path.exists(adams_file_path):
            blocks = []
            current_block = []
            with open(adams_file_path, 'r') as f:
                for line in f:
                    clean_line = line.strip().replace(';', ' ')
                    if not clean_line:
                        continue
                    
                    if clean_line[0].isalpha():
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        continue
                        
                    try:
                        # Attempt 1: Space-separated, with commas replaced by dots (handles Euro decimals)
                        parts = [float(x) for x in clean_line.replace(',', '.').split()]
                        if parts:
                            current_block.append(parts)
                    except ValueError:
                        try:
                            # Attempt 2: Comma-separated (Standard CSV)
                            parts = [float(x) for x in clean_line.split(',')]
                            if parts:
                                current_block.append(parts)
                        except ValueError:
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
            if current_block:
                blocks.append(current_block)
            if blocks:
                min_rows = min(len(b) for b in blocks)
                truncated_blocks = [np.array(b)[:min_rows] for b in blocks]
                adams_data = np.hstack(truncated_blocks)
        else:
            print(f"Warning: Adams file not found at {adams_file_path}")

    axes_labels = ['x', 'y', 'z']

    if len(adams_data) > 0:
        t_adams = adams_data[:, 0]
        if len(tspan) != len(t_adams):
            raise ValueError(f"Timestep lengths do not match: SOA ({len(tspan)}) vs Adams ({len(t_adams)})")
        if not np.allclose(tspan, t_adams, atol=1e-5):
            raise ValueError("Timestep values do not match between SOA and Adams data.")

    for b in range(n_bodies):
        ax = axes[b]
        
        # Plot SOA data
        active_axis = 'unknown'
        max_acc = 1e-10
        soa_acc = None
        for i, axis in enumerate(axes_labels):
            col_idx = 1 + 3 * b + i
            if col_idx < data.shape[1] and np.any(np.abs(data[:, col_idx]) > 1e-10):  # Check for non-zero entries (with a small tolerance)
                ax.plot(tspan, data[:, col_idx], label=f'SOA: Body {b+1} Ang Acc {axis}', linewidth=2)
                
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    active_axis = axis
                    soa_acc = data[:, col_idx]

        # Plot Adams data
        if len(adams_data) > 0:
            t_adams = adams_data[:, 0]
            col_idx = b + 1
            if col_idx < adams_data.shape[1]:
                if np.any(np.abs(adams_data[:, col_idx]) > 1e-10):
                    adams_acc = -adams_data[:, col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                    label_suffix = f" {active_axis}" if active_axis != 'unknown' else ""
                    ax.plot(t_adams, adams_acc, label=f'Adams: Body {b+1} Ang Acc{label_suffix}', linestyle='--', linewidth=2)
                    
                    if plot_diff and soa_acc is not None:
                        delta = soa_acc - adams_acc
                        ax.plot(tspan, delta, label=r'$\Delta$ (SOA - Adams)', color='red', linestyle='--', linewidth=1.5)

        ax.set_ylabel(r'Ang Acc [rad/s$^2$]')
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()

def adams_delta_pend(filename, adams_file_path=None):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

    # Determine number of bodies dynamically (1 column for time, 3 per body)
    n_bodies = (data.shape[1] - 1) // 3

    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 4 * n_bodies), sharex=True)
    if n_bodies == 1:
        axes = [axes]
        
    # Load Adams Data if provided
    adams_data = []
    if adams_file_path:
        if os.path.exists(adams_file_path):
            blocks = []
            current_block = []
            with open(adams_file_path, 'r') as f:
                for line in f:
                    clean_line = line.strip().replace(';', ' ')
                    if not clean_line:
                        continue
                    
                    if clean_line[0].isalpha():
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        continue
                        
                    try:
                        # Attempt 1: Space-separated, with commas replaced by dots (handles Euro decimals)
                        parts = [float(x) for x in clean_line.replace(',', '.').split()]
                        if parts:
                            current_block.append(parts)
                    except ValueError:
                        try:
                            # Attempt 2: Comma-separated (Standard CSV)
                            parts = [float(x) for x in clean_line.split(',')]
                            if parts:
                                current_block.append(parts)
                        except ValueError:
                            if current_block:
                                blocks.append(current_block)
                                current_block = []
            if current_block:
                blocks.append(current_block)
            if blocks:
                min_rows = min(len(b) for b in blocks)
                truncated_blocks = [np.array(b)[:min_rows] for b in blocks]
                adams_data = np.hstack(truncated_blocks)
        else:
            print(f"Warning: Adams file not found at {adams_file_path}")

    axes_labels = ['x', 'y', 'z']

    if len(adams_data) > 0:
        t_adams = adams_data[:, 0]
        if len(tspan) != len(t_adams):
            raise ValueError(f"Timestep lengths do not match: SOA ({len(tspan)}) vs Adams ({len(t_adams)})")
        if not np.allclose(tspan, t_adams, atol=1e-5):
            raise ValueError("Timestep values do not match between SOA and Adams data.")

    for b in range(n_bodies):
        ax = axes[b]
        
        soa_acc = None
        active_axis = 'unknown'
        max_acc = 1e-10
        
        # Find the active SOA axis for this body
        for i, axis in enumerate(axes_labels):
            col_idx = 1 + 3 * b + i
            if col_idx < data.shape[1]:
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    soa_acc = data[:, col_idx]
                    active_axis = axis

        # Plot delta if we have both data sources
        if soa_acc is not None and len(adams_data) > 0:
            adams_col_idx = b + 1
            if adams_col_idx < adams_data.shape[1]:
                if np.any(np.abs(adams_data[:, adams_col_idx]) > 1e-10):
                    adams_acc = -adams_data[:, adams_col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                    
                    # Compute error delta directly since we checked shape/values above
                    delta = soa_acc - adams_acc
                    
                    label_suffix = f" {active_axis}" if active_axis != 'unknown' else ""
                    ax.plot(tspan, delta, label=f'SOA - Adams: Body {b+1} Ang Acc{label_suffix}', color='red', linewidth=1.5)

        ax.set_ylabel(r'$\Delta$ Ang Acc [rad/s$^2$]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.5)

    axes[-1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    plt.show()

### CONFIG ###
filename1 = "wall_contact_energies.csv"
filename2 = "dp_gen_acc.csv"

# Resolve the absolute path relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "TE_delta.csv")

# Load the CSV data (assuming order: tspan, PE, KE, TE)
data = np.loadtxt(file_path, delimiter=',')
tspan, TE = data[:, 0], data[:, 1]

plt.plot(tspan,TE)

plot_all_energies(filename1)

plot_TE(filename1)

