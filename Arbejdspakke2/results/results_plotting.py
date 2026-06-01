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
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Energy [J]", fontsize=14)
    plt.legend(loc='right', fontsize=14, frameon=True, framealpha=0.9, labelspacing=1.2)
    plt.grid(True, alpha=0.5)

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def plot_TE_error(filename):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, PE, KE, TE)
    data = np.loadtxt(file_path, delimiter=',')
    tspan, TE_error = data[:, 0], data[:, 1]

    plt.figure(figsize=(10, 6))

    plt.plot(tspan, TE_error, color='black')

    # Formatting
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel(r"Relative Total Energy [J]", fontsize=14)
    plt.grid(True, alpha=0.5)

    # Force scientific notation for the y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.show()

def compare_TE_error(*filenames):
    """
    Compares Total Energy error across an arbitrary number of CSV files
    using a logarithmic y-axis and hardcoded legend labels.
    
    Parameters:
    *filenames (str): Variable number of file names/paths relative to the script.
    """
    if not filenames:
        print("No files provided for comparison.")
        return

    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.tab10.colors  
    line_styles = ['-', '--', '-.', ':']

    # Dictionary to manually map specific filenames to their respective labels
    # Update or add keys here if your filenames change
    hardcoded_labels = {
        '3_closed_TE_error_BG_50_500.csv': r'$A=50, B=500$',
        '3_closed_TE_error_BG_100_100.csv': r'$A=100, B=100$',
        '3_closed_TE_error_BG_10_50.csv': r'$A=10, B=50$',
        '3_closed_TE_error_BG_1_500.csv': r'$A=1, B=500$',
        '3_closed_TE_error_BG_01_500.csv': r'$A=0.1, B=500$'
    }

    for i, filename in enumerate(filenames):
        file_path = os.path.join(script_dir, filename)
        
        try:
            # Load the CSV data
            data = np.loadtxt(file_path, delimiter=',')
            tspan, TE_error = data[:, 0], data[:, 1]
            
            # Use the hardcoded label if available, otherwise default to the filename
            label_name = hardcoded_labels.get(filename, os.path.splitext(filename)[0])
            
            color = colors[i % len(colors)]
            style = line_styles[(i // len(colors)) % len(line_styles)]
            
            plt.plot(
                tspan, 
                TE_error, 
                color=color, 
                linestyle=style, 
                label=label_name
            )
            
        except Exception as e:
            print(f"Skipping file '{filename}' due to error: {e}")

    # Formatting
    plt.xlabel("Time [s]", fontsize=14)
    plt.ylabel("Total Energy error [J]", fontsize=14)
    
    # Enable logarithmic scale for y-axis
    plt.yscale('log')
    
    plt.legend(loc='lower right', fontsize=12, frameon=True, framealpha=0.9, labelspacing=1.0)
    plt.grid(True, which="both", alpha=0.3) 

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()

def adams_comp(filename, adams_file_path, plot_diff=False, savefig=False):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

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

    # Determine number of bodies dynamically (from Adams if available, else assume 3 DOFs per body)
    if len(adams_data) > 0:
        n_bodies = adams_data.shape[1] - 1
        dofs_per_body = (data.shape[1] - 1) // n_bodies
    else:
        dofs_per_body = 3
        n_bodies = (data.shape[1] - 1) // dofs_per_body

    # Added a bit extra vertical space (+ 0.6) to accommodate bottom legend
    fig, axes = plt.subplots(n_bodies, 1, figsize=(8, 2 * n_bodies + 0.6), layout="constrained")
    if n_bodies == 1:
        axes = [axes]

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
        for i in range(dofs_per_body):
            col_idx = 1 + dofs_per_body * b + i
            axis_label = axes_labels[i] if dofs_per_body == 3 else ''
            if col_idx < data.shape[1] and np.any(np.abs(data[:, col_idx]) > 1e-10):  # Check for non-zero entries (with a small tolerance)
                ax.plot(tspan, data[:, col_idx], label=fr'SOA', linewidth=2)
                
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    active_axis = axis_label
                    soa_acc = data[:, col_idx]

        # Plot Adams data
        if len(adams_data) > 0:
            t_adams = adams_data[:, 0]
            col_idx = b + 1
            if col_idx < adams_data.shape[1]:
                if np.any(np.abs(adams_data[:, col_idx]) > 1e-10):
                    adams_acc = -adams_data[:, col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                    ax.plot(t_adams, adams_acc, label=fr'Adams', linestyle='--', linewidth=2)
                    
                    if plot_diff and soa_acc is not None:
                        delta = soa_acc - adams_acc
                        ax.plot(tspan, delta, label=r'Error (SOA - Adams)', color='red', linestyle='--', linewidth=1.5)

        ax.set_ylabel(f'Body {b+1} $\\dot{{\\beta}}_y$\n[rad/s$^2$]', fontsize=14)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12) # Increases the size of the tick numbers

    axes[-1].set_xlabel('Time [s]', fontsize=14, labelpad=10)
    
    fig.align_ylabels(axes)

    # --- GLOBAL LEGEND GENERATION ---
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Filter out identical label duplicates (e.g. unique SOA and Adams keys)
    by_label = dict(zip(labels, handles))

    # Place unique legend at the bottom center of the whole figure window
    fig.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='outside lower center', 
        ncol=len(by_label), 
        fontsize=12, 
        frameon=True, 
        framealpha=0.9
    )

    if savefig == True:
        plt.savefig("adams_comp.pdf")
        print("Figure saved as adams_comp.pdf in current directory.")

    plt.show()

def adams_comp_error_only(filename, adams_file_path, savefig=False):
    # Resolve the absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Load the CSV data (assuming order: tspan, A)
    data = np.loadtxt(file_path, delimiter=',')
    tspan = data[:, 0]

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

    # Determine number of bodies dynamically (from Adams if available, else assume 3 DOFs per body)
    if len(adams_data) > 0:
        n_bodies = adams_data.shape[1] - 1
        dofs_per_body = (data.shape[1] - 1) // n_bodies
    else:
        dofs_per_body = 3
        n_bodies = (data.shape[1] - 1) // dofs_per_body

    # Added a bit extra vertical space (+ 0.6) to accommodate bottom legend
    fig, axes = plt.subplots(n_bodies, 1, figsize=(8, 2 * n_bodies + 0.6), layout="constrained")
    if n_bodies == 1:
        axes = [axes]

    if len(adams_data) > 0:
        t_adams = adams_data[:, 0]
        if len(tspan) != len(t_adams):
            raise ValueError(f"Timestep lengths do not match: SOA ({len(tspan)}) vs Adams ({len(t_adams)})")
        if not np.allclose(tspan, t_adams, atol=1e-5):
            raise ValueError("Timestep values do not match between SOA and Adams data.")

    for b in range(n_bodies):
        ax = axes[b]
        
        # Find the active SOA data (axis with the largest acceleration)
        max_acc = 1e-10
        soa_acc = None
        for i in range(dofs_per_body):
            col_idx = 1 + dofs_per_body * b + i
            if col_idx < data.shape[1] and np.any(np.abs(data[:, col_idx]) > 1e-10):  # Check for non-zero entries (with a small tolerance)
                
                max_val = np.max(np.abs(data[:, col_idx]))
                if max_val > max_acc:
                    max_acc = max_val
                    soa_acc = data[:, col_idx]

        # Plot difference with Adams data
        if len(adams_data) > 0:
            col_idx = b + 1
            if col_idx < adams_data.shape[1] and np.any(np.abs(adams_data[:, col_idx]) > 1e-10):
                adams_acc = -adams_data[:, col_idx] * (np.pi / 180.0) # Convert deg/s^2 to rad/s^2
                if soa_acc is not None:
                    delta = soa_acc - adams_acc
                    ax.plot(tspan, delta, label=r'Error (SOA - Adams)', color='red', linestyle='--', linewidth=1.5)
        
        ax.set_ylabel(f'Body {b+1} $\\dot{{\\beta}}_y$\n[rad/s$^2$]', fontsize=14)
        ax.grid(True, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12) # Increases the size of the tick numbers

    axes[-1].set_xlabel('Time [s]', fontsize=14, labelpad=10)
    
    fig.align_ylabels(axes)

    # --- GLOBAL LEGEND GENERATION ---
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Filter out identical label duplicates (e.g. unique SOA and Adams keys)
    by_label = dict(zip(labels, handles))

    # Place unique legend at the bottom center of the whole figure window
    fig.legend(
        by_label.values(), 
        by_label.keys(), 
        loc='outside lower center', 
        ncol=len(by_label), 
        fontsize=12, 
        frameon=True, 
        framealpha=0.9
    )

    if savefig == True:
        plt.savefig("adams_comp_error_only.pdf")
        print("Figure saved as adams_comp_error_only.pdf in current directory.")

    plt.show()

### CONFIG ###
filename1 = "3_closed_TE_error.csv"
filename2 = "3_closed_TE_error_half.csv"

# plot_TE_error("3_closed_TE_error_t100.csv")

# compare_TE_error("3_closed_TE_error_BG_10_50.csv","3_closed_TE_error_BG_100_100.csv","3_closed_TE_error_BG_01_500.csv","3_closed_TE_error_BG_50_500.csv")

# adams_path = r"C:\Users\kaspe\OneDrive - Aarhus universitet\6. semester\Bachelorprojekt\Adams\closed_3_body_adams_results.csv"
# filename3 = "3_closed_gen_acc.csv"

# adams_comp(filename3, adams_path, plot_diff=True, savefig=True)

# adams_comp_error_only(filename3, adams_path, savefig=True)

plot_all_energies("5p_energies.csv")