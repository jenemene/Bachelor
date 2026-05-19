import pandas as pd
import matplotlib.pyplot as plt

def statistical_plot(csv_filename):
    # 1. Load the benchmark data
    df = pd.read_csv(csv_filename)

    # 2. Group by the number of bodies and calculate statistical metrics
    # This computes the mean (the point) and standard deviation (the error bar length)
    stats = df.groupby("Num_Bodies").agg({
        "Solver_RK4_Time": ["mean", "std"],
        "Solver_RK45_Time": ["mean", "std"]
    }).reset_index()

    # Flatten the multi-level column names for easier access
    stats.columns = [
        "Num_Bodies", 
        "RK4_mean", "RK4_std", 
        "RK45_mean", "RK45_std"
    ]

    # 3. Create the plot
    plt.figure(figsize=(10, 6))

    # Plot RK4 with error bars
    plt.errorbar(
        stats["Num_Bodies"], 
        stats["RK4_mean"], 
        yerr=stats["RK4_std"], 
        fmt="o-",          # 'o' for circle markers, '-' for connecting line
        capsize=5,         # Width of the error bar caps
        label="Custom RK4 (Fixed dt)",
        color="#1f77b4",   # Clean blue
        alpha=0.8
    )

    # Plot RK45 with error bars
    plt.errorbar(
        stats["Num_Bodies"], 
        stats["RK45_mean"], 
        yerr=stats["RK45_std"], 
        fmt="s-",          # 's' for square markers, '-' for connecting line
        capsize=5,         
        label="solve_ivp RK45 (Adaptive)",
        color="#ff7f0e",   # Clean orange
        alpha=0.8
    )

    # 4. Styling and labels
    plt.xlabel("Number of Bodies", fontsize=14)
    plt.ylabel("Execution Time (s)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc='best', fontsize=14, frameon=True, framealpha=0.9, labelspacing=1.2)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Adjust x-ticks to dynamically match whatever body counts exist in your data
    plt.xticks(stats["Num_Bodies"])

    # Optimize layout and show the plot
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_TE_delta(csv_filename):
    # 1. Load the data
    df = pd.read_csv(csv_filename)
    
    # 2. Group by Num_Bodies to get mean and std of the energy delta
    stats = df.groupby("Num_Bodies").agg({
        "TE_delta_rk4": ["mean", "std"]
    }).reset_index()
    
    # Flatten column names
    stats.columns = ["Num_Bodies", "TE_delta_mean", "TE_delta_std"]
    
    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plotting directly with the real mean values (preserving negatives)
    plt.errorbar(
        stats["Num_Bodies"], 
        stats["TE_delta_mean"], 
        yerr=stats["TE_delta_std"], 
        fmt="o-", 
        capsize=5,
        color="#d62728",       # Deep red for energy tracking
        ecolor="#222222",      # Dark grey error bars
        elinewidth=1.5,
        label="RK4 Total Energy Delta"
    )
    
    # 4. Styling and Labels
    plt.title("Total Energy Conservation Error vs. Number of Bodies", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Bodies", fontsize=12)
    plt.ylabel("Mean Energy Delta ($\\Delta TE$)", fontsize=12)
    
    # Use standard scientific notation for tiny numbers on the Y-axis
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.xticks(stats["Num_Bodies"])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

csv_filename = "Arbejdspakke2/order_n_val_results/solver_benchmark_results.csv"

statistical_plot(csv_filename)

plot_TE_delta(csv_filename)