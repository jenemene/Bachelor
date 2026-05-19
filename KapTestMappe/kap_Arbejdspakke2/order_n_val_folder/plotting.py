import pandas as pd
import matplotlib.pyplot as plt

def statistical_plot():
    # 1. Load the benchmark data
    # Replace with your actual path if it's saved elsewhere
    csv_filename = "KapTestMappe/kap_arbejdspakke2/order_n_val_folder/solver_benchmark_results.csv"
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

statistical_plot()