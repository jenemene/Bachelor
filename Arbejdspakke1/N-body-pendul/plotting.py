import matplotlib.pyplot as plt
import numpy as np

def spatial_plot(t, spatialquantity, type="spatialquantity"):
    # t: time vector as a np array of shape (N,)
    # spatialquantity: spatial quantity as a np array of shape (6,N)
    # type: string indicating the type of spatial quantity ("force", "velocity", "acceleration")
    
    # Safety checks
    assert spatialquantity.shape[0] == 6, "Input spatial quantity must have shape (6,N)"
    assert spatialquantity.shape[1] == t.shape[0], "Time vector length must match spatial quantity length"

    # Configuration for labels and prefixes
    config = {
        "force":        {"l_label": "Force [N]",     "l_pre": "F", "r_label": "Torque [Nm]",   "r_pre": "N"},
        "velocity":     {"l_label": "Vel [m/s]",     "l_pre": "v", "r_label": "AngVel [rad/s]", "r_pre": "ω"},
        "acceleration": {"l_label": "Acc [m/s²]",    "l_pre": "a", "r_label": "AngAcc [rad/s²]","r_pre": "α"}
    }

    #labels x y and z
    labels = ['x', 'y', 'z']

    s = config[type] 
    
    # Create the plot and the twin axis
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    # colors. RGB = XYZ
    colors = ['#B22222', "#336933", '#000080']
   
    # 1. Plot Linear (Rows 0, 1, 2) on the left - Solid lines
    for i in range(3):
        ax_left.plot(t, spatialquantity[i,:], color=colors[i], linestyle='-', label=f'${s["l_pre"]}_{labels[i]}$')
    
    # 2. Plot Angular (Rows 3, 4, 5) on the right - Dashed lines
    for i in range(3, 6):
        # We use i-3 for colors so it stays 0, 1, 2
        ax_right.plot(t, spatialquantity[i,:], color=colors[i-3], linestyle='--', label=f'${s["r_pre"]}_{labels[i-3]}$')

    # Labels
    ax_left.set_xlabel("Time [s]")
    ax_left.set_ylabel(s["l_label"], fontweight='bold')
    ax_right.set_ylabel(s["r_label"], fontweight='bold', rotation=270, labelpad=15)

    # Grid and Legends
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')

    #limits
    ax_left.set_xlim(t[0], t[-1])
    plt.title(f"Spatial Plot: {type.capitalize()}")
    plt.show()