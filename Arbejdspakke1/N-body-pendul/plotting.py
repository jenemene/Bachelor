import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import soa as SOA
def spatial_plot(t, spatialquantity, type="spatialquantity",bodyno="body number"):
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
    plt.title(f"Spatial Plot: {type.capitalize()} (Body {bodyno})")
    plt.show()

def N_body_pendulum_gen_plot(t_vals,y_vals,n_bodies):
    # Create a figure with a subplot for each body
    fig, axes = plt.subplots(n_bodies, 1, figsize=(10, 2 * n_bodies), sharex=True)

    # If n_bodies is 1, axes is not an array, so we wrap it
    if n_bodies == 1:
        axes = [axes]

    for k in range(n_bodies):
        # Calculate the starting index for beta of body k
        idx_start = 4 * n_bodies + 3 * k
        
        # Extract components: x, y, z (shape: 3, len(t_vals))
        beta_k = y_vals[idx_start : idx_start + 3, :]
        
        # Plotting to the specific subplot
        axes[k].plot(t_vals, beta_k[0, :], label=r'$\omega_x$')
        axes[k].plot(t_vals, beta_k[1, :], label=r'$\omega_y$')
        axes[k].plot(t_vals, beta_k[2, :], label=r'$\omega_z$')
        
        axes[k].set_ylabel(f'Body {k+1}\n[rad/s]')
        axes[k].legend(loc='upper right', fontsize='small')
        axes[k].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Angular Velocity Components per Link', fontsize=14)
    plt.tight_layout()
    plt.show()

def animate_n_bodies(result, l_vec, interval=30):

    n_states, N = result.shape

    # spherical joints: 4 q + 3 omega per joint
    n = int(n_states / 7) + 1
    n_joints = n - 1
    quat_block_size = 4 * n_joints

    # ---- Setup figure ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-10, 30])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-10, 20])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    points, = ax.plot([], [], [], 'o-', lw=2)

    # ==================================
    # --- Forward Kinematics ---
    # ==================================

    def compute_positions(state_k):

        quat_block = state_k[:quat_block_size]

        quats = [
            quat_block[4*i:4*(i+1)]
            for i in range(n_joints)
        ]

        R_cumulative = []
        R = np.eye(3)

        for q in reversed(quats):
            R = R @ SOA.rotfromquat(q)
            R_cumulative.insert(0, R.copy())

        positions = [np.zeros(3)]

        for i in range(n_joints):
            next_pos = positions[-1] + R_cumulative[i] @ l_vec
            positions.append(next_pos)

        return np.array(positions)


    # ==================================
    # --- Animation Update ---
    # ==================================

    def update(frame):

        state_k = result[:, frame]
        positions = compute_positions(state_k)

        points.set_data(positions[:, 0], positions[:, 1])
        points.set_3d_properties(positions[:, 2])

        ax.set_title(f"Frame {frame}/{N}")

        return points,


    ani = animation(
        fig,
        update,
        frames=N,
        interval=interval,
        blit=False
    )

    plt.show()

    return ani

