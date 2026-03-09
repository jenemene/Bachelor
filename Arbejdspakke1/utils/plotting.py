import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from utils import soa as SOA
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter # Add this import

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

def plot_initial_state(state0, link, config="openclosed"):
    # Number of bodies
    n_bodies = int(state0.shape[0] / 7)    

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Defining plotlim based on the number of bodies, link length and configuration
    if config == "open":
        plotlim = (n_bodies+1) * np.linalg.norm(link.l_hinge)
    elif config == "closed":
        plotlim = ((n_bodies+1) * np.linalg.norm(link.l_hinge))/2
    else:
        raise ValueError("Invalid config value. Use 'open' or 'closed'.")
    
    ax.set_xlim([-plotlim, plotlim])
    ax.set_ylim([-plotlim, plotlim])
    ax.set_zlim([-plotlim, plotlim])
    ax.set(box_aspect=(1, 1, 1))
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Initial State Configuration")

    # Compute positions of the bodies in inertial frame based on the initial state
    positions = SOA.compute_pos_in_inertial_frame(state0[:4*n_bodies], link.l_hinge, n_bodies)

    # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
    # so we have all positions needed for plottigng in one array.
    R3_tip2I = SOA.get_rotation_tip_to_body_I(state0[:4*n_bodies], n_bodies)[:3,:3]
    tip_pos = (positions[1] + R3_tip2I @ link.l_hinge).flatten()
    positions[0] = tip_pos
    
    pos_array = np.array(positions)

    # Plotting the "skeleton"
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'o-', lw=3, markersize=8)

    ax.view_init(elev=0, azim=-90, roll=0)
    plt.grid(True)
    plt.show()

    return fig, ax

def check_energies(result, V_values, tspan, link, n, config="openclosed"):
    timesteps = len(tspan)
    KE = np.zeros(timesteps)
    PE = np.zeros(timesteps)
    TE = np.zeros(timesteps)

    start = 0.1
    step = 0.2
    g = 9.81

    if config == "open":
        z0 = np.arange(n) * step + start
        z0 = np.flip(z0) #Make it compatible with our convention -> body n connected to inertial.
        z0 = np.insert(z0, 0, 0)
    elif config == "closed":
        assert n % 2 == 0, "check_energies function only supports closed configuration with even number of bodies"
        ztemp = np.arange(n/2) * step + start
        z0 = np.concatenate([ztemp, np.flip(ztemp)]) # Mirror the z-positions for the second half of the system
        z0 = np.insert(z0, 0, 0)

    for i in range(timesteps):
        KE_t = 0.0
        PE_t = 0.0
        com_pos = SOA.compute_com_pos_in_inertial_frame(result[:,i], link.l_hinge, n)

        for k in range(1,n+1):            
            # Kinetic energy
            Vk = V_values[i][k] 
            KE_link = Vk @ link.M @ Vk
            KE_t += 0.5*KE_link
            
            # Potential energy
            zk = com_pos[k][-1] # z-pos of current body k
            zk_pot = zk + z0[k] # potential height of current body

            PE_link = link.m*g*zk_pot
            PE_t += PE_link

        KE[i] = KE_t
        PE[i] = PE_t
        TE[i] = KE_t + PE_t

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

def animation_plot(states, tspan, link, config="openclosed"):
    # Number of bodies
    n_bodies = int(states.shape[0] / 7)

    # Number of time steps
    assert states.shape[1] == len(tspan), "Number of time steps in states must match length of tspan"
    N = states.shape[1]

    # Setting up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Defining plotlim based on the number of bodies, link length and configuration
    if config == "open":
        plotlim = (n_bodies+1) * np.linalg.norm(link.l_hinge)
    elif config == "closed":
        plotlim = ((n_bodies+1) * np.linalg.norm(link.l_hinge))/2
    else:
        raise ValueError("Invalid config value. Use 'open' or 'closed'.")

    # Set plot limits and labels
    ax.set_xlim([-plotlim, plotlim])
    ax.set_ylim([-plotlim, plotlim])
    ax.set_zlim([-plotlim, plotlim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set(box_aspect=(1, 1, 1))

    # Initialize the line object that will be updated in the animation
    line, = ax.plot([], [], [], 'o-', lw=2)

    # Computes the positions of the joints/links based on the state vector at each time step AND position of the tip of the last link
    def compute_positions(state_k):
        positions = SOA.compute_pos_in_inertial_frame(state_k[:4*n_bodies], link.l_hinge, n_bodies)
        
        # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
        # so we have all positions in one array.
        R6_tip2I = SOA.get_rotation_tip_to_body_I(state_k[:4*n_bodies], n_bodies)
        R3_tip2I = R6_tip2I[:3,:3]
        tip_pos = (positions[1] + R3_tip2I @ link.l_hinge).flatten()
        positions[0] = tip_pos
        return np.array(positions) # Convert list of arrays to a single 2D array of shape (n_bodies+1, 3) for easier plotting
    
    # Update function for animation
    def update(frame):
        state_k = states[:, frame]
        positions = compute_positions(state_k)
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])
        ax.set_title(f"t = {tspan[frame]:.3f} s")

    # Just a robust way of calculating the interval between frames for the animation, based on the time vector. Could also do tspan[1] - tspan[0]. 
    dt = np.mean(np.diff(tspan))
    interval = dt * 1000 # Convert to milliseconds for FuncAnimation
    ani = FuncAnimation(
        fig, update, frames=N, interval=interval, blit=False
    )
    
    ax.view_init(elev=0, azim=-90, roll=0)
    plt.show()
    return ani