import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.animation as animation

# Simulation parameters
DATA_DIR = Path("./Data/")
OUTPUT_DIR = Path("./Plots/")
nx, ny, nz = 64, 64, 64
dx, dy, dz = 0.5, 0.5, 0.5
nPos = nx * ny * nz

# Original monopole positions from C++ code (index positions)
MONOPOLE_1_POS = (31.5, 31.5, 43.5)  # Index positions
MONOPOLE_2_POS = (31.5, 31.5, 19.5)  # Index positions
MONOPOLE_CENTER_Y = 31.5

# Create output directory for plots
OUTPUT_DIR.mkdir(exist_ok=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def extract_timestep(filename):
    """Extract timestep from filename"""
    match = re.search(r'timestep=(\d+)', filename.name)
    return int(match.group(1)) if match else 0

def find_t_gamma_files():
    """Find all t_gamma R-values files from the simulation"""
    files = list(DATA_DIR.glob("t_gamma=2pi_3_R_values_timestep=*_nx64_*.txt"))
    files.sort(key=extract_timestep)
    return files

def load_r_field_data(filepath):
    """Load R-field data and reshape to 3D grid"""
    try:
        data = pd.read_csv(filepath, sep=' ')
        r_fields = {}
        
        for col in ['R1nt', 'R2nt', 'R3nt']:
            if col in data.columns:
                field_1d = data[col].values
                if len(field_1d) == nPos:
                    r_fields[col] = field_1d.reshape(nx, ny, nz)
        
        return r_fields
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_and_close_plot(save_path, message):
    """Helper function to save and close plots"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(message)
    plt.close()

def plot_xz_slice_vectors(r_fields, timestep, save_individual=True):
    """Plot x-z slice with R-field vectors, z as vertical axis"""
    
    slice_index = int(MONOPOLE_CENTER_Y)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract slices (x-z plane at fixed y)
    field1_slice = r_fields['R1nt'][:, slice_index, :]  # R1
    field3_slice = r_fields['R3nt'][:, slice_index, :]  # R3
    
    # Create coordinate grids - z as vertical (y-axis), x as horizontal (x-axis)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz
    X, Z = np.meshgrid(x_coords, z_coords)
    
    field_names = ['R1nt', 'R3nt']
    field_slices = [field1_slice, field3_slice]
    
    # Plot both fields
    for idx, (field_slice, field_name, ax) in enumerate(zip(field_slices, field_names, axes)):
        
        # Transpose to make z vertical
        im = ax.contourf(X, Z, field_slice.T, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Plot arrows (subsample for clarity)
        step = 3
        for i in range(0, X.shape[0], step):
            for j in range(0, X.shape[1], step):
                ax.quiver(X[i, j], Z[i, j], 
                         field1_slice.T[i, j], field3_slice.T[i, j], 
                         scale=3, width=0.005, alpha=0.9, color='black')
        
        ax.set_xlabel('x position')
        ax.set_ylabel('z position')
        ax.set_title(f'{field_name} field + (R1,R3) vectors (y={slice_index}) - t={timestep}')
        
        # Mark original monopole positions with green crosses
        x1_phys = MONOPOLE_1_POS[0] * dx
        z1_phys = MONOPOLE_1_POS[2] * dz
        x2_phys = MONOPOLE_2_POS[0] * dx
        z2_phys = MONOPOLE_2_POS[2] * dz
        
        ax.scatter([x1_phys], [z1_phys], color='green', s=150, marker='+', linewidth=4, label='Original Monopole')
        ax.scatter([x2_phys], [z2_phys], color='green', s=150, marker='+', linewidth=4, label='Original Antimonopole')
        ax.legend()
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{field_name} field strength')
    
    if save_individual:
        save_and_close_plot(OUTPUT_DIR / f'xz_slice_vectors_t{timestep}.png',
                           f"    Saved: xz_slice_vectors_t{timestep}.png")
    else:
        return fig

def create_gif_animation(t_gamma_files, n_frames=20):
    """Create GIF animation of the x-z slice evolution"""
    print(f"\nCreating GIF animation with {n_frames} frames...")
    
    if len(t_gamma_files) < n_frames:
        print(f"Not enough timestep files for {n_frames} frames, using {len(t_gamma_files)} frames")
        n_frames = len(t_gamma_files)
    
    # Select files at regular intervals
    indices = np.linspace(0, len(t_gamma_files)-1, n_frames, dtype=int)
    selected_files = [t_gamma_files[i] for i in indices]
    
    # Load all data first
    all_data = []
    timesteps = []
    
    print("  Loading data for all frames...")
    for i, file in enumerate(selected_files):
        if i % max(1, n_frames // 5) == 0:
            print(f"    Loading frame {i+1}/{n_frames}")
        
        timestep = extract_timestep(file)
        r_fields = load_r_field_data(file)
        
        if r_fields and all(key in r_fields for key in ['R1nt', 'R3nt']):
            all_data.append(r_fields)
            timesteps.append(timestep)
        else:
            print(f"    Skipped: Invalid data for timestep {timestep}")
    
    if len(all_data) == 0:
        print("  Error: No valid data found for GIF creation")
        return
    
    print(f"  Successfully loaded {len(all_data)} frames")
    print("  Creating animation...")
    
    # Set up the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create coordinate grids
    slice_index = int(MONOPOLE_CENTER_Y)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz
    X, Z = np.meshgrid(x_coords, z_coords)
    
    # Initialize empty plots
    contour_plots = []
    quiver_plots = []
    
    def animate(frame):
        # Clear previous plots
        for ax in axes:
            ax.clear()
        
        r_fields = all_data[frame]
        timestep = timesteps[frame]
        
        # Extract slices
        field1_slice = r_fields['R1nt'][:, slice_index, :]  # R1
        field3_slice = r_fields['R3nt'][:, slice_index, :]  # R3
        
        field_names = ['R1nt', 'R3nt']
        field_slices = [field1_slice, field3_slice]
        
        # Plot both fields
        for idx, (field_slice, field_name, ax) in enumerate(zip(field_slices, field_names, axes)):
            
            # Transpose to make z vertical
            im = ax.contourf(X, Z, field_slice.T, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Plot arrows (subsample for clarity)
            step = 4  # Slightly larger step for GIF performance
            for i in range(0, X.shape[0], step):
                for j in range(0, X.shape[1], step):
                    ax.quiver(X[i, j], Z[i, j], 
                             field1_slice.T[i, j], field3_slice.T[i, j], 
                             scale=3, width=0.005, alpha=0.8, color='black')
            
            ax.set_xlabel('x position')
            ax.set_ylabel('z position')
            ax.set_title(f'{field_name} field + (R1,R3) vectors (y={slice_index}) - t={timestep}')
            
            # Mark original monopole positions with green crosses
            x1_phys = MONOPOLE_1_POS[0] * dx
            z1_phys = MONOPOLE_1_POS[2] * dz
            x2_phys = MONOPOLE_2_POS[0] * dx
            z2_phys = MONOPOLE_2_POS[2] * dz
            
            ax.scatter([x1_phys], [z1_phys], color='green', s=150, marker='+', linewidth=4, label='Original Monopole')
            ax.scatter([x2_phys], [z2_phys], color='green', s=150, marker='+', linewidth=4, label='Original Antimonopole')
            ax.legend()
        
        plt.tight_layout()
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(all_data), interval=200, blit=False, repeat=True)
    
    # Save as GIF
    gif_path = OUTPUT_DIR / 'monopole_evolution_xz_slice.gif'
    print(f"  Saving GIF to: {gif_path}")
    
    try:
        anim.save(gif_path, writer='pillow', fps=5, dpi=150)
        print(f"  ✓ Successfully saved GIF: monopole_evolution_xz_slice.gif")
    except Exception as e:
        print(f"  Error saving GIF: {e}")
        print("  Note: Make sure Pillow is installed: pip install Pillow")
    
    plt.close(fig)

def analyze_at_intervals(t_gamma_files, n_intervals=12):
    """Create x-z plots at spaced intervals"""
    if len(t_gamma_files) < n_intervals:
        print(f"Not enough timestep files for {n_intervals} intervals")
        return
    
    # Select files at regular intervals
    indices = np.linspace(0, len(t_gamma_files)-1, n_intervals, dtype=int)
    selected_files = [t_gamma_files[i] for i in indices]
    
    print(f"\nCreating x-z slice plots for {n_intervals} selected timesteps...")
    
    for plot_num, file in enumerate(selected_files):
        timestep = extract_timestep(file)
        print(f"  [{plot_num+1}/{n_intervals}] Loading data for timestep {timestep}...")
        
        r_fields = load_r_field_data(file)
        
        if r_fields and all(key in r_fields for key in ['R1nt', 'R3nt']):
            print(f"    Creating x-z slice plot for timestep {timestep}...")
            plot_xz_slice_vectors(r_fields, timestep)
            print(f"    ✓ Completed plot for timestep {timestep}")
        else:
            print(f"    Skipped: Invalid data for timestep {timestep}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("MONOPOLE VECTOR FIELD ANALYSIS")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 4
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Searching for data files...")
    
    # Find files
    t_gamma_files = find_t_gamma_files()
    
    print(f"Found {len(t_gamma_files)} t_gamma files")
    
    if not t_gamma_files:
        print("ERROR: No t_gamma files found!")
        exit()
    
    print_progress(3, total_steps, "Creating x-z slice vector plots...")
    analyze_at_intervals(t_gamma_files, n_intervals=12)
    
    print_progress(4, total_steps, "Creating GIF animation...")
    create_gif_animation(t_gamma_files, n_frames=25)
    
    print(f"All vector plots and GIF saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)