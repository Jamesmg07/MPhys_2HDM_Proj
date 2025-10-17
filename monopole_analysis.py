import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.animation as animation

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_3/Monopole_05pi/")
OUTPUT_DIR = Path("/share/centaurus_nas/mkza/Plots/")
nx, ny, nz = 256, 256, 256
dx, dy, dz = 0.5, 0.5, 0.5
dt = 0.1  # Simulation timestep
nPos = nx * ny * nz
nt = int((nx * dx) / (2 * dt))  # Total timesteps: 160

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

def find_energy_files():
    """Find energy files from the simulation"""
    # Look for energy files with gamma in the name
    files = list(DATA_DIR.glob(f"energy_gamma=*_nx{nx}_*.txt"))
    return files

def find_r_values_files():
    """Find all R-values files from the simulation"""
    # Updated to look for the new gamma-dependent naming
    files = list(DATA_DIR.glob(f"R_values_gamma=*_timestep=*_nx{nx}_*.txt"))
    files.sort(key=extract_timestep)
    return files

def load_energy_data(filepath):
    """Load energy data from file"""
    try:
        # Energy files should have one column: Energy
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header if present
        energy_values = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Energy'):  # Skip header
                try:
                    energy_values.append(float(line.split()[0]))  # Take first column
                except (ValueError, IndexError):
                    continue
        
        return np.array(energy_values)
    except Exception as e:
        print(f"Error loading energy data from {filepath}: {e}")
        return None

def plot_energy_vs_time(energy_files):
    """Plot energy evolution over time"""
    print("\nAnalyzing energy evolution...")
    
    if not energy_files:
        print("  No energy files found!")
        return
    
    for energy_file in energy_files:
        print(f"  Loading energy data from: {energy_file.name}")
        
        energy_data = load_energy_data(energy_file)
        
        if energy_data is None or len(energy_data) == 0:
            print(f"    Error: Could not load energy data from {energy_file}")
            continue
        
        # Create time array
        timesteps = np.arange(len(energy_data))
        time_values = timesteps * dt  # Convert to physical time
        
        # Extract gamma value from filename for title
        gamma_match = re.search(r'gamma=([^_]+)', energy_file.name)
        gamma_str = gamma_match.group(1) if gamma_match else "unknown"
        
        plt.figure(figsize=(12, 8))
        
        # Plot energy vs time
        plt.subplot(2, 1, 1)
        plt.plot(time_values, energy_data, 'b-', linewidth=2, label='Total Energy')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title(f'Energy Evolution (γ = {gamma_str})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot energy vs timestep
        plt.subplot(2, 1, 2)
        plt.plot(timesteps, energy_data, 'r-', linewidth=2, label='Total Energy')
        plt.xlabel('Timestep')
        plt.ylabel('Energy')
        plt.title(f'Energy Evolution vs Timestep (γ = {gamma_str})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add simulation info
        plt.figtext(0.02, 0.02, f'Grid: {nx}×{ny}×{nz}, dt = {dt}, Total steps = {len(energy_data)}', 
                   fontsize=10, ha='left')
        
        save_path = OUTPUT_DIR / f'energy_evolution_gamma_{gamma_str}.png'
        save_and_close_plot(save_path, f"    Saved: energy_evolution_gamma_{gamma_str}.png")
        
        # Print energy statistics
        print(f"    Energy statistics for γ = {gamma_str}:")
        print(f"      Initial energy: {energy_data[0]:.6f}")
        print(f"      Final energy: {energy_data[-1]:.6f}")
        print(f"      Energy change: {energy_data[-1] - energy_data[0]:.6f}")
        print(f"      Energy conservation: {abs(energy_data[-1] - energy_data[0])/energy_data[0]*100:.4f}%")

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
    """Plot single x-z slice with R1²+R2²+R3² colormap and (R1,R3) vectors, z as vertical axis"""
    
    slice_index = int(MONOPOLE_CENTER_Y)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract slices (x-z plane at fixed y)
    field1_slice = r_fields['R1nt'][:, slice_index, :]  # R1
    field2_slice = r_fields['R2nt'][:, slice_index, :]  # R2
    field3_slice = r_fields['R3nt'][:, slice_index, :]  # R3
    
    # Calculate R1² + R2² + R3² for monopole identification
    monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2
    
    # Create coordinate grids - z as vertical (y-axis), x as horizontal (x-axis)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz
    X, Z = np.meshgrid(x_coords, z_coords)
    
    # Plot monopole field colormap
    im = ax.contourf(X, Z, monopole_field.T, levels=20, cmap='plasma', vmin=0, vmax=np.max(monopole_field))
    
    # Plot normalized arrows (subsample for clarity)
    step = 3
    # Compute normalization factor for each arrow
    norm = np.sqrt(field1_slice**2 + field2_slice**2 + field3_slice**2)
    # Avoid division by zero
    norm[norm == 0] = 1
    field1_norm = field1_slice / norm
    field3_norm = field3_slice / norm
    for i in range(0, X.shape[0], step):
        for j in range(0, X.shape[1], step):
            ax.quiver(X[i, j], Z[i, j], 
                     field1_norm.T[i, j], field3_norm.T[i, j], 
                     scale=8*3, width=0.004, alpha=0.9, color='white')  # Further divided by 3
    
    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}')
    
    # Removed hardcoded markers for original monopole and antimonopole
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')
    
    if save_individual:
        save_and_close_plot(OUTPUT_DIR / f'xz_monopole_field_t{timestep}.png',
                           f"    Saved: xz_monopole_field_t{timestep}.png")
    else:
        return fig

def create_gif_animation(r_values_files, n_frames=20):
    """Create GIF animation of the x-z slice evolution with monopole field"""
    print(f"\nCreating GIF animation with all available timesteps...")

    # Use all files, sorted by timestep
    selected_files = r_values_files

    all_data = []
    timesteps = []
    global_vmax = 0

    print("  Loading data for all frames...")
    for i, file in enumerate(selected_files):
        if i % max(1, len(selected_files) // 5) == 0:
            print(f"    Loading frame {i+1}/{len(selected_files)}")

        timestep = extract_timestep(file)
        r_fields = load_r_field_data(file)

        if r_fields and all(key in r_fields for key in ['R1nt', 'R2nt', 'R3nt']):
            all_data.append(r_fields)
            timesteps.append(timestep)
            slice_index = int(MONOPOLE_CENTER_Y)
            field1_slice = r_fields['R1nt'][:, slice_index, :]
            field2_slice = r_fields['R2nt'][:, slice_index, :]
            field3_slice = r_fields['R3nt'][:, slice_index, :]
            monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2
            global_vmax = max(global_vmax, np.max(monopole_field))
        else:
            print(f"    Skipped: Invalid data for timestep {timestep}")

    if len(all_data) == 0:
        print("  Error: No valid data found for GIF creation")
        return

    print(f"  Successfully loaded {len(all_data)} frames")
    print(f"  Global vmax for colormap: {global_vmax:.3f}")
    print("  Creating animation...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    slice_index = int(MONOPOLE_CENTER_Y)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz

    # Prepare first frame
    r_fields = all_data[0]
    field1_slice = r_fields['R1nt'][:, slice_index, :]
    field2_slice = r_fields['R2nt'][:, slice_index, :]
    field3_slice = r_fields['R3nt'][:, slice_index, :]
    monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2

    # Normalize vectors for quiver
    norm = np.sqrt(field1_slice**2 + field2_slice**2 + field3_slice**2)
    norm[norm == 0] = 1
    field1_norm = field1_slice / norm
    field3_norm = field3_slice / norm

    im = ax.imshow(
        monopole_field.T,
        origin='lower',
        extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
        aspect='auto',
        cmap='plasma',
        vmin=0,
        vmax=global_vmax,
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')

    step = 4
    quiv = ax.quiver(
        x_coords[::step], z_coords[::step],
        field1_norm.T[::step, ::step], field3_norm.T[::step, ::step],
        scale=8*4.5, width=0.004, alpha=0.8, color='white'  # Further divided by 4.5
    )

    # Removed hardcoded markers for original monopole and antimonopole

    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timesteps[0]}')
    ax.legend()
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(z_coords[0], z_coords[-1])

    def animate(frame):
        r_fields = all_data[frame]
        timestep = timesteps[frame]
        field1_slice = r_fields['R1nt'][:, slice_index, :]
        field2_slice = r_fields['R2nt'][:, slice_index, :]
        field3_slice = r_fields['R3nt'][:, slice_index, :]
        monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2

        # Normalize vectors for quiver
        norm = np.sqrt(field1_slice**2 + field2_slice**2 + field3_slice**2)
        norm[norm == 0] = 1
        field1_norm = field1_slice / norm
        field3_norm = field3_slice / norm

        im.set_data(monopole_field.T)
        quiv.set_UVC(field1_norm.T[::step, ::step], field3_norm.T[::step, ::step])
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}')
        return []

    anim = animation.FuncAnimation(fig, animate, frames=len(all_data), interval=300, blit=False, repeat=True)

    gif_path = OUTPUT_DIR / 'monopole_field_evolution_xz_slice.gif'
    print(f"  Saving GIF to: {gif_path}")

    try:
        anim.save(gif_path, writer='pillow', fps=5, dpi=150)
        print(f"  ✓ Successfully saved GIF: monopole_field_evolution_xz_slice.gif")
    except Exception as e:
        print(f"  Error saving GIF: {e}")
        print("  Note: Make sure Pillow is installed: pip install Pillow")

    plt.close(fig)

def analyze_at_intervals(r_values_files, n_intervals=12):
    """Create x-z monopole field plots at spaced intervals"""
    if len(r_values_files) < n_intervals:
        print(f"Not enough timestep files for {n_intervals} intervals")
        return
    
    # Select files at regular intervals
    indices = np.linspace(0, len(r_values_files)-1, n_intervals, dtype=int)
    selected_files = [r_values_files[i] for i in indices]
    
    print(f"\nCreating x-z monopole field plots for {n_intervals} selected timesteps...")
    
    for plot_num, file in enumerate(selected_files):
        timestep = extract_timestep(file)
        print(f"  [{plot_num+1}/{n_intervals}] Loading data for timestep {timestep}...")
        
        r_fields = load_r_field_data(file)
        
        if r_fields and all(key in r_fields for key in ['R1nt', 'R2nt', 'R3nt']):
            print(f"    Creating x-z monopole field plot for timestep {timestep}...")
            plot_xz_slice_vectors(r_fields, timestep)
            print(f"    ✓ Completed plot for timestep {timestep}")
        else:
            print(f"    Skipped: Invalid data for timestep {timestep}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("MONOPOLE FIELD ANALYSIS WITH ENERGY EVOLUTION")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 5
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Simulation parameters: Grid={nx}×{ny}×{nz}, dt={dt}, Expected timesteps={nt}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Searching for data files...")
    
    # Find files
    energy_files = find_energy_files()
    r_values_files = find_r_values_files()
    
    print(f"Found {len(energy_files)} energy files")
    print(f"Found {len(r_values_files)} R-values files")
    
    if not energy_files and not r_values_files:
        print("ERROR: No data files found!")
        exit()
    
    print_progress(3, total_steps, "Analyzing energy evolution...")
    plot_energy_vs_time(energy_files)
    
    if r_values_files:
        print_progress(4, total_steps, "Creating x-z monopole field plots...")
        analyze_at_intervals(r_values_files, n_intervals=12)
        
        print_progress(5, total_steps, "Creating monopole field GIF animation...")
        create_gif_animation(r_values_files, n_frames=25)
    else:
        print("Skipping R-field analysis (no R-values files found)")
    
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)