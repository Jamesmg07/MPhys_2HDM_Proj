import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.animation as animation
import time  # Add timing

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_3/Monopole_05pi/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp/Week_3/Monopole_05pi/")
nx, ny, nz = 256, 256, 256
dx, dy, dz = 0.5, 0.5, 0.5
dt = 0.1  # Simulation timestep
nPos = nx * ny * nz
nt = int((nx * dx) / (2 * dt))  # Total timesteps: 160

# Original monopole positions from C++ code (index positions)
MONOPOLE_1_POS = (31.5, 31.5, 43.5)  # Index positions
MONOPOLE_2_POS = (31.5, 31.5, 19.5)  # Index positions
MONOPOLE_CENTER_Y = 31.5

# Grid-dependent scaling parameters
def get_grid_scaling():
    """Calculate grid-dependent scaling factors for arrows"""
    # Base grid size for reference (64^3)
    base_grid = 64
    
    # Calculate scaling factor based on current grid size
    grid_scale = nx / base_grid
    
    # Arrow spacing: increase step size with larger grids
    arrow_step = max(2, int(3 * grid_scale))
    
    # Arrow scale: increase with grid size to maintain visibility
    arrow_scale = 8 * (grid_scale ** 0.7)  # Sublinear scaling
    
    # Arrow width: decrease slightly with larger grids to avoid crowding
    arrow_width = 0.004 / (grid_scale ** 0.3)
    
    return arrow_step, arrow_scale, arrow_width

# Get scaling parameters
ARROW_STEP, ARROW_SCALE, ARROW_WIDTH = get_grid_scaling()

print(f"Grid scaling factors: step={ARROW_STEP}, scale={ARROW_SCALE:.1f}, width={ARROW_WIDTH:.6f}")

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
    """Load R-field data and reshape to 3D grid - optimized version"""
    try:
        # Use numpy for faster loading instead of pandas
        data = np.loadtxt(filepath)
        r_fields = {}
        
        # Assuming columns are in order: R1nt, R2nt, R3nt
        if data.shape[1] >= 3:
            for i, col in enumerate(['R1nt', 'R2nt', 'R3nt']):
                if data.shape[0] == nPos:
                    r_fields[col] = data[:, i].reshape(nx, ny, nz)
        
        return r_fields
    except Exception as e:
        # Fallback to pandas if numpy fails
        try:
            data = pd.read_csv(filepath, sep=' ')
            r_fields = {}
            
            for col in ['R1nt', 'R2nt', 'R3nt']:
                if col in data.columns:
                    field_1d = data[col].values
                    if len(field_1d) == nPos:
                        r_fields[col] = field_1d.reshape(nx, ny, nz)
            
            return r_fields
        except Exception as e2:
            print(f"Error loading {filepath}: {e2}")
            return None

def save_and_close_plot(save_path, message):
    """Helper function to save and close plots"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(message)
    plt.close()

def plot_xz_slice_vectors(r_fields, timestep, save_individual=True):
    """Plot single x-z slice with R1²+R2²+R3² colormap and (R1,R3) vectors, z as vertical axis - optimized"""
    
    start_time = time.time()
    
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
    
    # Use imshow instead of contourf for better performance
    im = ax.imshow(monopole_field.T, origin='lower', 
                   extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                   aspect='auto', cmap='plasma', vmin=0, vmax=np.max(monopole_field),
                   interpolation='bilinear')  # Faster interpolation
    
    # VECTORIZED arrow plotting - much faster than nested loops
    # Create subsampled coordinate arrays
    x_sub = x_coords[::ARROW_STEP]
    z_sub = z_coords[::ARROW_STEP]
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub)
    
    # Subsample and normalize field components vectorized
    field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
    field3_sub = field3_slice[::ARROW_STEP, ::ARROW_STEP]
    
    # Vectorized normalization
    norm_sub = np.sqrt(field1_sub**2 + field3_sub**2)
    norm_sub[norm_sub == 0] = 1  # Avoid division by zero
    field1_norm = field1_sub / norm_sub
    field3_norm = field3_sub / norm_sub
    
    # Single vectorized quiver call instead of nested loops
    ax.quiver(X_sub, Z_sub, field1_norm.T, field3_norm.T,
              scale=ARROW_SCALE, width=ARROW_WIDTH, alpha=0.9, color='white')
    
    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Total timesteps: {nt}')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')
    
    print(f"    Plot creation time: {time.time() - start_time:.2f}s")
    
    if save_individual:
        save_and_close_plot(OUTPUT_DIR / f'xz_monopole_field_t{timestep}.png',
                           f"    Saved: xz_monopole_field_t{timestep}.png")
    else:
        return fig

def create_gif_animation(r_values_files, n_frames=20):
    """Create GIF animation of the x-z slice evolution with monopole field - optimized"""
    print(f"\nCreating GIF animation with exactly 10 evenly spaced timesteps...")

    # Always use exactly 10 frames, evenly spaced
    if len(r_values_files) < 10:
        print(f"  Warning: Only {len(r_values_files)} files available, using all of them")
        selected_files = r_values_files
    else:
        # Select exactly 10 evenly spaced frames
        indices = np.linspace(0, len(r_values_files)-1, 10, dtype=int)
        selected_files = [r_values_files[i] for i in indices]
        print(f"  Using 10 frames out of {len(r_values_files)} total timesteps saved")

    all_data = []
    timesteps = []
    global_vmax = 0

    print("  Loading data for all frames...")
    load_start = time.time()
    for i, file in enumerate(selected_files):
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

    print(f"  Data loading time: {time.time() - load_start:.2f}s")

    if len(all_data) == 0:
        print("  Error: No valid data found for GIF creation")
        return

    print(f"  Successfully loaded {len(all_data)} frames")
    print(f"  Global vmax for colormap: {global_vmax:.3f}")
    print("  Creating animation...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Smaller figure for faster rendering
    slice_index = int(MONOPOLE_CENTER_Y)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz

    # Prepare first frame
    r_fields = all_data[0]
    field1_slice = r_fields['R1nt'][:, slice_index, :]
    field2_slice = r_fields['R2nt'][:, slice_index, :]
    field3_slice = r_fields['R3nt'][:, slice_index, :]
    monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2

    # Pre-compute subsampled coordinates for arrows
    x_sub = x_coords[::ARROW_STEP]
    z_sub = z_coords[::ARROW_STEP]
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub)

    im = ax.imshow(
        monopole_field.T,
        origin='lower',
        extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
        aspect='auto',
        cmap='plasma',
        vmin=0,
        vmax=global_vmax,
        interpolation='bilinear'  # Faster than 'nearest'
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')

    # Initialize arrows with first frame
    field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
    field3_sub = field3_slice[::ARROW_STEP, ::ARROW_STEP]
    norm_sub = np.sqrt(field1_sub**2 + field3_sub**2)
    norm_sub[norm_sub == 0] = 1
    field1_norm = field1_sub / norm_sub
    field3_norm = field3_sub / norm_sub

    quiv = ax.quiver(X_sub, Z_sub, field1_norm.T, field3_norm.T,
                     scale=ARROW_SCALE, width=ARROW_WIDTH, alpha=0.8, color='white')

    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timesteps[0]}\nGrid: {nx}×{ny}×{nz}, Total timesteps: {nt}')
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(z_coords[0], z_coords[-1])

    def animate(frame):
        r_fields = all_data[frame]
        timestep = timesteps[frame]
        field1_slice = r_fields['R1nt'][:, slice_index, :]
        field2_slice = r_fields['R2nt'][:, slice_index, :]
        field3_slice = r_fields['R3nt'][:, slice_index, :]
        monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2

        # Vectorized arrow updates
        field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
        field3_sub = field3_slice[::ARROW_STEP, ::ARROW_STEP]
        norm_sub = np.sqrt(field1_sub**2 + field3_sub**2)
        norm_sub[norm_sub == 0] = 1
        field1_norm = field1_sub / norm_sub
        field3_norm = field3_sub / norm_sub

        im.set_data(monopole_field.T)
        quiv.set_UVC(field1_norm.T, field3_norm.T)
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Total timesteps: {nt}')
        return [im, quiv]

    # Faster animation settings
    anim = animation.FuncAnimation(fig, animate, frames=len(all_data), 
                                 interval=200, blit=True, repeat=True)  # Enabled blit for speed

    gif_path = OUTPUT_DIR / 'monopole_field_evolution_xz_slice.gif'
    print(f"  Saving GIF to: {gif_path}")

    anim_start = time.time()
    try:
        # Lower quality settings for faster generation
        anim.save(gif_path, writer='pillow', fps=8, dpi=100)  # Reduced DPI and increased FPS
        print(f"  ✓ Successfully saved GIF: monopole_field_evolution_xz_slice.gif")
        print(f"  Animation generation time: {time.time() - anim_start:.2f}s")
    except Exception as e:
        print(f"  Error saving GIF: {e}")
        print("  Note: Make sure Pillow is installed: pip install Pillow")

    plt.close(fig)

def analyze_at_intervals(r_values_files, n_intervals=12):
    """Create x-z monopole field plots at spaced intervals"""
    # Always use exactly 10 intervals
    n_intervals = 10
    
    if len(r_values_files) < n_intervals:
        print(f"Only {len(r_values_files)} files available, using all of them")
        selected_files = r_values_files
    else:
        # Select exactly 10 evenly spaced files
        indices = np.linspace(0, len(r_values_files)-1, n_intervals, dtype=int)
        selected_files = [r_values_files[i] for i in indices]
    
    print(f"\nCreating x-z monopole field plots for {len(selected_files)} evenly spaced timesteps...")
    
    for plot_num, file in enumerate(selected_files):
        timestep = extract_timestep(file)
        print(f"  [{plot_num+1}/{len(selected_files)}] Loading data for timestep {timestep}...")
        
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