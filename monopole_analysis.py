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
nt = int((nx * dx) / (2 * dt))  

def get_monopole_positions():
    """Calculate monopole positions from C++ parameters"""
    # From your C++ code:
    offset_from_centre = 0.25
    
    # Center positions - should be integer grid indices
    center_x = (nx - 1) // 2  # 127 for 256³ grid
    center_y = (ny - 1) // 2  # 127 for 256³ grid  
    center_z = (nz - 1) // 2  # 127 for 256³ grid
    
    # Monopole positions with offsets (convert to grid indices)
    offset_grid = int(offset_from_centre * nz)  # 64 grid points for 256³
    monopole1_z = center_z + offset_grid  # 127 + 64 = 191
    monopole2_z = center_z - offset_grid  # 127 - 64 = 63
    
    return (center_x, center_y, center_z), (center_x, center_y, monopole1_z), (center_x, center_y, monopole2_z)

# Fix 3: Update your slice selection
center_pos, monopole1_pos, monopole2_pos = get_monopole_positions()

# Use the monopole y-coordinate for slicing (both monopoles have same x,y coordinates)
MONOPOLE_CENTER_Y = int(monopole1_pos[1])  # 127 for 256³ grid - where monopoles actually are
MONOPOLE_1_Y = int(monopole1_pos[1])       # 127 for 256³ grid
MONOPOLE_2_Y = int(monopole2_pos[1])       # 127 for 256³ grid

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

def prepare_slice_data(r_fields, slice_index):
    """Prepare slice data for plotting - common function for both individual plots and GIF"""
    # Extract slices (x-z plane at fixed y where monopoles are located)
    field1_slice = r_fields['R1nt'][:, slice_index, :]  # R1
    field2_slice = r_fields['R2nt'][:, slice_index, :]  # R2
    field3_slice = r_fields['R3nt'][:, slice_index, :]  # R3
    
    # Calculate R1² + R2² + R3² for monopole identification
    monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2
    
    # Create coordinate grids - z as vertical (y-axis), x as horizontal (x-axis)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz
    
    # Subsample field components for arrows
    field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
    field3_sub = field3_slice[::ARROW_STEP, ::ARROW_STEP]
    
    # Calculate field magnitude for color coding
    field_magnitude = np.sqrt(field1_sub**2 + field3_sub**2)
    
    # Don't normalize direction - show actual field strength, but make arrows visible
    mask = field_magnitude > 1e-10
    field1_plot = np.where(mask, field1_sub, 0)
    field3_plot = np.where(mask, field3_sub, 0)
    
    return {
        'monopole_field': monopole_field,
        'field1_plot': field1_plot,
        'field3_plot': field3_plot,
        'field_magnitude': field_magnitude,
        'x_coords': x_coords,
        'z_coords': z_coords
    }

def setup_slice_plot(ax, slice_data, vmin, vmax, arrow_scale, global_arrow_max, slice_index, timestep):
    """Setup the slice plot with common styling - used by both individual plots and GIF"""
    monopole_field = slice_data['monopole_field']
    field1_plot = slice_data['field1_plot']
    field3_plot = slice_data['field3_plot']
    field_magnitude = slice_data['field_magnitude']
    x_coords = slice_data['x_coords']
    z_coords = slice_data['z_coords']
    
    # Create subsampled coordinate arrays for arrows
    x_sub = x_coords[::ARROW_STEP]
    z_sub = z_coords[::ARROW_STEP]
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub)
    
    # Background field plot
    im = ax.imshow(monopole_field.T, origin='lower', 
                   extent=[x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]],
                   aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
    
    # Arrow plot
    quiv = ax.quiver(X_sub, Z_sub, field1_plot.T, field3_plot.T,
                     field_magnitude.T, scale=arrow_scale, width=ARROW_WIDTH, 
                     alpha=0.9, cmap='Reds', scale_units='xy',
                     clim=(0, global_arrow_max))
    
    # Labels and title
    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Monopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}')
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(z_coords[0], z_coords[-1])
    
    return im, quiv

def create_individual_plot(slice_data, timestep, vmin, vmax, global_arrow_max):
    """Create and save individual plot from pre-calculated slice data"""
    slice_index = int(MONOPOLE_CENTER_Y)
    arrow_scale = max(global_arrow_max * 0.25, 1e-6)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Setup plot using common function
    im, quiv = setup_slice_plot(ax, slice_data, vmin, vmax, arrow_scale, global_arrow_max, slice_index, timestep)
    
    # Add colorbars
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')
    
    cbar2 = plt.colorbar(quiv, ax=ax, shrink=0.6, pad=0.1)
    cbar2.set_label('Field magnitude |R1,R3|')
    
    # Save and close
    save_and_close_plot(OUTPUT_DIR / f'xz_monopole_field_t{timestep}.png',
                       f"    Saved: xz_monopole_field_t{timestep}.png")

def analyze_and_create_all_outputs(r_values_files):
    """Single function that creates both individual plots and GIF - no duplication of work"""
    print(f"\nAnalyzing field data and creating all outputs...")
    
    # Always use exactly 10 intervals
    n_intervals = 10
    
    if len(r_values_files) < n_intervals:
        print(f"  Only {len(r_values_files)} files available, using all of them")
        selected_files = r_values_files
    else:
        # Select exactly 10 evenly spaced files
        indices = np.linspace(0, len(r_values_files)-1, n_intervals, dtype=int)
        selected_files = [r_values_files[i] for i in indices]
        print(f"  Using {len(selected_files)} evenly spaced timesteps out of {len(r_values_files)} total")

    # SINGLE PASS: Load all data and calculate everything once
    all_slice_data = []
    timesteps = []
    global_vmax = 0
    global_vmin = float('inf')
    global_arrow_max = 0
    slice_index = int(MONOPOLE_CENTER_Y)

    print("  Loading and processing all data (single pass)...")
    load_start = time.time()
    
    for i, file in enumerate(selected_files):
        print(f"    [{i+1}/{len(selected_files)}] Processing file for timestep {extract_timestep(file)}...")

        timestep = extract_timestep(file)
        r_fields = load_r_field_data(file)

        if r_fields and all(key in r_fields for key in ['R1nt', 'R2nt', 'R3nt']):
            # Calculate slice data once
            slice_data = prepare_slice_data(r_fields, slice_index)
            all_slice_data.append(slice_data)
            timesteps.append(timestep)
            
            monopole_field = slice_data['monopole_field']
            field_magnitude = slice_data['field_magnitude']
            
            # Debug: Print field statistics
            print(f"      Field stats: min={np.min(monopole_field):.6f}, max={np.max(monopole_field):.6f}, mean={np.mean(monopole_field):.6f}")
            print(f"      Arrow magnitude: min={np.min(field_magnitude):.6f}, max={np.max(field_magnitude):.6f}")
            
            # Update global scaling parameters
            frame_vmax = np.percentile(monopole_field, 99)
            frame_vmin = np.min(monopole_field)
            frame_arrow_max = np.max(field_magnitude)
            
            global_vmax = max(global_vmax, frame_vmax)
            global_vmin = min(global_vmin, frame_vmin)
            global_arrow_max = max(global_arrow_max, frame_arrow_max)
        else:
            print(f"      Skipped: Invalid data for timestep {timestep}")

    print(f"  Data processing completed in {time.time() - load_start:.2f}s")

    if len(all_slice_data) == 0:
        print("  Error: No valid data found!")
        return

    print(f"  Successfully processed {len(all_slice_data)} timesteps")
    print(f"  Global scaling - vmin: {global_vmin:.6f}, vmax: {global_vmax:.6f}, arrow_max: {global_arrow_max:.6f}")

    # CREATE INDIVIDUAL PLOTS using pre-calculated data
    print("\n  Creating individual plots...")
    individual_start = time.time()
    
    for i, (slice_data, timestep) in enumerate(zip(all_slice_data, timesteps)):
        print(f"    [{i+1}/{len(all_slice_data)}] Creating individual plot for timestep {timestep}...")
        create_individual_plot(slice_data, timestep, global_vmin, global_vmax, global_arrow_max)
    
    print(f"  Individual plots completed in {time.time() - individual_start:.2f}s")

    # CREATE GIF ANIMATION using the same pre-calculated data
    print("\n  Creating GIF animation...")
    gif_start = time.time()
    
    arrow_scale = max(global_arrow_max * 0.25, 1e-6)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Setup first frame using common function
    im, quiv = setup_slice_plot(ax, all_slice_data[0], global_vmin, global_vmax, 
                                arrow_scale, global_arrow_max, slice_index, timesteps[0])

    # Add colorbars
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')

    cbar2 = fig.colorbar(quiv, ax=ax, shrink=0.6, pad=0.1)
    cbar2.set_label('Field magnitude |R1,R3|')

    def animate(frame):
        slice_data = all_slice_data[frame]
        timestep = timesteps[frame]
        
        monopole_field = slice_data['monopole_field']
        field1_plot = slice_data['field1_plot']
        field3_plot = slice_data['field3_plot']
        field_magnitude = slice_data['field_magnitude']

        # Update plots
        im.set_data(monopole_field.T)
        im.set_clim(vmin=global_vmin, vmax=global_vmax)
        quiv.set_UVC(field1_plot.T, field3_plot.T)
        quiv.set_array(field_magnitude.T.flatten())
        quiv.set_clim(0, global_arrow_max)
        ax.set_title(f'Monopole Field Evolution (y={slice_index}) - t={timestep}\nMonopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}')
        return [im, quiv]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(all_slice_data), 
                                 interval=200, blit=True, repeat=True)

    gif_path = OUTPUT_DIR / 'monopole_field_evolution_xz_slice.gif'
    print(f"    Saving GIF to: {gif_path}")

    try:
        anim.save(gif_path, writer='pillow', fps=8, dpi=100)
        print(f"    ✓ Successfully saved GIF: monopole_field_evolution_xz_slice.gif")
    except Exception as e:
        print(f"    Error saving GIF: {e}")
        print("    Note: Make sure Pillow is installed: pip install Pillow")

    plt.close(fig)
    print(f"  GIF creation completed in {time.time() - gif_start:.2f}s")
    
    total_time = time.time() - load_start
    print(f"  Total analysis time: {total_time:.2f}s")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("MONOPOLE FIELD ANALYSIS WITH ENERGY EVOLUTION")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 4  # Reduced from 5 since we combined steps 4 and 5
    
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
        print_progress(4, total_steps, "Creating field plots and GIF animation...")
        analyze_and_create_all_outputs(r_values_files)  # Single function does everything
    else:
        print("Skipping R-field analysis (no R-values files found)")
    
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)