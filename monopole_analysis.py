import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.animation as animation
import time  # Add timing

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_3/Monopole_0495pi/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp")
nx, ny, nz = 128,128,128  # Grid dimensions
dx, dy, dz = 0.5, 0.5, 0.5
dt = 0.1  # Simulation timestep
nPos = nx * ny * nz
nt = int((nx * dx) / (2 * dt))
gamma_string = 0.495

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
    # Updated pattern to match new C++ naming: energy_gamma=X.Xpi_nx64_sep32_nt1280_seed73_monopole.csv
    files = list(DATA_DIR.glob(f"energy_gamma=*pi_nx={nx}_*.csv"))
    return files

def find_r_values_files():
    """Find all R-values files from the simulation"""
    # Updated pattern to match new C++ naming: R_values__timestep=X_gamma=X.Xpi_nx64_sep32_nt1280_seed73_monopole.csv
    files = list(DATA_DIR.glob(f"R_values__timestep=*gamma=*pi_nx={nx}_*.csv"))
    files.sort(key=extract_timestep)
    return files

def find_monopole_tracking_files():
    """Find monopole tracking files from the simulation"""
    # Pattern: monopole_tracking_gamma=X.Xpi_nx64_sep32_nt1280_seed73_monopole.csv
    files = list(DATA_DIR.glob(f"monopole_tracking_*nx={nx}_*.csv"))
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

def load_monopole_tracking_data(filepath):
    """Load monopole tracking data from file"""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"Error loading monopole tracking data from {filepath}: {e}")
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
        
        # Calculate vacuum energy to subtract
        # Using physical volume: (nx-1)*dx * (ny-1)*dy * (nz-1)*dz for length-based calculation
        # or nx*dx * ny*dy * nz*dz for grid-based calculation
        vacuum_energy = (1/8) * ((nx-2) * dx)**3  # Using gridsize-1 as you suggested
        
        # Subtract vacuum energy from all values
        energy_data_corrected = energy_data + vacuum_energy
        
        # Create time array
        timesteps = np.arange(len(energy_data_corrected))
        time_values = timesteps * dt  # Convert to physical time
        
        # Extract gamma value from filename for title
        gamma_match = re.search(r'gamma=([^_]+)', energy_file.name)
        gamma_str = gamma_match.group(1) if gamma_match else "unknown"
        
        plt.figure(figsize=(12, 8))
        
        # Single plot of energy vs time
        plt.plot(time_values, energy_data_corrected, 'b-', linewidth=2, marker='o', markersize=4,
                label='Total Energy (vacuum subtracted)')
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title(f'Energy Evolution (γ = {gamma_str}) - Vacuum Energy Subtracted')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add simulation info as text box
        info_text = (f'Grid: {nx}×{ny}×{nz}, dt = {dt}\n'
                    f'Total steps = {len(energy_data_corrected)}\n'
                    f'Vacuum energy subtracted: {vacuum_energy:.6f}')
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        save_path = OUTPUT_DIR / f'energy_evolution_gamma_{gamma_str}.png'
        save_and_close_plot(save_path, f"    Saved: energy_evolution_gamma_{gamma_str}.png")
        
        # Print energy statistics
        print(f"    Energy statistics for γ = {gamma_str}:")
        print(f"      Vacuum energy subtracted: {vacuum_energy:.6f}")
        print(f"      Initial energy (corrected): {energy_data_corrected[0]:.6f}")
        print(f"      Final energy (corrected): {energy_data_corrected[-1]:.6f}")
        print(f"      Energy change: {energy_data_corrected[-1] - energy_data_corrected[0]:.6f}")
        print(f"      Energy conservation: {abs(energy_data_corrected[-1] - energy_data_corrected[0])/abs(energy_data_corrected[0])*100:.4f}%")

def plot_monopole_separation(tracking_files):
    """Plot monopole separation over time"""
    print("\nAnalyzing monopole separation...")
    
    if not tracking_files:
        print("  No monopole tracking files found!")
        return
    
    for tracking_file in tracking_files:
        print(f"  Loading monopole tracking data from: {tracking_file.name}")
        
        tracking_data = load_monopole_tracking_data(tracking_file)
        
        if tracking_data is None or len(tracking_data) == 0:
            print(f"    Error: Could not load monopole tracking data from {tracking_file}")
            continue
        
        # Extract gamma value from filename for title
        gamma_match = re.search(r'gamma=([^_]+)', tracking_file.name)
        gamma_str = gamma_match.group(1) if gamma_match else "unknown"
        
        # Calculate separation for each timestep
        separations = []
        valid_timesteps = []
        
        for _, row in tracking_data.iterrows():
            # Check if both monopoles were found (coordinates are not -1)
            if (row['x1_center'] != -1 and row['y1_center'] != -1 and row['z1_center'] != -1 and
                row['x2_center'] != -1 and row['y2_center'] != -1 and row['z2_center'] != -1):
                
                # Calculate 3D distance
                dx = row['x2_center'] - row['x1_center']
                dy = row['y2_center'] - row['y1_center'] 
                dz = row['z2_center'] - row['z1_center']
                separation = np.sqrt(dx**2 + dy**2 + dz**2)
                
                separations.append(separation)
                valid_timesteps.append(row['timestep'])
        
        if len(separations) == 0:
            print(f"    Warning: No valid monopole pairs found in {tracking_file.name}")
            continue
        
        # Convert to numpy arrays for plotting
        valid_timesteps = np.array(valid_timesteps)
        separations = np.array(separations)
        time_values = valid_timesteps * dt  # Convert to physical time
        
        # Calculate initial separation for reference
        initial_separation = separations[0] if len(separations) > 0 else 0
        
        plt.figure(figsize=(12, 8))
        
        # Single plot of separation vs time
        plt.plot(time_values, separations, 'g-', linewidth=2, marker='o', markersize=4, 
                label='Monopole-Antimonopole Separation')
        plt.axhline(y=initial_separation, color='r', linestyle='--', alpha=0.7, 
                   label=f'Initial separation = {initial_separation:.2f}')
        plt.xlabel('Time')
        plt.ylabel('Separation')
        plt.title(f'Monopole-Antimonopole Separation Evolution (γ = {gamma_str})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add simulation info as text box
        info_text = (f'Grid: {nx}×{ny}×{nz}, dt = {dt}\n'
                    f'Valid data points: {len(separations)}/{len(tracking_data)} timesteps\n'
                    f'Final separation: {separations[-1]:.2f}')
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        save_path = OUTPUT_DIR / f'monopole_separation_gamma_{gamma_str}.png'
        save_and_close_plot(save_path, f"    Saved: monopole_separation_gamma_{gamma_str}.png")
        
        # Print separation statistics
        print(f"    Separation statistics for γ = {gamma_str}:")
        print(f"      Valid timesteps: {len(separations)}/{len(tracking_data)}")
        print(f"      Initial separation: {initial_separation:.6f}")
        print(f"      Final separation: {separations[-1]:.6f}")
        print(f"      Maximum separation: {np.max(separations):.6f}")
        print(f"      Minimum separation: {np.min(separations):.6f}")
        print(f"      Average separation: {np.mean(separations):.6f}")
        if len(separations) > 1:
            separation_change = separations[-1] - separations[0]
            print(f"      Total change: {separation_change:.6f}")
            print(f"      Relative change: {(separation_change/initial_separation)*100:.2f}%")

def load_r_field_data(filepath):
    """Load R-field data and reshape to 3D grid - optimized version"""
    try:
        # Use numpy for faster loading, skip header row
        data = np.loadtxt(filepath, skiprows=1)
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
            data = pd.read_csv(filepath, sep=None, engine='python')  # Auto-detect separator
            r_fields = {}
            
            for col in ['R1nt', 'R2nt', 'R3nt']:
                if col in data.columns:
                    field_1d = data[col].values
                    if len(field_1d) == nPos:
                        r_fields[col] = field_1d.reshape(nx, ny, nz)
            
            return r_fields
        except Exception as e2:
            print(f"Error loading {filepath}: numpy error: {e}, pandas error: {e2}")
            return None

def save_and_close_plot(save_path, message):
    """Helper function to save and close plots"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(message)
    plt.close()

def prepare_slice_data(r_fields, slice_index, slice_type):
    """Prepare slice data for plotting - common function for both individual plots and GIF"""
    if slice_type == 'xz':
        # Extract slices (x-z plane at fixed y where monopoles are located)
        field1_slice = r_fields['R1nt'][:, slice_index, :]  # R1
        field2_slice = r_fields['R2nt'][:, slice_index, :]  # R2
        field3_slice = r_fields['R3nt'][:, slice_index, :]  # R3
        
        # Create coordinate grids - z as vertical (y-axis), x as horizontal (x-axis)
        coord1 = np.arange(nx) * dx  # x coordinates
        coord2 = np.arange(nz) * dz  # z coordinates
        
        # Subsample field components for arrows (R1, R3)
        field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
        field3_sub = field3_slice[::ARROW_STEP, ::ARROW_STEP]
        
        # Calculate field magnitude for color coding (R1, R3)
        field_magnitude = np.sqrt(field1_sub**2 + field3_sub**2)
        
        # Don't normalize direction - show actual field strength, but make arrows visible
        mask = field_magnitude > 1e-10
        field1_plot = np.where(mask, field1_sub, 0)
        field2_plot = np.where(mask, field3_sub, 0)  # Using R3 for xz slice
        
    elif slice_type == 'xy':
        # Extract slices (x-y plane at fixed z)
        field1_slice = r_fields['R1nt'][:, :, slice_index]  # R1
        field2_slice = r_fields['R2nt'][:, :, slice_index]  # R2
        field3_slice = r_fields['R3nt'][:, :, slice_index]  # R3
        
        # Create coordinate grids - y as vertical (y-axis), x as horizontal (x-axis)
        coord1 = np.arange(nx) * dx  # x coordinates
        coord2 = np.arange(ny) * dy  # y coordinates
        
        # Subsample field components for arrows (R1, R2)
        field1_sub = field1_slice[::ARROW_STEP, ::ARROW_STEP]
        field2_sub = field2_slice[::ARROW_STEP, ::ARROW_STEP]
        
        # Calculate field magnitude for color coding (R1, R2)
        field_magnitude = np.sqrt(field1_sub**2 + field2_sub**2)
        
        # Don't normalize direction - show actual field strength, but make arrows visible
        mask = field_magnitude > 1e-10
        field1_plot = np.where(mask, field1_sub, 0)
        field2_plot = np.where(mask, field2_sub, 0)  # Using R2 for xy slice
        
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}. Must be 'xz' or 'xy'")
    
    # Calculate R1² + R2² + R3² for monopole identification (common to both slice types)
    monopole_field = field1_slice**2 + field2_slice**2 + field3_slice**2
    
    return {
        'monopole_field': monopole_field,
        'field1_plot': field1_plot,
        'field2_plot': field2_plot,
        'field_magnitude': field_magnitude,
        'coord1': coord1,
        'coord2': coord2,
        'slice_type': slice_type
    }

def setup_slice_plot(ax, slice_data, vmin, vmax, arrow_scale, global_arrow_max, slice_index, timestep):
    """Setup the slice plot with common styling - used by both individual plots and GIF"""
    monopole_field = slice_data['monopole_field']
    field1_plot = slice_data['field1_plot']
    field2_plot = slice_data['field2_plot']
    field_magnitude = slice_data['field_magnitude']
    coord1 = slice_data['coord1']
    coord2 = slice_data['coord2']
    slice_type = slice_data['slice_type']
    
    # Create subsampled coordinate arrays for arrows
    coord1_sub = coord1[::ARROW_STEP]
    coord2_sub = coord2[::ARROW_STEP]
    X_sub, Y_sub = np.meshgrid(coord1_sub, coord2_sub)
    
    # Background field plot
    im = ax.imshow(monopole_field.T, origin='lower', 
                   extent=[coord1[0], coord1[-1], coord2[0], coord2[-1]],
                   aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
    
    # Arrow plot
    quiv = ax.quiver(X_sub, Y_sub, field1_plot.T, field2_plot.T,
                     field_magnitude.T, scale=arrow_scale, width=ARROW_WIDTH, 
                     alpha=0.9, cmap='Reds', scale_units='xy',
                     clim=(0, global_arrow_max))
    
    # Labels and title based on slice type
    if slice_type == 'xz':
        ax.set_xlabel('x position')
        ax.set_ylabel('z position')
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Monopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}, γ = {gamma_string}pi')
        vector_label = 'Field magnitude |R1,R3|'
    elif slice_type == 'xy':
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R2) vectors (z={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Monopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}, γ = {gamma_str}pi')
        vector_label = 'Field magnitude |R1,R2|'
    
    ax.set_xlim(coord1[0], coord1[-1])
    ax.set_ylim(coord2[0], coord2[-1])
    
    return im, quiv, vector_label

def create_individual_plot(slice_data, timestep, vmin, vmax, global_arrow_max, slice_index):
    """Create and save individual plot from pre-calculated slice data"""
    arrow_scale = max(global_arrow_max * 0.25, 1e-6)
    slice_type = slice_data['slice_type']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Setup plot using common function
    im, quiv, vector_label = setup_slice_plot(ax, slice_data, vmin, vmax, arrow_scale, global_arrow_max, slice_index, timestep)
    
    # Add colorbars
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R1² + R2² + R3² (monopole field strength)')
    
    cbar2 = plt.colorbar(quiv, ax=ax, shrink=0.6, pad=0.1)
    cbar2.set_label(vector_label)
    
    # Save and close with appropriate filename
    filename = f'γ = {gamma_string}pi_{slice_type}_monopole_field_t{timestep}.png'
    save_and_close_plot(OUTPUT_DIR / filename, f"    Saved: {filename}")

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

    # SINGLE PASS: Load all data and calculate everything once for both slice types
    all_slice_data_xz = []
    all_slice_data_xy = []
    timesteps = []
    global_vmax_xz = 0
    global_vmin_xz = float('inf')
    global_arrow_max_xz = 0
    global_vmax_xy = 0
    global_vmin_xy = float('inf')
    global_arrow_max_xy = 0
    
    slice_index_xz = int(MONOPOLE_CENTER_Y)  # y slice for xz plot
    slice_index_xy = int((nz - 1) // 2)      # z=0 slice for xy plot (center of z)

    print("  Loading and processing all data (single pass)...")
    load_start = time.time()
    
    for i, file in enumerate(selected_files):
        print(f"    [{i+1}/{len(selected_files)}] Processing file for timestep {extract_timestep(file)}...")

        timestep = extract_timestep(file)
        r_fields = load_r_field_data(file)

        if r_fields and all(key in r_fields for key in ['R1nt', 'R2nt', 'R3nt']):
            # Calculate slice data once for both slice types
            slice_data_xz = prepare_slice_data(r_fields, slice_index_xz, 'xz')
            slice_data_xy = prepare_slice_data(r_fields, slice_index_xy, 'xy')
            
            all_slice_data_xz.append(slice_data_xz)
            all_slice_data_xy.append(slice_data_xy)
            timesteps.append(timestep)
            
            # XZ slice statistics
            monopole_field_xz = slice_data_xz['monopole_field']
            field_magnitude_xz = slice_data_xz['field_magnitude']
            
            # XY slice statistics
            monopole_field_xy = slice_data_xy['monopole_field']
            field_magnitude_xy = slice_data_xy['field_magnitude']
            
            # Debug: Print field statistics
            print(f"      XZ slice stats: min={np.min(monopole_field_xz):.6f}, max={np.max(monopole_field_xz):.6f}")
            print(f"      XY slice stats: min={np.min(monopole_field_xy):.6f}, max={np.max(monopole_field_xy):.6f}")
            
            # Update global scaling parameters for XZ
            frame_vmax_xz = np.percentile(monopole_field_xz, 99)
            frame_vmin_xz = np.min(monopole_field_xz)
            frame_arrow_max_xz = np.max(field_magnitude_xz)
            
            global_vmax_xz = max(global_vmax_xz, frame_vmax_xz)
            global_vmin_xz = min(global_vmin_xz, frame_vmin_xz)
            global_arrow_max_xz = max(global_arrow_max_xz, frame_arrow_max_xz)
            
            # Update global scaling parameters for XY
            frame_vmax_xy = np.percentile(monopole_field_xy, 99)
            frame_vmin_xy = np.min(monopole_field_xy)
            frame_arrow_max_xy = np.max(field_magnitude_xy)
            
            global_vmax_xy = max(global_vmax_xy, frame_vmax_xy)
            global_vmin_xy = min(global_vmin_xy, frame_vmin_xy)
            global_arrow_max_xy = max(global_arrow_max_xy, frame_arrow_max_xy)
        else:
            print(f"      Skipped: Invalid data for timestep {timestep}")

    print(f"  Data processing completed in {time.time() - load_start:.2f}s")

    if len(all_slice_data_xz) == 0:
        print("  Error: No valid data found!")
        return

    print(f"  Successfully processed {len(all_slice_data_xz)} timesteps")
    print(f"  XZ Global scaling - vmin: {global_vmin_xz:.6f}, vmax: {global_vmax_xz:.6f}, arrow_max: {global_arrow_max_xz:.6f}")
    print(f"  XY Global scaling - vmin: {global_vmin_xy:.6f}, vmax: {global_vmax_xy:.6f}, arrow_max: {global_arrow_max_xy:.6f}")

    # CREATE INDIVIDUAL PLOTS using pre-calculated data for both slice types
    print("\n  Creating individual XZ plots...")
    individual_start = time.time()
    
    for i, (slice_data, timestep) in enumerate(zip(all_slice_data_xz, timesteps)):
        print(f"    [{i+1}/{len(all_slice_data_xz)}] Creating XZ plot for timestep {timestep}...")
        create_individual_plot(slice_data, timestep, global_vmin_xz, global_vmax_xz, global_arrow_max_xz, slice_index_xz)
    
    print("\n  Creating individual XY plots...")
    
    for i, (slice_data, timestep) in enumerate(zip(all_slice_data_xy, timesteps)):
        print(f"    [{i+1}/{len(all_slice_data_xy)}] Creating XY plot for timestep {timestep}...")
        create_individual_plot(slice_data, timestep, global_vmin_xy, global_vmax_xy, global_arrow_max_xy, slice_index_xy)
    
    print(f"  Individual plots completed in {time.time() - individual_start:.2f}s")

    # CREATE GIF ANIMATIONS using the same pre-calculated data for both slice types
    print("\n  Creating XZ GIF animation...")
    gif_start = time.time()
    
    # XZ GIF
    arrow_scale_xz = max(global_arrow_max_xz * 0.25, 1e-6)
    fig_xz, ax_xz = plt.subplots(1, 1, figsize=(12, 8))

    # Setup first frame using common function
    im_xz, quiv_xz, vector_label_xz = setup_slice_plot(ax_xz, all_slice_data_xz[0], global_vmin_xz, global_vmax_xz, 
                                                       arrow_scale_xz, global_arrow_max_xz, slice_index_xz, timesteps[0])

    # Add colorbars
    cbar_xz = fig_xz.colorbar(im_xz, ax=ax_xz)
    cbar_xz.set_label('R1² + R2² + R3² (monopole field strength)')

    cbar2_xz = fig_xz.colorbar(quiv_xz, ax=ax_xz, shrink=0.6, pad=0.1)
    cbar2_xz.set_label(vector_label_xz)

    def animate_xz(frame):
        slice_data = all_slice_data_xz[frame]
        timestep = timesteps[frame]
        
        monopole_field = slice_data['monopole_field']
        field1_plot = slice_data['field1_plot']
        field2_plot = slice_data['field2_plot']
        field_magnitude = slice_data['field_magnitude']

        # Update plots
        im_xz.set_data(monopole_field.T)
        im_xz.set_clim(vmin=global_vmin_xz, vmax=global_vmax_xz)
        quiv_xz.set_UVC(field1_plot.T, field2_plot.T)
        quiv_xz.set_array(field_magnitude.T.flatten())
        quiv_xz.set_clim(0, global_arrow_max_xz)
        ax_xz.set_title(f'Monopole Field Evolution (y={slice_index_xz}) - t={timestep}\nMonopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}')
        return [im_xz, quiv_xz]

    # Create XZ animation
    anim_xz = animation.FuncAnimation(fig_xz, animate_xz, frames=len(all_slice_data_xz), 
                                     interval=200, blit=True, repeat=True)

    gif_path_xz = OUTPUT_DIR / 'monopole_field_evolution_xz_slice.gif'
    print(f"    Saving XZ GIF to: {gamma_string}_{gif_path_xz}")

    try:
        anim_xz.save(gif_path_xz, writer='pillow', fps=8, dpi=100)
        print(f"    ✓ Successfully saved XZ GIF: monopole_field_evolution_xz_slice.gif")
    except Exception as e:
        print(f"    Error saving XZ GIF: {e}")

    plt.close(fig_xz)
    
    # XY GIF
    print("\n  Creating XY GIF animation...")
    arrow_scale_xy = max(global_arrow_max_xy * 0.25, 1e-6)
    fig_xy, ax_xy = plt.subplots(1, 1, figsize=(12, 8))

    # Setup first frame using common function
    im_xy, quiv_xy, vector_label_xy = setup_slice_plot(ax_xy, all_slice_data_xy[0], global_vmin_xy, global_vmax_xy, 
                                                       arrow_scale_xy, global_arrow_max_xy, slice_index_xy, timesteps[0])

    # Add colorbars
    cbar_xy = fig_xy.colorbar(im_xy, ax=ax_xy)
    cbar_xy.set_label('R1² + R2² + R3² (monopole field strength)')

    cbar2_xy = fig_xy.colorbar(quiv_xy, ax=ax_xy, shrink=0.6, pad=0.1)
    cbar2_xy.set_label(vector_label_xy)

    def animate_xy(frame):
        slice_data = all_slice_data_xy[frame]
        timestep = timesteps[frame]
        
        monopole_field = slice_data['monopole_field']
        field1_plot = slice_data['field1_plot']
        field2_plot = slice_data['field2_plot']
        field_magnitude = slice_data['field_magnitude']

        # Update plots
        im_xy.set_data(monopole_field.T)
        im_xy.set_clim(vmin=global_vmin_xy, vmax=global_vmax_xy)
        quiv_xy.set_UVC(field1_plot.T, field2_plot.T)
        quiv_xy.set_array(field_magnitude.T.flatten())
        quiv_xy.set_clim(0, global_arrow_max_xy)
        ax_xy.set_title(f'Monopole Field Evolution (z={slice_index_xy}) - t={timestep}\nMonopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}')
        return [im_xy, quiv_xy]

    # Create XY animation
    anim_xy = animation.FuncAnimation(fig_xy, animate_xy, frames=len(all_slice_data_xy), 
                                     interval=200, blit=True, repeat=True)

    gif_path_xy = OUTPUT_DIR / 'monopole_field_evolution_xy_slice.gif'
    print(f"    Saving XY GIF to: {gamma_string}_{gif_path_xy}")

    try:
        anim_xy.save(gif_path_xy, writer='pillow', fps=8, dpi=100)
        print(f"    ✓ Successfully saved XY GIF: monopole_field_evolution_xy_slice.gif")
    except Exception as e:
        print(f"    Error saving XY GIF: {e}")

    plt.close(fig_xy)
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
    
    total_steps = 5  # Increased to include monopole separation analysis
    
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
    monopole_tracking_files = find_monopole_tracking_files()
    
    print(f"Found {len(energy_files)} energy files")
    print(f"Found {len(r_values_files)} R-values files")
    print(f"Found {len(monopole_tracking_files)} monopole tracking files")
    
    if not energy_files and not r_values_files and not monopole_tracking_files:
        print("ERROR: No data files found!")
        exit()
    
    print_progress(3, total_steps, "Analyzing energy evolution...")
    plot_energy_vs_time(energy_files)
    
    print_progress(4, total_steps, "Analyzing monopole separation...")
    plot_monopole_separation(monopole_tracking_files)
    
    if r_values_files:
        print_progress(5, total_steps, "Creating field plots and GIF animation...")
        analyze_and_create_all_outputs(r_values_files)  # Single function does everything
    else:
        print("Skipping R-field analysis (no R-values files found)")
    
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)