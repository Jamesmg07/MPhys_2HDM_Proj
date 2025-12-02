import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.animation as animation
import time  # Add timing




# Data directories  
DATA_DIR = Path("/share/centaurus_nas/jmg_temp/pi3_512_long/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp/pi3_512_long/")



# Default simulation parameters (fallback values)
DEFAULT_PARAMS = {
    'nx': 0, 'ny': 0, 'nz': 0,
    'dx': 0.5, 'dy': 0.5, 'dz': 0.5,
    'dt': 0.1, 'nt': 320,
    'gamma_mult': 0.495, 'offset_from_centre': 0.25,
    'n_samples': 10
}

def load_simulation_parameters(data_dir):
    """Load simulation parameters from C++ generated file"""
    
    # Look for parameter files in the data directory
    param_files = list(data_dir.glob("simulation_parameters_*.txt"))
    
    if not param_files:
        print("Warning: No simulation parameters file found. Using default values.")
        return DEFAULT_PARAMS
    
    # Use the most recent parameter file if multiple exist
    param_file = max(param_files, key=lambda p: p.stat().st_mtime)
    
    print(f"Loading parameters from: {param_file.name}")
    
    params = DEFAULT_PARAMS.copy()
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert to appropriate type
                        if key in ['nx', 'ny', 'nz', 'nt', 'seed', 'sep_SaveFreq', 'R_saveFreq']:
                            params[key] = int(value)
                        elif key in ['dx', 'dy', 'dz', 'dt', 'gamma_mult', 'offset_from_centre',
                                   'monopole1_vx', 'monopole1_vy', 'monopole1_vz',
                                   'monopole2_vx', 'monopole2_vy', 'monopole2_vz',
                                   'monopole1_x', 'monopole1_y', 'monopole1_z',
                                   'monopole2_x', 'monopole2_y', 'monopole2_z']:
                            params[key] = float(value)
                        else:
                            params[key] = value  # Keep as string
        
        print(f"Successfully loaded parameters:")
        print(f"  Grid: {params['nx']}×{params['ny']}×{params['nz']}")
        print(f"  Timesteps: {params['nt']}, dt: {params['dt']}")
        print(f"  Gamma: {params['gamma_mult']}π")
        print(f"  Output tag: {params.get('outTag', 'unknown')}")
        
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        print("Using default parameters.")
    
    return params


# Load simulation parameters from C++ generated file
PARAMS = load_simulation_parameters(DATA_DIR)

# Extract parameters for global use
nx, ny, nz = PARAMS['nx'], PARAMS['ny'], PARAMS['nz']
dx, dy, dz = PARAMS['dx'], PARAMS['dy'], PARAMS['dz']
dt = PARAMS['dt']
nt = PARAMS['nt']
nPos = nx * ny * nz
gamma_string = PARAMS['gamma_mult']
n_samples = PARAMS.get('n_samples', 10)
offset_from_centre = PARAMS['offset_from_centre']

def get_monopole_positions():
    """Calculate monopole positions from C++ parameters"""
    # Use loaded parameters if available, otherwise calculate
    if all(key in PARAMS for key in ['monopole1_x', 'monopole1_y', 'monopole1_z', 
                                     'monopole2_x', 'monopole2_y', 'monopole2_z']):
        print("Using monopole positions from parameter file")
        monopole1_pos = (int(PARAMS['monopole1_x']), int(PARAMS['monopole1_y']), int(PARAMS['monopole1_z']))
        monopole2_pos = (int(PARAMS['monopole2_x']), int(PARAMS['monopole2_y']), int(PARAMS['monopole2_z']))
        center_pos = ((nx-1)//2, (ny-1)//2, (nz-1)//2)
        
        print(f"  Monopole 1: ({monopole1_pos[0]:.1f}, {monopole1_pos[1]:.1f}, {monopole1_pos[2]:.1f})")
        print(f"  Monopole 2: ({monopole2_pos[0]:.1f}, {monopole2_pos[1]:.1f}, {monopole2_pos[2]:.1f})")
        
        return center_pos, monopole1_pos, monopole2_pos
    else:
        print("Calculating monopole positions from offset parameters")
        # Fallback to calculation
        center_x = (nx - 1) // 2
        center_y = (ny - 1) // 2  
        center_z = (nz - 1) // 2
        
        offset_grid = int(offset_from_centre * nz)
        monopole1_z = center_z + offset_grid
        monopole2_z = center_z - offset_grid
        
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
    # Use outTag from parameters if available
    if 'outTag' in PARAMS:
        pattern = f"energy_{PARAMS['outTag']}.csv"
        files = list(DATA_DIR.glob(pattern))
        if files:
            return files
    
    # Fallback to pattern matching
    files = list(DATA_DIR.glob(f"energy_gamma=*pi_nx={nx}_*.csv"))
    return files

def find_r_values_files_efficient(n_samples):
    """Find R-values files and immediately select a subset for efficiency"""
    # Use outTag from parameters if available
    if 'outTag' in PARAMS:
        pattern = f"R_values_*{PARAMS['outTag']}.csv"
        all_files = list(DATA_DIR.glob(pattern))
    else:
        # Fallback to pattern matching
        pattern = f"R_values__timestep=*gamma=*pi_nx={nx}_*.csv"
        all_files = list(DATA_DIR.glob(pattern))
    
    if len(all_files) == 0:
        print(f"  No R-values files found matching pattern: {pattern}")
        return []
    
    # Sort by timestep (required for proper selection)
    all_files.sort(key=extract_timestep)
    
    print(f"  Found {len(all_files)} total R-values files")
    
    # Select subset immediately
    if len(all_files) <= n_samples:
        print(f"  Using all {len(all_files)} files (requested {n_samples})")
        return all_files
    else:
        # Select evenly spaced files
        indices = np.linspace(0, len(all_files)-1, n_samples, dtype=int)
        selected_files = [all_files[i] for i in indices]
        print(f"  Selected {len(selected_files)} evenly spaced files from {len(all_files)} total")
        return selected_files

def find_monopole_tracking_files():
    """Find monopole tracking files from the simulation"""
    # Use outTag from parameters if available
    if 'outTag' in PARAMS:
        pattern = f"monopole_tracking_{PARAMS['outTag']}.csv"
        files = list(DATA_DIR.glob(pattern))
        if files:
            return files
    
    # Fallback to pattern matching
    files = list(DATA_DIR.glob(f"monopole_tracking_*nx={nx}_*.csv"))
    return files

def load_energy_data(filepath):
    """Load energy data from file"""
    try:
        data = pd.read_csv(filepath)
        
        # Check if 'Energy' column exists (case-insensitive)
        energy_col = None
        for col in data.columns:
            if col.strip().lower() == 'energy':
                energy_col = col
                break
        
        if energy_col is None:
            print(f"Error: No 'Energy' column found in {filepath}")
            print(f"Available columns: {data.columns.tolist()}")
            return None
        
        return data[energy_col].values
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
        # Vacuum energy formula: E_vac = (1/8) * V where V is the volume
        # Volume = (nx-2)*dx * (ny-2)*dy * (nz-2)*dz for interior points
        vacuum_energy = (1.0/8.0) * ((nx-2) * dx) * ((ny-2) * dy) * ((nz-2) * dz)
        
        # Subtract vacuum energy from all values
        energy_data_corrected = energy_data - vacuum_energy
        
        # Create time array
        timesteps = np.arange(len(energy_data_corrected))
        time_values = timesteps * dt
        
        # Extract gamma value from filename for title
        gamma_match = re.search(r'gamma=([^_]+)', energy_file.name)
        gamma_str = gamma_match.group(1) if gamma_match else str(gamma_string)
        
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
        if abs(energy_data_corrected[0]) > 1e-10:
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
        gamma_str = gamma_match.group(1) if gamma_match else str(gamma_string)
        
        # Calculate separation for each timestep
        separations = []
        valid_timesteps = []
        
        for _, row in tracking_data.iterrows():
            # Check if both monopoles were found (coordinates are not -1)
            if (row['x1_center'] != -1 and row['y1_center'] != -1 and row['z1_center'] != -1 and
                row['x2_center'] != -1 and row['y2_center'] != -1 and row['z2_center'] != -1):
                
                # Calculate 3D distance
                dx_sep = row['x2_center'] - row['x1_center']
                dy_sep = row['y2_center'] - row['y1_center'] 
                dz_sep = row['z2_center'] - row['z1_center']
                separation = np.sqrt(dx_sep**2 + dy_sep**2 + dz_sep**2)
                
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

def load_r_field_data_slices(filepath):
    """Load R-field data from slice-format CSV file"""
    try:
        # Read the CSV file
        data = pd.read_csv(filepath)
        
        # Debug: print column names and first few rows
        if len(data) == 0:
            print(f"Warning: Empty data file {filepath}")
            return None
        
        # Check if this is slice-format data (has slice_type column)
        if 'slice_type' not in data.columns:
            print(f"Warning: No 'slice_type' column found in {filepath}")
            print(f"Columns found: {data.columns.tolist()}")
            # Fall back to old format
            return load_r_field_data(filepath)
        
        # Separate XY and XZ slices
        xy_data = data[data['slice_type'] == 'xy'].copy()
        xz_data = data[data['slice_type'] == 'xz'].copy()
        
        r_fields = {'xy': {}, 'xz': {}}
        
        # Process XY slice (varies in x and y, fixed z)
        if len(xy_data) > 0:
            # Get the fixed z coordinate
            center_z = int(xy_data['k'].iloc[0])
            
            # Initialize 3D arrays (will only fill one slice)
            for col in ['R1nt', 'R2nt', 'R3nt']:
                r_fields['xy'][col] = np.zeros((nx, ny, nz))
            
            # Fill the XY slice
            for _, row in xy_data.iterrows():
                i_coord = int(row['i'])
                j_coord = int(row['j'])
                k_coord = int(row['k'])
                
                # Add bounds checking
                if 0 <= i_coord < nx and 0 <= j_coord < ny and 0 <= k_coord < nz:
                    r_fields['xy']['R1nt'][i_coord, j_coord, k_coord] = row['R1nt']
                    r_fields['xy']['R2nt'][i_coord, j_coord, k_coord] = row['R2nt']
                    r_fields['xy']['R3nt'][i_coord, j_coord, k_coord] = row['R3nt']
            
            r_fields['xy']['slice_index'] = center_z
        else:
            print(f"Warning: No XY slice data found in {filepath}")
        
        # Process XZ slice (varies in x and z, fixed y)
        if len(xz_data) > 0:
            # Get the fixed y coordinate
            center_y = int(xz_data['j'].iloc[0])
            
            # Initialize 3D arrays (will only fill one slice)
            for col in ['R1nt', 'R2nt', 'R3nt']:
                r_fields['xz'][col] = np.zeros((nx, ny, nz))
            
            # Fill the XZ slice
            for _, row in xz_data.iterrows():
                i_coord = int(row['i'])
                j_coord = int(row['j'])
                k_coord = int(row['k'])
                
                # Add bounds checking
                if 0 <= i_coord < nx and 0 <= j_coord < ny and 0 <= k_coord < nz:
                    r_fields['xz']['R1nt'][i_coord, j_coord, k_coord] = row['R1nt']
                    r_fields['xz']['R2nt'][i_coord, j_coord, k_coord] = row['R2nt']
                    r_fields['xz']['R3nt'][i_coord, j_coord, k_coord] = row['R3nt']
            
            r_fields['xz']['slice_index'] = center_y
        else:
            print(f"Warning: No XZ slice data found in {filepath}")
        
        # Verify we have data
        if (len(xy_data) == 0 and len(xz_data) == 0):
            print(f"Error: No valid slice data found in {filepath}")
            return None
        
        return r_fields
            
    except Exception as e:
        print(f"Error loading slice data from {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_r_field_data(filepath):
    """Load R-field data and reshape to 3D grid - optimized version (for full grid data)"""
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
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R3) vectors (y={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Monopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}, γ = {gamma_string}π')
        vector_label = 'Field magnitude |R1,R3|'
    elif slice_type == 'xy':
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        ax.set_title(f'Monopole Field (R1² + R2² + R3²) + (R1,R2) vectors (z={slice_index}) - t={timestep}\nGrid: {nx}×{ny}×{nz}, Monopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}, γ = {gamma_string}π')
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

def analyze_and_create_all_outputs(selected_files):
    """Create both individual plots and GIF - now receives pre-selected files"""
    print(f"\nAnalyzing field data and creating all outputs...")
    
    print(f"  Using {len(selected_files)} pre-selected timesteps")

    # Remove the file selection logic since files are already selected
    # selected_files is now passed in directly
    
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
        
        # Try to load as slice-format data first
        r_fields_all = load_r_field_data_slices(file)
        
        if r_fields_all and isinstance(r_fields_all, dict) and 'xy' in r_fields_all and 'xz' in r_fields_all:
            # Slice-format data loaded successfully
            r_fields_xz = r_fields_all['xz']
            r_fields_xy = r_fields_all['xy']
            
            # Get the actual slice indices from the data
            slice_index_xz_actual = r_fields_xz.get('slice_index', slice_index_xz)
            slice_index_xy_actual = r_fields_xy.get('slice_index', slice_index_xy)
            
            # Check if we have valid data
            if all(key in r_fields_xz for key in ['R1nt', 'R2nt', 'R3nt']) and \
               all(key in r_fields_xy for key in ['R1nt', 'R2nt', 'R3nt']):
                
                # Calculate slice data for both slices
                slice_data_xz = prepare_slice_data(r_fields_xz, slice_index_xz_actual, 'xz')
                slice_data_xy = prepare_slice_data(r_fields_xy, slice_index_xy_actual, 'xy')
                
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
            # Try old format (full grid data)
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
        ax_xz.set_title(f'Monopole Field Evolution (y={slice_index_xz}) - t={timestep}\nMonopole positions: z={monopole1_pos[2]:.0f}, z={monopole2_pos[2]:.0f}, γ = {gamma_string}π')
        return [im_xz, quiv_xz]

    # Create XZ animation
    anim_xz = animation.FuncAnimation(fig_xz, animate_xz, frames=len(all_slice_data_xz), 
                                     interval=200, blit=True, repeat=True)

    gif_path_xz = OUTPUT_DIR / f'{gamma_string}pi_monopole_field_evolution_xz_slice.gif'
    print(f"    Saving XZ GIF to: {gamma_string}pi_{gif_path_xz}")

    try:
        anim_xz.save(gif_path_xz, writer='pillow', fps=8, dpi=100)
        print(f"    ✓ Successfully saved XZ GIF: {gamma_string}pi_monopole_field_evolution_xz_slice.gif")
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

    gif_path_xy = OUTPUT_DIR / f'{gamma_string}pi_monopole_field_evolution_xy_slice.gif'
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
    
    total_steps = 5
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Simulation parameters: Grid={nx}×{ny}×{nz}, dt={dt}, Expected timesteps={nt}")
    print(f"Gamma parameter: {gamma_string}π")
    print(f"Using {n_samples} samples for field analysis")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Searching for data files...")
    
    # Find files - use efficient version for R-values
    energy_files = find_energy_files()
    r_values_files = find_r_values_files_efficient(n_samples)  # Select 10 files immediately
    monopole_tracking_files = find_monopole_tracking_files()
    
    print(f"Found {len(energy_files)} energy files")
    print(f"Selected {len(r_values_files)} R-values files for analysis")
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
        analyze_and_create_all_outputs(r_values_files)  # Pass pre-selected files
    else:
        print("Skipping R-field analysis (no R-values files found)")
    
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)