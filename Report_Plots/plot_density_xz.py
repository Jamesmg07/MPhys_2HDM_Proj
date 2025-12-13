import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import pandas as pd
import re

# Data directories - now supporting two configurations for comparison
DATA_DIR_1 = Path("/share/centaurus_nas/jmg_temp/0_512_energy_density/")
DATA_DIR_2 = Path("/share/centaurus_nas/jmg_temp/pi_512_energy_density/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp/pi_512_energy_density/")

# Default simulation parameters (fallback values)
DEFAULT_PARAMS = {
    'nx': 256, 'ny': 256, 'nz': 256,
    'dx': 0.5, 'dy': 0.5, 'dz': 0.5,
    'separation': 0.25,
    'gamma_mult_1': 0.0,
    'gamma_mult_2': 0.0,
    'seed': 73
}

def load_simulation_parameters(data_dir):
    """Load simulation parameters from C++ generated file"""
    
    # Look for parameter files in the data directory
    param_files = list(data_dir.glob("simulation_parameters*.txt"))
    
    if not param_files:
        print(f"Warning: No simulation parameters file found in {data_dir}.")
        print("Trying to extract parameters from CSV filenames...")
        # Try to get nx from CSV filenames
        csv_files = list(data_dir.glob("energy_density_xzslice_*.csv"))
        if csv_files:
            # Extract nx from first CSV filename
            for csv_file in csv_files:
                nx_match = re.search(r'_nx=(\d+)_', csv_file.name)
                if nx_match:
                    nx_from_file = int(nx_match.group(1))
                    params = DEFAULT_PARAMS.copy()
                    params['nx'] = nx_from_file
                    params['ny'] = nx_from_file
                    params['nz'] = nx_from_file
                    print(f"Extracted grid size from filename: {nx_from_file}x{nx_from_file}x{nx_from_file}")
                    return params
        print("Using default values.")
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
                        if key in ['nx', 'ny', 'nz', 'seed']:
                            params[key] = int(value)
                        elif key in ['dx', 'dy', 'dz', 'separation', 'gamma_mult_1', 'gamma_mult_2']:
                            params[key] = float(value)
                        else:
                            params[key] = value  # Keep as string
        
        print(f"Successfully loaded parameters:")
        print(f"  Grid: {params['nx']}×{params['ny']}×{params['nz']}")
        print(f"  Separation: {params['separation']}")
        print(f"  Gamma_1: {params['gamma_mult_1']}π, Gamma_2: {params['gamma_mult_2']}π")
        
    except Exception as e:
        print(f"Error reading parameter file: {e}")
        print("Using default parameters.")
    
    return params

# Load simulation parameters
PARAMS = load_simulation_parameters(DATA_DIR_1)

# Extract parameters for global use
nx, ny, nz = PARAMS['nx'], PARAMS['ny'], PARAMS['nz']
dx, dy, dz = PARAMS['dx'], PARAMS['dy'], PARAMS['dz']

def find_energy_density_files(data_dir):
    """Find energy density files matching the C++ output pattern"""
    # Get nx from the directory's parameters (or filename)
    params = load_simulation_parameters(data_dir)
    nx = params['nx']
    
    # Updated pattern to match C++ output format
    # C++ outputs: energy_density_xzslice_gamma1={gamma1}pi_gamma2={gamma2}pi_nx={nx}_sep={sep}_seed={seed}_monopole.csv
    pattern = f"energy_density_xzslice_gamma1=*pi_gamma2=*pi_nx={nx}_sep=*_seed=*_monopole.csv"
    files = list(data_dir.glob(pattern))
    
    if not files:
        print(f"No energy density files found matching pattern: {pattern}")
        print(f"Looking in directory: {data_dir}")
        # Try a more general pattern
        general_pattern = "energy_density_xzslice_*.csv"
        files = list(data_dir.glob(general_pattern))
        if files:
            print(f"Found {len(files)} files with general pattern: {general_pattern}")
            # Filter files that match the expected structure
            filtered_files = []
            for f in files:
                if re.search(r'gamma1=[\d.]+pi_gamma2=[\d.]+pi_nx=\d+_sep=[\d.]+_seed=\d+_monopole\.csv', f.name):
                    filtered_files.append(f)
            if filtered_files:
                print(f"Filtered to {len(filtered_files)} files matching expected format")
                return filtered_files
            else:
                print("No files match expected naming format")
        else:
            print(f"No CSV files found at all in {data_dir}")
    
    return files

def extract_params_from_filename(filename):
    """Extract parameters from energy density filename"""
    # Extract gamma1, gamma2, separation from filename
    gamma1_match = re.search(r'gamma1=([\d.]+)pi', filename.name)
    gamma2_match = re.search(r'gamma2=([\d.]+)pi', filename.name)
    sep_match = re.search(r'sep=([\d.]+)', filename.name)
    seed_match = re.search(r'seed=(\d+)', filename.name)
    
    if gamma1_match and gamma2_match and sep_match and seed_match:
        return {
            'gamma1': float(gamma1_match.group(1)),
            'gamma2': float(gamma2_match.group(1)),
            'separation': float(sep_match.group(1)),
            'seed': int(seed_match.group(1))
        }
    return None

def read_energy_density(filepath):
    """Read energy density data from CSV file"""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"Error loading energy density from {filepath}: {e}")
        return None

def get_monopole_positions(separation):
    """Calculate monopole positions from separation parameter (matching C++ code)"""
    # From C++ code:
    # z1 = 0.5 * (nz - 1) + separation * nz
    # z2 = 0.5 * (nz - 1) - separation * nz
    # Both monopoles at center in x,y
    
    center_x = 0.5 * (nx - 1) * dx
    center_y = 0.5 * (ny - 1) * dy
    center_z = 0.5 * (nz - 1) * dz
    
    z1 = center_z + separation * nz * dz
    z2 = center_z - separation * nz * dz
    
    return (center_x, center_y, z1), (center_x, center_y, z2)

def analyze_falloff_behavior(x_line_data, z_line_data, mono_x, mono_z1, mono_z2):
    """Analyze the falloff behavior of energy density around monopoles
    
    This function fits the energy density E(r) to power law models in regions OUTSIDE the monopole pair.
    
    Fitting procedure:
    1. Take logarithm: log(E) = log(A) + n*log(r)
    2. Perform linear regression in log-log space using np.polyfit
    3. Extract slope 'n' (power exponent) and intercept 'log(A)'
    4. Reconstruct: E(r) = A * r^n
    
    The exponent 'n' tells us the falloff type:
    - n ≈ -1: Coulomb-like 1/r falloff (monopole field)
    - n ≈ -2: Dipole-like 1/r² falloff
    - Other values: Different field configuration
    """
    print("\n  Analyzing energy density falloff in outer regions...")
    
    # Analyze x-direction falloff (perpendicular to monopole separation axis)
    x_vals = x_line_data['x'].values
    x_energy = x_line_data['energy_density'].values
    
    # Analyze z-direction falloff (along monopole separation axis)
    z_vals = z_line_data['z'].values
    z_energy = z_line_data['energy_density'].values
    
    # === X-DIRECTION: Analyze falloff from center (perpendicular to dipole axis) ===
    x_region = np.abs(x_vals - mono_x)
    valid_region_x = (x_region > 10.0) & (x_region < 50.0)  # Far from monopoles, but not at edge
    
    falloff_data_x = None
    if np.sum(valid_region_x) > 5:
        r_x = x_region[valid_region_x]
        E_x = x_energy[valid_region_x]
        
        # Log-log fit: log(E) vs log(r)
        # This assumes E = A * r^n, so log(E) = log(A) + n*log(r)
        log_r_x = np.log(r_x)
        log_E_x = np.log(np.maximum(E_x, 1e-10))
        
        # Linear fit returns [slope, intercept] = [n, log(A)]
        coeffs_power_x = np.polyfit(log_r_x, log_E_x, 1)
        power_exponent_x = coeffs_power_x[0]  # This is 'n'
        log_A_x = coeffs_power_x[1]
        A_power_x = np.exp(log_A_x)
        
        print(f"    X-direction (perpendicular to dipole, sep > 10):")
        print(f"      Power law fit: E ~ {A_power_x:.6f} * r^{power_exponent_x:.3f}")
        print(f"      Fitting method: Linear regression on log(E) vs log(r)")
        if abs(power_exponent_x + 1.0) < 0.3:
            print(f"      -> Consistent with 1/r (monopole-like)")
        elif abs(power_exponent_x + 2.0) < 0.3:
            print(f"      -> Consistent with 1/r² (dipole-like)")
        elif abs(power_exponent_x + 3.0) < 0.3:
            print(f"      -> Consistent with 1/r³ (quadrupole-like)")
        
        falloff_type_x = "1/r" if abs(power_exponent_x + 1.0) < 0.5 else ("1/r²" if abs(power_exponent_x + 2.0) < 0.5 else f"r^{power_exponent_x:.2f}")
        
        falloff_data_x = {
            'type': falloff_type_x,
            'power_A': A_power_x,
            'power_exp': power_exponent_x,
            'center': mono_x,
            'fit_region': 'perpendicular, sep > 10'
        }
    else:
        falloff_type_x = "Unknown"
        print(f"    X-direction: Insufficient data for falloff analysis")
    
    # === Z-DIRECTION: Analyze falloff OUTSIDE the monopole pair (beyond both monopoles) ===
    # Region 1: Beyond monopole 1 (z > mono_z1)
    beyond_mono1 = z_vals > mono_z1
    z_beyond1 = z_vals[beyond_mono1]
    E_beyond1 = z_energy[beyond_mono1]
    
    # Only use points far enough from monopole 1
    r_beyond1 = np.abs(z_beyond1 - mono_z1)
    valid_beyond1 = (r_beyond1 > 10.0) & (r_beyond1 < 50.0)
    
    falloff_data_beyond1 = None
    if np.sum(valid_beyond1) > 5:
        r1 = r_beyond1[valid_beyond1]
        E1 = E_beyond1[valid_beyond1]
        
        log_r1 = np.log(r1)
        log_E1 = np.log(np.maximum(E1, 1e-10))
        
        coeffs_power1 = np.polyfit(log_r1, log_E1, 1)
        power_exponent1 = coeffs_power1[0]
        log_A1 = coeffs_power1[1]
        A_power1 = np.exp(log_A1)
        
        print(f"    Z-direction (beyond monopole 1, z > {mono_z1:.0f}, sep > 10):")
        print(f"      Power law fit: E ~ {A_power1:.6f} * r^{power_exponent1:.3f}")
        print(f"      Fitting method: Linear regression on log(E) vs log(r)")
        if abs(power_exponent1 + 1.0) < 0.3:
            print(f"      -> Consistent with 1/r (monopole-like)")
        elif abs(power_exponent1 + 2.0) < 0.3:
            print(f"      -> Consistent with 1/r² (dipole-like)")
        elif abs(power_exponent1 + 3.0) < 0.3:
            print(f"      -> Consistent with 1/r³ (quadrupole-like)")
        
        falloff_type_beyond1 = "1/r" if abs(power_exponent1 + 1.0) < 0.5 else ("1/r²" if abs(power_exponent1 + 2.0) < 0.5 else f"r^{power_exponent1:.2f}")
        
        falloff_data_beyond1 = {
            'type': falloff_type_beyond1,
            'power_A': A_power1,
            'power_exp': power_exponent1,
            'center': mono_z1,
            'fit_region': f'beyond M1 (z > {mono_z1:.0f}), sep > 10'
        }
    else:
        falloff_type_beyond1 = "Unknown"
        print(f"    Beyond monopole 1: Insufficient data for falloff analysis")
    
    # Region 2: Beyond monopole 2 (z < mono_z2)
    beyond_mono2 = z_vals < mono_z2
    z_beyond2 = z_vals[beyond_mono2]
    E_beyond2 = z_energy[beyond_mono2]
    
    r_beyond2 = np.abs(z_beyond2 - mono_z2)
    valid_beyond2 = (r_beyond2 > 10.0) & (r_beyond2 < 50.0)
    
    falloff_data_beyond2 = None
    if np.sum(valid_beyond2) > 5:
        r2 = r_beyond2[valid_beyond2]
        E2 = E_beyond2[valid_beyond2]
        
        log_r2 = np.log(r2)
        log_E2 = np.log(np.maximum(E2, 1e-10))
        
        coeffs_power2 = np.polyfit(log_r2, log_E2, 1)
        power_exponent2 = coeffs_power2[0]
        log_A2 = coeffs_power2[1]
        A_power2 = np.exp(log_A2)
        
        print(f"    Z-direction (beyond monopole 2, z < {mono_z2:.0f}, sep > 10):")
        print(f"      Power law fit: E ~ {A_power2:.6f} * r^{power_exponent2:.3f}")
        print(f"      Fitting method: Linear regression on log(E) vs log(r)")
        if abs(power_exponent2 + 1.0) < 0.3:
            print(f"      -> Consistent with 1/r (monopole-like)")
        elif abs(power_exponent2 + 2.0) < 0.3:
            print(f"      -> Consistent with 1/r² (dipole-like)")
        elif abs(power_exponent2 + 3.0) < 0.3:
            print(f"      -> Consistent with 1/r³ (quadrupole-like)")
        
        falloff_type_beyond2 = "1/r" if abs(power_exponent2 + 1.0) < 0.5 else ("1/r²" if abs(power_exponent2 + 2.0) < 0.5 else f"r^{power_exponent2:.2f}")
        
        falloff_data_beyond2 = {
            'type': falloff_type_beyond2,
            'power_A': A_power2,
            'power_exp': power_exponent2,
            'center': mono_z2,
            'fit_region': f'beyond M2 (z < {mono_z2:.0f}), sep > 10'
        }
    else:
        falloff_type_beyond2 = "Unknown"
        print(f"    Beyond monopole 2: Insufficient data for falloff analysis")
    
    print("\n  Key insight:")
    print("    - If exponent ≈ -2: Dipole field dominates (monopole-antimonopole pair)")
    print("    - If exponent ≈ -1: Individual monopole field dominates")
    print("    - Log-log fitting extracts power law: linear slope in log-log space = power exponent")
    
    return falloff_type_x, falloff_type_beyond1, falloff_type_beyond2, falloff_data_x, falloff_data_beyond1, falloff_data_beyond2

def plot_density_comparison(energy_file1, energy_file2):
    """Create comparison plots of 1D line profiles for two different gamma configurations"""
    
    print(f"\nComparing configurations:")
    print(f"  Config 1: {energy_file1.name}")
    print(f"  Config 2: {energy_file2.name}")
    
    # Extract parameters from both filenames
    params1 = extract_params_from_filename(energy_file1)
    params2 = extract_params_from_filename(energy_file2)
    
    if not params1 or not params2:
        print(f"  Error: Could not extract parameters from filenames")
        return
    
    gamma1_1, gamma2_1 = params1['gamma1'], params1['gamma2']
    gamma1_2, gamma2_2 = params2['gamma1'], params2['gamma2']
    separation = params1['separation']  # Assume same separation
    seed1, seed2 = params1['seed'], params2['seed']
    
    # Read energy density data for both
    data1 = read_energy_density(energy_file1)
    data2 = read_energy_density(energy_file2)
    
    if data1 is None or data2 is None or len(data1) == 0 or len(data2) == 0:
        print(f"  Error: Could not load energy density data")
        return
    
    # Load simulation parameters from first directory
    params = load_simulation_parameters(energy_file1.parent)
    nx_val, ny_val, nz_val = params['nx'], params['ny'], params['nz']
    dx_val, dy_val, dz_val = params['dx'], params['dy'], params['dz']
    
    # Get monopole positions (same for both configurations)
    def get_positions(separation, nx_val, ny_val, nz_val, dx_val, dy_val, dz_val):
        center_x = 0.5 * (nx_val - 1) * dx_val
        center_y = 0.5 * (ny_val - 1) * dy_val
        center_z = 0.5 * (nz_val - 1) * dz_val
        
        z1 = center_z + separation * nz_val * dz_val
        z2 = center_z - separation * nz_val * dz_val
        
        return (center_x, center_y, z1), (center_x, center_y, z2)
    
    monopole1_pos, monopole2_pos = get_positions(separation, nx_val, ny_val, nz_val, dx_val, dy_val, dz_val)
    mono_x, mono_y, mono_z1 = monopole1_pos
    _, _, mono_z2 = monopole2_pos
    
    print(f"  Monopole 1 position: ({mono_x:.2f}, {mono_y:.2f}, {mono_z1:.2f})")
    print(f"  Monopole 2 position: ({mono_x:.2f}, {mono_y:.2f}, {mono_z2:.2f})")
    
    # Extract coordinates for slicing
    x_coords = np.unique(data1['x'].values)
    z_coords = np.unique(data1['z'].values)
    
    x_idx = np.argmin(np.abs(x_coords - mono_x))
    z1_idx = np.argmin(np.abs(z_coords - mono_z1))
    
    x_slice = x_coords[x_idx]
    z_slice = z_coords[z1_idx]
    
    print(f"  Using x={x_slice:.2f}, z={z_slice:.2f} for line plots")
    
    # Extract 1D slices for both configurations
    x_line_data1 = data1[np.abs(data1['z'] - z_slice) < 1e-6].sort_values('x')
    z_line_data1 = data1[np.abs(data1['x'] - x_slice) < 1e-6].sort_values('z')
    
    x_line_data2 = data2[np.abs(data2['z'] - z_slice) < 1e-6].sort_values('x')
    z_line_data2 = data2[np.abs(data2['x'] - x_slice) < 1e-6].sort_values('z')
    
    # Analyze falloff for both configurations
    print("\nConfiguration 1 analysis:")
    falloff_x1, falloff_beyond1_1, falloff_beyond2_1, falloff_data_x1, falloff_data_beyond1_1, falloff_data_beyond2_1 = analyze_falloff_behavior(
        x_line_data1, z_line_data1, mono_x, mono_z1, mono_z2)
    
    print("\nConfiguration 2 analysis:")
    falloff_x2, falloff_beyond1_2, falloff_beyond2_2, falloff_data_x2, falloff_data_beyond1_2, falloff_data_beyond2_2 = analyze_falloff_behavior(
        x_line_data2, z_line_data2, mono_x, mono_z1, mono_z2)
    
    # Create figure with 2 subplots (no heatmap)
    fig = plt.figure(figsize=(14, 6))
    
    # 1D slice along x - comparing both configurations
    ax1 = plt.subplot(121)
    
    # Plot data for both configurations
    x_vals1 = x_line_data1['x'].values
    x_energy1 = x_line_data1['energy_density'].values
    ax1.plot(x_vals1, x_energy1, 'b-', linewidth=2, label=f'γ₁={gamma1_1}π, γ₂={gamma2_1}π')
    
    x_vals2 = x_line_data2['x'].values
    x_energy2 = x_line_data2['energy_density'].values
    ax1.plot(x_vals2, x_energy2, 'r-', linewidth=2, label=f'γ₁={gamma1_2}π, γ₂={gamma2_2}π')
    
    # Plot fitted curves if available
    if falloff_data_x1 is not None:
        x_fit_range = (np.abs(x_vals1 - mono_x) > 10.0) & (np.abs(x_vals1 - mono_x) < 50.0)
        x_fit = x_vals1[x_fit_range]
        r_fit = np.abs(x_fit - falloff_data_x1['center'])
        E_power_fit = falloff_data_x1['power_A'] * (r_fit ** falloff_data_x1['power_exp'])
        ax1.plot(x_fit, E_power_fit, 'b--', linewidth=1.5, alpha=0.5,
                label=f'Fit 1: r^{falloff_data_x1["power_exp"]:.2f}')
    
    if falloff_data_x2 is not None:
        x_fit_range = (np.abs(x_vals2 - mono_x) > 10.0) & (np.abs(x_vals2 - mono_x) < 50.0)
        x_fit = x_vals2[x_fit_range]
        r_fit = np.abs(x_fit - falloff_data_x2['center'])
        E_power_fit = falloff_data_x2['power_A'] * (r_fit ** falloff_data_x2['power_exp'])
        ax1.plot(x_fit, E_power_fit, 'r--', linewidth=1.5, alpha=0.5,
                label=f'Fit 2: r^{falloff_data_x2["power_exp"]:.2f}')
    
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('Energy Density', fontsize=14)
    ax1.set_title(f'Energy Density, X Slice', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.tick_params(labelsize=12)
    
    # 1D slice along z - comparing both configurations
    ax2 = plt.subplot(122)
    
    # Plot data for both configurations
    z_vals1 = z_line_data1['z'].values
    z_energy1 = z_line_data1['energy_density'].values
    ax2.plot(z_vals1, z_energy1, 'b-', linewidth=2, label=f'γ₁={gamma1_1}π, γ₂={gamma2_1}π')
    
    z_vals2 = z_line_data2['z'].values
    z_energy2 = z_line_data2['energy_density'].values
    ax2.plot(z_vals2, z_energy2, 'r-', linewidth=2, label=f'γ₁={gamma1_2}π, γ₂={gamma2_2}π')
    
    # Mark monopole positions
    ax2.axvline(mono_z1, color='gray', linestyle=':', alpha=0.3)
    ax2.axvline(mono_z2, color='gray', linestyle=':', alpha=0.3)
    
    # Plot fitted curves for config 1
    if falloff_data_beyond1_1 is not None:
        z_fit_range = (z_vals1 > mono_z1) & (np.abs(z_vals1 - mono_z1) > 10.0) & (np.abs(z_vals1 - mono_z1) < 50.0)
        z_fit = z_vals1[z_fit_range]
        r_fit = np.abs(z_fit - falloff_data_beyond1_1['center'])
        E_power_fit = falloff_data_beyond1_1['power_A'] * (r_fit ** falloff_data_beyond1_1['power_exp'])
        ax2.plot(z_fit, E_power_fit, 'b--', linewidth=1.5, alpha=0.5, 
                label=f'Fit 1 (M1): r^{falloff_data_beyond1_1["power_exp"]:.2f}')
    
    if falloff_data_beyond2_1 is not None:
        z_fit_range = (z_vals1 < mono_z2) & (np.abs(z_vals1 - mono_z2) > 10.0) & (np.abs(z_vals1 - mono_z2) < 50.0)
        z_fit = z_vals1[z_fit_range]
        r_fit = np.abs(z_fit - falloff_data_beyond2_1['center'])
        E_power_fit = falloff_data_beyond2_1['power_A'] * (r_fit ** falloff_data_beyond2_1['power_exp'])
        ax2.plot(z_fit, E_power_fit, 'b-.', linewidth=1.5, alpha=0.5,
                label=f'Fit 1 (M2): r^{falloff_data_beyond2_1["power_exp"]:.2f}')
    
    # Plot fitted curves for config 2
    if falloff_data_beyond1_2 is not None:
        z_fit_range = (z_vals2 > mono_z1) & (np.abs(z_vals2 - mono_z1) > 10.0) & (np.abs(z_vals2 - mono_z1) < 50.0)
        z_fit = z_vals2[z_fit_range]
        r_fit = np.abs(z_fit - falloff_data_beyond1_2['center'])
        E_power_fit = falloff_data_beyond1_2['power_A'] * (r_fit ** falloff_data_beyond1_2['power_exp'])
        ax2.plot(z_fit, E_power_fit, 'r--', linewidth=1.5, alpha=0.5, 
                label=f'Fit 2 (M1): r^{falloff_data_beyond1_2["power_exp"]:.2f}')
    
    if falloff_data_beyond2_2 is not None:
        z_fit_range = (z_vals2 < mono_z2) & (np.abs(z_vals2 - mono_z2) > 10.0) & (np.abs(z_vals2 - mono_z2) < 50.0)
        z_fit = z_vals2[z_fit_range]
        r_fit = np.abs(z_fit - falloff_data_beyond2_2['center'])
        E_power_fit = falloff_data_beyond2_2['power_A'] * (r_fit ** falloff_data_beyond2_2['power_exp'])
        ax2.plot(z_fit, E_power_fit, 'r-.', linewidth=1.5, alpha=0.5,
                label=f'Fit 2 (M2): r^{falloff_data_beyond2_2["power_exp"]:.2f}')
    
    ax2.set_xlabel('z', fontsize=14)
    ax2.set_ylabel('Energy Density', fontsize=14)
    ax2.set_title(f'Energy Density, Z Slice', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    # Save comparison plot
    outTag = f"comparison_gamma1={gamma1_1}pi-vs-{gamma1_2}pi_sep={separation}_seed={seed1}-{seed2}"
    output_file = OUTPUT_DIR / f'energy_density_comparison_{outTag}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file.name}")
    plt.close()

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("ENERGY DENSITY COMPARISON ANALYSIS")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"Configuration 1 directory: {DATA_DIR_1}")
    print(f"Configuration 2 directory: {DATA_DIR_2}")
    
    if not DATA_DIR_1.exists():
        print(f"ERROR: Data directory {DATA_DIR_1} does not exist!")
        exit()
    
    if not DATA_DIR_2.exists():
        print(f"ERROR: Data directory {DATA_DIR_2} does not exist!")
        exit()
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Find energy density files from both directories
    energy_files1 = find_energy_density_files(DATA_DIR_1)
    energy_files2 = find_energy_density_files(DATA_DIR_2)
    
    if not energy_files1 or not energy_files2:
        print("ERROR: No energy density files found in one or both directories!")
        exit()
    
    print(f"Found {len(energy_files1)} file(s) in config 1")
    print(f"Found {len(energy_files2)} file(s) in config 2")
    
    # Match files with same separation and seed (if possible)
    # For simplicity, compare first file from each directory
    # You can extend this to match by separation/seed if needed
    
    for i, (file1, file2) in enumerate(zip(energy_files1, energy_files2)):
        print(f"\n[{i+1}] Creating comparison plot...")
        plot_density_comparison(file1, file2)
    
    print(f"\nAll comparison plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
