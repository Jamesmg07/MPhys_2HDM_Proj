import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
import re

# Configuration
DATA_DIR = Path("/share/centaurus_nas/jmg_temp/dedr_fine/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp/dedr_fine/")
seed = 73

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def parse_parameter_file(param_file):
    """Parse the run parameters file to extract configuration details"""
    config = {}
    
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Extract nx
            if line.startswith('nx ='):
                config['nx'] = int(line.split('=')[1].strip())
            
            # Extract dx
            elif line.startswith('dx ='):
                config['dx'] = float(line.split('=')[1].strip())
            
            # Extract dy
            elif line.startswith('dy ='):
                config['dy'] = float(line.split('=')[1].strip())
            
            # Extract dz
            elif line.startswith('dz ='):
                config['dz'] = float(line.split('=')[1].strip())
            
            # Extract seed
            elif line.startswith('Random seed ='):
                config['seed'] = int(line.split('=')[1].strip())
    
    return config

def find_all_configurations():
    """Find all parameter files and extract unique configurations"""
    # Updated pattern to match: run_parameters_nx=*_dx=*_seed=*.txt
    param_files = list(DATA_DIR.glob(f"run_parameters_nx=*_dx=*_seed={seed}.txt"))
    
    configurations = {}
    
    for param_file in param_files:
        # Parse nx and dx from filename
        filename = param_file.stem  # e.g., "run_parameters_nx=256_dx=0.25_seed=73"
        
        # Extract nx
        nx_match = re.search(r'nx=(\d+)', filename)
        # Extract dx (FIX: use group(1) which captures the decimal value)
        dx_match = re.search(r'dx=([\d.]+)', filename)
        
        if nx_match and dx_match:
            nx = int(nx_match.group(1))
            dx = float(dx_match.group(1))  # FIXED: was dx_match.group(1), now correctly gets dx value
            
            # Parse full config from file contents for validation
            config = parse_parameter_file(param_file)
            
            # Verify parsed values match
            if config.get('nx') != nx or abs(config.get('dx', 0) - dx) > 1e-6:  # Use tolerance for float comparison
                print(f"Warning: Filename mismatch for {param_file.name}")
                print(f"  Filename: nx={nx}, dx={dx}")
                print(f"  File contents: nx={config.get('nx')}, dx={config.get('dx')}")
            
            # Use (nx, dx) as the configuration key
            config_key = (nx, dx)
            configurations[config_key] = config
        else:
            print(f"Warning: Could not parse nx/dx from filename: {param_file.name}")
    
    print(f"Found {len(configurations)} unique configuration(s):")
    for (nx, dx), config in sorted(configurations.items()):
        print(f"  nx={nx}, dx={dx}")
    
    return configurations

def find_force_files(configurations):
    """Find all binding force CSV files and organize by gamma values and configuration"""
    files_by_gamma = {}
    
    for (nx, dx), config in configurations.items():
        # Find all possible matching files using a more flexible pattern
        # This handles variations in how dx might be formatted (e.g., 0.25 vs 0.250000)
        pattern = f"binding_force_gamma1_*_gamma2_*_box{nx}_seed{seed}dx_*.csv"
        files = list(DATA_DIR.glob(pattern))
        
        # Filter files to match this specific dx value
        matched_files = []
        for file in files:
            # Extract dx from the actual filename
            dx_match = re.search(r'dx_([\d.]+)\.csv$', file.name)
            if dx_match:
                file_dx = float(dx_match.group(1))
                # Compare with tolerance for floating point
                if abs(file_dx - dx) < 1e-6:
                    matched_files.append(file)
        
        print(f"Found {len(matched_files)} files for nx={nx}, dx={dx}")
        
        for file in matched_files:
            # Extract gamma values from filename
            parts = file.stem.split('_')
            gamma1_idx = parts.index('gamma1') + 1
            gamma2_idx = parts.index('gamma2') + 1
            
            gamma1_str = parts[gamma1_idx].replace('pi', '')
            gamma2_str = parts[gamma2_idx].replace('pi', '')
            
            gamma_key = (gamma1_str, gamma2_str)
            
            if gamma_key not in files_by_gamma:
                files_by_gamma[gamma_key] = {}
            
            # Store by configuration key
            config_key = (nx, dx)
            files_by_gamma[gamma_key][config_key] = file
    
    return files_by_gamma

def generate_color_palette(n):
    """Generate a diverse color palette for n different configurations"""
    if n <= 10:
        # Use standard tableau colors
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    elif n <= 20:
        # Use tab20
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n]
    else:
        # Use a continuous colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    return colors

def generate_markers(n):
    """Generate n different marker styles"""
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # Repeat if needed
    markers = (marker_styles * ((n // len(marker_styles)) + 1))[:n]
    return markers

def plot_force_comparison(gamma_key, file_dict):
    """Plot dE/dR vs R for all configurations for a given gamma combination"""
    gamma1_str, gamma2_str = gamma_key
    
    # Check if this is a single gamma case (γ₁ = γ₂)
    is_single_gamma = (gamma1_str == gamma2_str)
    
    if is_single_gamma:
        print(f"\nPlotting for γ₁ = γ₂ = {gamma1_str}π")
    else:
        print(f"\nPlotting for γ₁={gamma1_str}π, γ₂={gamma2_str}π")
    
    # Generate colors and markers for all configurations
    n_configs = len(file_dict)
    colors = generate_color_palette(n_configs)
    markers = generate_markers(n_configs)
    
    # Create figure with more space for legend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Track data for deviation analysis
    deviation_data = {}
    
    # Sort configurations by nx (descending) then dx
    sorted_configs = sorted(file_dict.keys(), key=lambda x: (-x[0], x[1]))
    
    for idx, config_key in enumerate(sorted_configs):
        nx, dx = config_key
        filepath = file_dict[config_key]
        
        try:
            data = pd.read_csv(filepath)
            R = data['R_real'].values
            E = data['E_total'].values
            dE_dR = data['dE_dR'].values
            
            label = f'nx={nx}, dx={dx}'
            
            # Plot 1: dE/dR vs R
            ax1.plot(R, dE_dR, color=colors[idx], marker=markers[idx],
                    linewidth=2, markersize=5, label=label, alpha=0.8)
            
            # Plot 2: Energy vs R
            ax2.plot(R, E, color=colors[idx], marker=markers[idx],
                    linewidth=2, markersize=5, label=label, alpha=0.8)
            
            deviation_data[config_key] = {'R': R, 'dE_dR': dE_dR, 'E': E}
            
            print(f"  nx={nx}, dx={dx}: {len(R)} data points")
            
        except Exception as e:
            print(f"  Error loading data for nx={nx}, dx={dx}: {e}")
    
    # Customize plot 1
    ax1.set_xlabel('Separation R (real units)', fontsize=13)
    ax1.set_ylabel('dE/dR (Binding Force)', fontsize=13)
    
    if is_single_gamma:
        title = f'Binding Force vs Separation\nγ₁ = γ₂ = {gamma1_str}π'
    else:
        title = f'Binding Force vs Separation\nγ₁={gamma1_str}π, γ₂={gamma2_str}π'
    
    ax1.set_title(title, fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Place legend outside plot area (to the right)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize plot 2
    ax2.set_xlabel('Separation R (real units)', fontsize=13)
    ax2.set_ylabel('Total Energy', fontsize=13)
    ax2.set_title('Total Energy vs Separation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Place legend outside plot area (to the right)
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    
    # Add info text in upper right corner of plot
    if is_single_gamma:
        info_text = (f'γ₁ = γ₂ = {gamma1_str}π\n'
                    f'Seed: {seed}\n'
                    f'{n_configs} configuration(s)')
    else:
        info_text = (f'γ₁ = {gamma1_str}π\n'
                    f'γ₂ = {gamma2_str}π\n'
                    f'Seed: {seed}\n'
                    f'{n_configs} configuration(s)')
    
    ax1.text(0.98, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save plot
    filename = f'binding_force_comparison_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.png'
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    # Analyze deviations (find where smaller grids diverge from largest)
    largest_nx = max(config_key[0] for config_key in deviation_data.keys())
    largest_configs = [k for k in deviation_data.keys() if k[0] == largest_nx]
    
    if largest_configs:
        # Use the configuration with smallest dx as reference
        reference_config = min(largest_configs, key=lambda x: x[1])
        analyze_deviations(gamma_key, deviation_data, reference_config)

def analyze_deviations(gamma_key, deviation_data, reference_config):
    """Analyze where other configurations deviate from the reference configuration"""
    gamma1_str, gamma2_str = gamma_key
    ref_nx, ref_dx = reference_config
    
    reference = deviation_data[reference_config]
    R_ref = reference['R']
    dE_dR_ref = reference['dE_dR']
    
    print(f"\n  Deviation analysis (relative to nx={ref_nx}, dx={ref_dx}):")
    print("  " + "="*60)
    
    # Define deviation threshold (e.g., 5% relative deviation)
    threshold = 0.05
    
    for config_key in sorted([k for k in deviation_data.keys() if k != reference_config]):
        nx, dx = config_key
        data = deviation_data[config_key]
        R = data['R']
        dE_dR = data['dE_dR']
        
        # Interpolate reference onto this grid
        dE_dR_ref_interp = np.interp(R, R_ref, dE_dR_ref)
        
        # Calculate relative deviation
        mask = np.abs(dE_dR_ref_interp) > 1e-10
        rel_deviation = np.zeros_like(dE_dR)
        rel_deviation[mask] = np.abs((dE_dR[mask] - dE_dR_ref_interp[mask]) / dE_dR_ref_interp[mask])
        
        # Find first point where deviation exceeds threshold
        exceed_idx = np.where(rel_deviation > threshold)[0]
        
        if len(exceed_idx) > 0:
            first_exceed = exceed_idx[0]
            R_diverge = R[first_exceed]
            print(f"  nx={nx}, dx={dx}: Diverges at R = {R_diverge:.4f} "
                  f"(deviation = {rel_deviation[first_exceed]*100:.2f}%)")
        else:
            print(f"  nx={nx}, dx={dx}: No significant deviation found")
    
    print("  " + "="*60)

def main():
    print("="*70)
    print("BINDING FORCE ANALYSIS (dE/dR vs R)")
    print("="*70)
    
    print(f"\nLooking for data in: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        return
    
    # Find all configurations from parameter files
    configurations = find_all_configurations()
    
    if not configurations:
        print("ERROR: No parameter files found!")
        return
    
    # Find all force files organized by gamma and configuration
    files_by_gamma = find_force_files(configurations)
    
    if not files_by_gamma:
        print("ERROR: No binding force files found!")
        return
    
    # Determine if we have single gamma or full grid data
    gamma_keys = list(files_by_gamma.keys())
    single_gamma_cases = sum(1 for g1, g2 in gamma_keys if g1 == g2)
    mixed_gamma_cases = sum(1 for g1, g2 in gamma_keys if g1 != g2)
    
    print(f"\nFound data for {len(files_by_gamma)} gamma combination(s)")
    if single_gamma_cases > 0:
        print(f"  - {single_gamma_cases} single gamma cases (γ₁ = γ₂)")
    if mixed_gamma_cases > 0:
        print(f"  - {mixed_gamma_cases} mixed gamma cases (γ₁ ≠ γ₂)")
    
    # Create plots for each gamma combination
    for gamma_key, file_dict in files_by_gamma.items():
        plot_force_comparison(gamma_key, file_dict)
    
    print("\n" + "="*70)
    print("BINDING FORCE ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    main()
