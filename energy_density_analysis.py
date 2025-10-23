import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import time

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/")
OUTPUT_DIR = Path("/share/centaurus_nas/mkza/Plots/EnergyDensity/")
nx, ny, nz = 256, 256, 256  # Grid dimensions from C++ code
dx, dy, dz = 0.5, 0.5, 0.5  # Grid spacings
gamma_mult = 0.5  # From C++ code
seed = 73  # From C++ code

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def find_separation_energy_file():
    """Find the separation vs energy file"""
    pattern = f"separation_vs_energy_gamma={gamma_mult}pi_nx={nx}_seed={seed}.csv"
    files = list(DATA_DIR.glob(pattern))
    
    if not files:
        # Try alternative pattern without exact match
        files = list(DATA_DIR.glob("separation_vs_energy_*.csv"))
    
    return files[0] if files else None

def find_energy_density_files():
    """Find all energy density files from different separations"""
    # Pattern: energy_density_gamma=0.5pi_nx=256_sep=0.1_seed=73_monopole.csv
    pattern = f"energy_density_gamma={gamma_mult}pi_nx={nx}_sep=*_seed={seed}_monopole.csv"
    files = list(DATA_DIR.glob(pattern))
    
    # Sort by separation value
    def extract_separation(filename):
        match = re.search(r'sep=([0-9.]+)', filename.name)
        return float(match.group(1)) if match else 0.0
    
    files.sort(key=extract_separation)
    return files

def load_separation_energy_data(filepath):
    """Load separation vs energy data"""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"Error loading separation energy data from {filepath}: {e}")
        return None

def load_energy_density_data(filepath):
    """Load energy density data from CSV file"""
    try:
        data = pd.read_csv(filepath)
        
        # Check if we have the expected columns
        expected_cols = ['x', 'y', 'z', 'energy_density']
        if not all(col in data.columns for col in expected_cols):
            print(f"Warning: Missing expected columns in {filepath}")
            print(f"Available columns: {list(data.columns)}")
            return None
        
        return data
    except Exception as e:
        print(f"Error loading energy density data from {filepath}: {e}")
        return None

def extract_parameters_from_filename(filename):
    """Extract simulation parameters from filename"""
    # Extract gamma
    gamma_match = re.search(r'gamma=([0-9.]+)', filename)
    gamma_val = float(gamma_match.group(1)) if gamma_match else gamma_mult
    
    # Extract grid size
    nx_match = re.search(r'nx=(\d+)', filename)
    grid_size = int(nx_match.group(1)) if nx_match else nx
    
    # Extract separation
    sep_match = re.search(r'sep=([0-9.]+)', filename)
    separation = float(sep_match.group(1)) if sep_match else 0.0
    
    # Extract seed
    seed_match = re.search(r'seed=(\d+)', filename)
    seed_val = int(seed_match.group(1)) if seed_match else seed
    
    return gamma_val, grid_size, separation, seed_val

def plot_energy_vs_separation():
    """Plot total energy vs monopole separation"""
    print("\nAnalyzing energy vs separation...")
    
    sep_energy_file = find_separation_energy_file()
    if not sep_energy_file:
        print("  No separation vs energy file found!")
        return
    
    print(f"  Loading data from: {sep_energy_file.name}")
    
    data = load_separation_energy_data(sep_energy_file)
    if data is None or len(data) == 0:
        print("  Error: Could not load separation energy data")
        return
    
    # Extract parameters from filename or use defaults
    gamma_val, grid_size, _, seed_val = extract_parameters_from_filename(sep_energy_file.name)
    
    plt.figure(figsize=(12, 8))
    
    # Plot energy vs separation
    plt.plot(data['separation'], data['total_energy'], 'bo-', linewidth=2, markersize=8,
             label='Total Energy')
    
    plt.xlabel('Monopole-Antimonopole Separation', fontsize=12)
    plt.ylabel('Total Energy', fontsize=12)
    plt.title(f'Total Energy vs Monopole-Antimonopole Separation\n'
              f'γ = {gamma_val}π, Grid: {grid_size}³, Seed: {seed_val}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add simulation info as text box
    info_text = (f'Grid: {grid_size}³\n'
                f'γ = {gamma_val}π\n'
                f'Seed: {seed_val}\n'
                f'Data points: {len(data)}')
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Save plot
    filename = f'energy_vs_separation_gamma_{gamma_val}pi_nx{grid_size}_seed{seed_val}.png'
    save_path = OUTPUT_DIR / filename
    save_and_close_plot(save_path, f"  Saved: {filename}")
    
    # Print statistics
    print(f"  Energy vs separation statistics:")
    print(f"    Separation range: {data['separation'].min():.3f} to {data['separation'].max():.3f}")
    print(f"    Energy range: {data['total_energy'].min():.6f} to {data['total_energy'].max():.6f}")
    print(f"    Energy change: {data['total_energy'].iloc[-1] - data['total_energy'].iloc[0]:.6f}")

def save_and_close_plot(save_path, message):
    """Helper function to save and close plots"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(message)
    plt.close()

def create_energy_density_slice_plot(data, separation, gamma_val, grid_size, seed_val, slice_type='xz', slice_index=None):
    """Create a 2D slice plot of energy density"""
    
    # Reshape 1D data to 3D grid
    try:
        energy_3d = data['energy_density'].values.reshape(grid_size, grid_size, grid_size)
    except ValueError as e:
        print(f"    Error reshaping data: {e}")
        return None
    
    # Determine slice index if not provided
    if slice_index is None:
        slice_index = grid_size // 2  # Middle slice
    
    # Extract slice based on type
    if slice_type == 'xz':
        # x-z plane at fixed y
        energy_slice = energy_3d[:, slice_index, :]
        xlabel = 'x position'
        ylabel = 'z position'
        extent = [0, (grid_size-1)*dx, 0, (grid_size-1)*dz]
        slice_info = f'y={slice_index*dy:.1f}'
    elif slice_type == 'xy':
        # x-y plane at fixed z
        energy_slice = energy_3d[:, :, slice_index]
        xlabel = 'x position'
        ylabel = 'y position'
        extent = [0, (grid_size-1)*dx, 0, (grid_size-1)*dy]
        slice_info = f'z={slice_index*dz:.1f}'
    elif slice_type == 'yz':
        # y-z plane at fixed x
        energy_slice = energy_3d[slice_index, :, :]
        xlabel = 'y position'
        ylabel = 'z position'
        extent = [0, (grid_size-1)*dy, 0, (grid_size-1)*dz]
        slice_info = f'x={slice_index*dx:.1f}'
    else:
        raise ValueError(f"Unknown slice_type: {slice_type}")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot energy density slice
    im = plt.imshow(energy_slice.T, origin='lower', extent=extent, 
                   aspect='auto', cmap='plasma', interpolation='bilinear')
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Energy Density ({slice_type.upper()} slice, {slice_info})\n'
              f'Separation = {separation}, γ = {gamma_val}π, Grid: {grid_size}³, Seed: {seed_val}', 
              fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Energy Density', fontsize=11)
    
    # Add statistics text box
    stats_text = (f'Min: {np.min(energy_slice):.2e}\n'
                 f'Max: {np.max(energy_slice):.2e}\n'
                 f'Mean: {np.mean(energy_slice):.2e}\n'
                 f'Std: {np.std(energy_slice):.2e}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return energy_slice

def plot_energy_density_snapshots():
    """Create individual plots for each energy density file"""
    print("\nAnalyzing energy density snapshots...")
    
    energy_files = find_energy_density_files()
    if not energy_files:
        print("  No energy density files found!")
        return
    
    print(f"  Found {len(energy_files)} energy density files")
    
    for i, file in enumerate(energy_files):
        print(f"  [{i+1}/{len(energy_files)}] Processing: {file.name}")
        
        # Load data
        data = load_energy_density_data(file)
        if data is None:
            print(f"    Skipping due to load error")
            continue
        
        # Extract parameters
        gamma_val, grid_size, separation, seed_val = extract_parameters_from_filename(file.name)
        
        print(f"    Parameters: γ={gamma_val}π, separation={separation}, grid={grid_size}³")
        
        # Create plots for different slice types
        slice_types = ['xz', 'xy', 'yz']
        slice_indices = [grid_size//2, grid_size//2, grid_size//2]  # Middle slices
        
        for slice_type, slice_idx in zip(slice_types, slice_indices):
            try:
                energy_slice = create_energy_density_slice_plot(
                    data, separation, gamma_val, grid_size, seed_val, 
                    slice_type, slice_idx
                )
                
                if energy_slice is not None:
                    # Save plot with descriptive filename
                    filename = (f'energy_density_{slice_type}_slice_sep{separation}_'
                              f'gamma{gamma_val}pi_nx{grid_size}_seed{seed_val}.png')
                    save_path = OUTPUT_DIR / filename
                    save_and_close_plot(save_path, f"    Saved: {filename}")
                
            except Exception as e:
                print(f"    Error creating {slice_type} plot: {e}")

def create_3d_energy_summary():
    """Create a summary plot showing energy density statistics for all separations"""
    print("\nCreating 3D energy summary...")
    
    energy_files = find_energy_density_files()
    if not energy_files:
        print("  No energy density files found!")
        return
    
    separations = []
    max_energies = []
    mean_energies = []
    total_energies = []
    
    for file in energy_files:
        data = load_energy_density_data(file)
        if data is None:
            continue
        
        gamma_val, grid_size, separation, seed_val = extract_parameters_from_filename(file.name)
        
        energy_values = data['energy_density'].values
        
        separations.append(separation)
        max_energies.append(np.max(energy_values))
        mean_energies.append(np.mean(energy_values))
        
        # Calculate total energy (integrate over volume)
        total_energy = np.sum(energy_values) * dx * dy * dz
        total_energies.append(total_energy)
    
    if not separations:
        print("  No valid data found!")
        return
    
    # Sort by separation
    sort_idx = np.argsort(separations)
    separations = np.array(separations)[sort_idx]
    max_energies = np.array(max_energies)[sort_idx]
    mean_energies = np.array(mean_energies)[sort_idx]
    total_energies = np.array(total_energies)[sort_idx]
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Maximum energy density
    ax1.plot(separations, max_energies, 'ro-', linewidth=2, markersize=6)
    ax1.set_xlabel('Separation')
    ax1.set_ylabel('Maximum Energy Density')
    ax1.set_title('Maximum Energy Density vs Separation')
    ax1.grid(True, alpha=0.3)
    
    # Mean energy density
    ax2.plot(separations, mean_energies, 'go-', linewidth=2, markersize=6)
    ax2.set_xlabel('Separation')
    ax2.set_ylabel('Mean Energy Density')
    ax2.set_title('Mean Energy Density vs Separation')
    ax2.grid(True, alpha=0.3)
    
    # Total energy
    ax3.plot(separations, total_energies, 'bo-', linewidth=2, markersize=6)
    ax3.set_xlabel('Separation')
    ax3.set_ylabel('Total Energy')
    ax3.set_title('Total Energy vs Separation')
    ax3.grid(True, alpha=0.3)
    
    # Energy density ratio (max/mean)
    energy_ratio = max_energies / mean_energies
    ax4.plot(separations, energy_ratio, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Separation')
    ax4.set_ylabel('Max/Mean Energy Ratio')
    ax4.set_title('Energy Density Concentration vs Separation')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Energy Density Analysis Summary\n'
                f'γ = {gamma_val}π, Grid: {grid_size}³, Seed: {seed_val}', 
                fontsize=16)
    
    # Save summary plot
    filename = f'energy_density_summary_gamma{gamma_val}pi_nx{grid_size}_seed{seed_val}.png'
    save_path = OUTPUT_DIR / filename
    save_and_close_plot(save_path, f"  Saved: {filename}")
    
    print(f"  Summary statistics for {len(separations)} separations:")
    print(f"    Separation range: {separations[0]:.3f} to {separations[-1]:.3f}")
    print(f"    Max energy density range: {np.min(max_energies):.2e} to {np.max(max_energies):.2e}")
    print(f"    Total energy range: {np.min(total_energies):.2e} to {np.max(total_energies):.2e}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("ENERGY DENSITY ANALYSIS")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 4
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Simulation parameters: Grid={nx}³, γ={gamma_mult}π, Seed={seed}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Plotting energy vs separation...")
    plot_energy_vs_separation()
    
    print_progress(3, total_steps, "Creating energy density snapshots...")
    plot_energy_density_snapshots()
    
    print_progress(4, total_steps, "Creating energy summary analysis...")
    create_3d_energy_summary()
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ENERGY DENSITY ANALYSIS COMPLETE")
    print("="*60)
