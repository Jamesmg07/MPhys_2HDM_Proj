import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_6/2gamma_loop_large/")
OUTPUT_DIR = Path("/share/centaurus_nas/mkza/Plots/")
dx, dy, dz = 0.7, 0.7, 0.7  # Grid spacings (same for all box sizes)
seed = 73  # From C++ code

# Box sizes to analyze
BOX_SIZES = [128, 256, 512, 1024]

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def find_master_energy_file(box_size):
    """Find the master energy file for a specific box size"""
    pattern = f"master_energy_gamma1_gamma2_sep_nx={box_size}_*.csv"
    files = list(DATA_DIR.glob(pattern))
    
    if files:
        print(f"  Found file for box size {box_size}³: {files[0].name}")
        return files[0]
    else:
        print(f"  WARNING: No file found for box size {box_size}³")
        return None

def load_master_energy_data(filepath):
    """Load master energy data with gamma1, gamma2, separation, and energy"""
    try:
        data = pd.read_csv(filepath)
        print(f"    Loaded {len(data)} data points")
        return data
    except Exception as e:
        print(f"    Error loading data from {filepath}: {e}")
        return None

def extract_parameters_from_filename(filename):
    """Extract simulation parameters from master file name"""
    nx_match = re.search(r'nx=(\d+)', filename)
    grid_size = int(nx_match.group(1)) if nx_match else None
    
    seed_match = re.search(r'seed=(\d+)', filename)
    seed_val = int(seed_match.group(1)) if seed_match else seed
    
    return grid_size, seed_val

def plot_multibox_energy_comparison(data_dict):
    """
    Plot energy vs separation for all box sizes on the same axes.
    One plot per unique (gamma1, gamma2) combination.
    Returns dictionary of maximum separations and all plotted data for each box size.
    """
    
    # Get all unique gamma combinations from all datasets
    all_gamma_pairs = set()
    for box_size, data in data_dict.items():
        if data is not None:
            pairs = set(zip(data['gamma_mult_1'], data['gamma_mult_2']))
            all_gamma_pairs.update(pairs)
    
    all_gamma_pairs = sorted(list(all_gamma_pairs))
    print(f"\n  Found {len(all_gamma_pairs)} unique (γ₁, γ₂) combinations")
    
    # Store maximum separations and all plotted data for each box size and gamma combination
    max_separations = {}
    all_plotted_data = {}
    
    # Define colors and markers for different box sizes
    box_colors = {128: 'blue', 256: 'green', 512: 'orange', 1024: 'red'}
    box_markers = {128: 'o', 256: 's', 512: '^', 1024: 'D'}
    
    # Create plots for each gamma combination
    for gamma1, gamma2 in all_gamma_pairs:
        print(f"\n  Processing γ₁={gamma1}π, γ₂={gamma2}π")
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Track if we have any valid data for this gamma combination
        has_data = False
        
        for box_size in BOX_SIZES:
            if box_size not in data_dict or data_dict[box_size] is None:
                print(f"    No data for box size {box_size}³")
                continue
            
            data = data_dict[box_size]
            
            # Filter data for this gamma combination
            mask = (data['gamma_mult_1'] == gamma1) & (data['gamma_mult_2'] == gamma2)
            group_data = data[mask]
            
            if len(group_data) == 0:
                print(f"    No data for box size {box_size}³")
                continue
            
            has_data = True
            
            # Get separation and energy
            separations = group_data['separation'].values
            energies = group_data['total_energy'].values
            
            # Calculate vacuum energy correction for this box size
            vacuum_energy = (1/8) * ((box_size) * dx)**3
            energies_corrected = energies + vacuum_energy
            
            # Convert to real separation
            real_separations = 2 * separations * dz * box_size
            
            # Sort by separation
            sort_idx = np.argsort(real_separations)
            real_separations = real_separations[sort_idx]
            energies_corrected = energies_corrected[sort_idx]
            
            # Find maximum energy location
            max_idx = np.argmax(energies_corrected)
            max_sep = real_separations[max_idx]
            max_energy = energies_corrected[max_idx]
            
            # Store maximum separation and all plotted data
            key = (gamma1, gamma2, box_size)
            max_separations[key] = (max_sep, max_energy)
            all_plotted_data[key] = {
                'separations': real_separations,
                'energies': energies_corrected
            }
            
            print(f"    Box {box_size}³: max at separation={max_sep:.3f} (real units)")
            
            # Plot curve
            ax.plot(real_separations, energies_corrected, 
                   color=box_colors[box_size], marker=box_markers[box_size],
                   linewidth=2, markersize=6, linestyle='-',
                   label=f'{box_size}³ (max at {max_sep:.2f})')
            
            # Mark the maximum with a star
            ax.plot(max_sep, max_energy, '*', 
                   color=box_colors[box_size], markersize=18, 
                   markeredgecolor='black', markeredgewidth=0.8)
        
        if not has_data:
            print(f"    Skipping plot - no data for any box size")
            plt.close()
            continue
        
        # Customize plot
        ax.set_xlabel('Monopole-Antimonopole Separation (real distance)', fontsize=13)
        ax.set_ylabel('Total Energy (Vacuum Corrected)', fontsize=13)
        ax.set_title(f'Energy vs Separation: Multi-Box Comparison\n'
                    f'γ₁={gamma1}π, γ₂={gamma2}π, Seed: {seed}',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Add info text box
        info_text = (f'γ₁ = {gamma1}π\n'
                    f'γ₂ = {gamma2}π\n'
                    f'Seed: {seed}\n'
                    f'dx = dy = dz = {dx}\n'
                    f'Box sizes: 128³, 256³, 512³, 1024³')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save plot
        gamma1_str = str(gamma1).replace('.', 'p')
        gamma2_str = str(gamma2).replace('.', 'p')
        filename = f'multibox_energy_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.png'
        save_path = OUTPUT_DIR / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()
    
    return max_separations, all_plotted_data

def save_maximum_separations(max_separations, all_plotted_data):
    """Save maximum separation values and all plotted data to text files"""
    
    output_file = OUTPUT_DIR / f'maximum_separations_seed{seed}.txt'
    
    with open(output_file, 'w') as f:
        f.write("Maximum Energy Separation Values (Real Units)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Grid spacing: dx = dy = dz = {dx}\n")
        f.write(f"Seed: {seed}\n")
        f.write("=" * 70 + "\n\n")
        
        # Group by gamma values
        gamma_pairs = sorted(set((g1, g2) for g1, g2, _ in max_separations.keys()))
        
        for gamma1, gamma2 in gamma_pairs:
            f.write(f"\nγ₁={gamma1}π, γ₂={gamma2}π:\n")
            f.write("-" * 70 + "\n")
            
            for box_size in BOX_SIZES:
                key = (gamma1, gamma2, box_size)
                if key in max_separations:
                    sep, energy = max_separations[key]
                    f.write(f"  Box size {box_size:4d}³:  max separation = {sep:8.4f},  "
                           f"vacuum-corrected energy = {energy:12.6e}\n")
                else:
                    f.write(f"  Box size {box_size:4d}³:  no data\n")
    
    print(f"\n  Maximum separations saved to: {output_file}")
    
    # Save all plotted data to separate files for each gamma combination
    print("\n  Saving all plotted data...")
    gamma_pairs = sorted(set((g1, g2) for g1, g2, _ in all_plotted_data.keys()))
    
    for gamma1, gamma2 in gamma_pairs:
        gamma1_str = str(gamma1).replace('.', 'p')
        gamma2_str = str(gamma2).replace('.', 'p')
        data_file = OUTPUT_DIR / f'plotted_data_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.txt'
        
        with open(data_file, 'w') as f:
            f.write(f"All Plotted Data: γ₁={gamma1}π, γ₂={gamma2}π\n")
            f.write("=" * 90 + "\n")
            f.write(f"Grid spacing: dx = dy = dz = {dx}\n")
            f.write(f"Seed: {seed}\n")
            f.write("=" * 90 + "\n\n")
            
            for box_size in BOX_SIZES:
                key = (gamma1, gamma2, box_size)
                if key in all_plotted_data:
                    f.write(f"\nBox size {box_size}³:\n")
                    f.write("-" * 90 + "\n")
                    f.write(f"{'Separation (real units)':>25}  {'Vacuum-Corrected Energy':>25}\n")
                    f.write("-" * 90 + "\n")
                    
                    separations = all_plotted_data[key]['separations']
                    energies = all_plotted_data[key]['energies']
                    
                    for sep, energy in zip(separations, energies):
                        f.write(f"{sep:25.6f}  {energy:25.12e}\n")
                    
                    f.write("\n")
        
        print(f"    Saved plotted data to: {data_file.name}")
    
    # Also print summary to console
    print("\n" + "="*90)
    print("SUMMARY: Maximum Energy Separations (Real Units)")
    print("="*90)
    
    for gamma1, gamma2 in gamma_pairs:
        print(f"\nγ₁={gamma1}π, γ₂={gamma2}π:")
        for box_size in BOX_SIZES:
            key = (gamma1, gamma2, box_size)
            if key in max_separations:
                sep, energy = max_separations[key]
                print(f"  {box_size:4d}³: separation = {sep:8.4f}, energy = {energy:12.6e}")
            else:
                print(f"  {box_size:4d}³: no data")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("MULTI-BOX ENERGY COMPARISON ANALYSIS")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 3
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Box sizes to analyze: {BOX_SIZES}")
    print(f"Grid spacing: dx = dy = dz = {dx}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Loading data from all box sizes...")
    
    # Load data from all box sizes
    data_dict = {}
    for box_size in BOX_SIZES:
        print(f"\nLoading data for box size {box_size}³...")
        master_file = find_master_energy_file(box_size)
        if master_file:
            data = load_master_energy_data(master_file)
            data_dict[box_size] = data
        else:
            data_dict[box_size] = None
    
    # Check if we have any valid data
    if not any(data is not None for data in data_dict.values()):
        print("ERROR: No valid data found for any box size!")
        exit()
    
    print_progress(3, total_steps, "Creating multi-box comparison plots...")
    max_separations, all_plotted_data = plot_multibox_energy_comparison(data_dict)
    
    if not max_separations:
        print("ERROR: No maximum separations found!")
        exit()
    
    # Save maximum separations and all plotted data to files
    save_maximum_separations(max_separations, all_plotted_data)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("MULTI-BOX ANALYSIS COMPLETE")
    print("="*60)
