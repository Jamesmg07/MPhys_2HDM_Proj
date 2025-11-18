import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_7/conv_test_VEVcorrected/")
OUTPUT_DIR = Path("/share/centaurus_nas/mkza/Plots/")
dx, dy, dz = 0.7, 0.7, 0.7  # Grid spacings
seed = 73  # From C++ code

# Grid configurations
GRID_CONFIG_1 = (128, 128, 128)  # Standard cubic grid
GRID_CONFIG_2 = (128, 128, 256)  # Extended z-axis

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def find_energy_files(box_size):
    """Find both energy files for a specific box size"""
    pattern_base = f"master_energy_gamma1_gamma2_sep_nx={box_size}_*.csv"
    pattern_extended = f"master_energy_gamma1_gamma2_sep_nx={box_size}_*_1.csv"
    
    # Find base file (cubic grid)
    all_files = list(DATA_DIR.glob(pattern_base))
    extended_files = list(DATA_DIR.glob(pattern_extended))
    
    # Base file is the one without "_1"
    base_files = [f for f in all_files if f not in extended_files]
    
    base_file = base_files[0] if base_files else None
    extended_file = extended_files[0] if extended_files else None
    
    if base_file:
        print(f"  Found base file (128³): {base_file.name}")
    else:
        print(f"  WARNING: No base file found for box size {box_size}³")
    
    if extended_file:
        print(f"  Found extended file (128²×256): {extended_file.name}")
    else:
        print(f"  WARNING: No extended file found for box size {box_size}²×256")
    
    return base_file, extended_file

def load_master_energy_data(filepath):
    """Load master energy data with gamma1, gamma2, separation, and energy"""
    try:
        data = pd.read_csv(filepath)
        print(f"    Loaded {len(data)} data points")
        return data
    except Exception as e:
        print(f"    Error loading data from {filepath}: {e}")
        return None

def plot_comparison(base_data, extended_data, box_size):
    """
    Plot energy vs separation comparing cubic and extended grid configurations.
    One plot per unique (gamma1, gamma2) combination.
    """
    
    # Get all unique gamma combinations from both datasets
    all_gamma_pairs = set()
    if base_data is not None:
        pairs = set(zip(base_data['gamma_mult_1'], base_data['gamma_mult_2']))
        all_gamma_pairs.update(pairs)
    if extended_data is not None:
        pairs = set(zip(extended_data['gamma_mult_1'], extended_data['gamma_mult_2']))
        all_gamma_pairs.update(pairs)
    
    all_gamma_pairs = sorted(list(all_gamma_pairs))
    print(f"\n  Found {len(all_gamma_pairs)} unique (γ₁, γ₂) combinations")
    
    # Store all plotted data
    all_plotted_data = {}
    
    # Create plots for each gamma combination
    for gamma1, gamma2 in all_gamma_pairs:
        print(f"\n  Processing γ₁={gamma1}π, γ₂={gamma2}π")
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        has_data = False
        
        # Plot base configuration (128³)
        if base_data is not None:
            mask = (base_data['gamma_mult_1'] == gamma1) & (base_data['gamma_mult_2'] == gamma2)
            group_data = base_data[mask]
            
            if len(group_data) > 0:
                has_data = True
                
                separations = group_data['separation'].values
                energies = group_data['total_energy'].values
                
                # Convert to real separation (cubic grid)
                real_separations = 2 * separations * dz * box_size
                
                # Sort by separation
                sort_idx = np.argsort(real_separations)
                real_separations = real_separations[sort_idx]
                energies = energies[sort_idx]
                
                # Find maximum
                max_idx = np.argmax(energies)
                max_sep = real_separations[max_idx]
                max_energy = energies[max_idx]
                
                # Store data
                key = (gamma1, gamma2, 'cubic')
                all_plotted_data[key] = {
                    'separations': real_separations,
                    'energies': energies
                }
                
                print(f"    128³: max at separation={max_sep:.3f} (real units)")
                
                # Plot
                ax.plot(real_separations, energies, 
                       color='blue', marker='o',
                       linewidth=2, markersize=6, linestyle='-',
                       label=f'128³ (cubic, max at {max_sep:.2f})')
                
                ax.plot(max_sep, max_energy, '*', 
                       color='blue', markersize=18, 
                       markeredgecolor='black', markeredgewidth=0.8)
        
        # Plot extended configuration (128²×256)
        if extended_data is not None:
            mask = (extended_data['gamma_mult_1'] == gamma1) & (extended_data['gamma_mult_2'] == gamma2)
            group_data = extended_data[mask]
            
            if len(group_data) > 0:
                has_data = True
                
                separations = group_data['separation'].values
                energies = group_data['total_energy'].values
                
                # Convert to real separation (extended z-axis: nz = 256)
                real_separations = 2 * separations * dz * 256
                
                # Sort by separation
                sort_idx = np.argsort(real_separations)
                real_separations = real_separations[sort_idx]
                energies = energies[sort_idx]
                
                # Find maximum
                max_idx = np.argmax(energies)
                max_sep = real_separations[max_idx]
                max_energy = energies[max_idx]
                
                # Store data
                key = (gamma1, gamma2, 'extended')
                all_plotted_data[key] = {
                    'separations': real_separations,
                    'energies': energies
                }
                
                print(f"    128²×256: max at separation={max_sep:.3f} (real units)")
                
                # Plot
                ax.plot(real_separations, energies, 
                       color='red', marker='s',
                       linewidth=2, markersize=6, linestyle='-',
                       label=f'128²×256 (extended z, max at {max_sep:.2f})')
                
                ax.plot(max_sep, max_energy, '*', 
                       color='red', markersize=18, 
                       markeredgecolor='black', markeredgewidth=0.8)
        
        if not has_data:
            print(f"    Skipping plot - no data for any configuration")
            plt.close()
            continue
        
        # Customize plot
        ax.set_xlabel('Monopole-Antimonopole Separation (real distance)', fontsize=13)
        ax.set_ylabel('Total Energy', fontsize=13)
        ax.set_title(f'Energy vs Separation: Grid Configuration Comparison\n'
                    f'γ₁={gamma1}π, γ₂={gamma2}π, Seed: {seed}',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Add info text box
        info_text = (f'γ₁ = {gamma1}π\n'
                    f'γ₂ = {gamma2}π\n'
                    f'Seed: {seed}\n'
                    f'dx = dy = dz = {dx}\n'
                    f'Blue: 128³ (cubic)\n'
                    f'Red: 128²×256 (extended z)')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save plot
        gamma1_str = str(gamma1).replace('.', 'p')
        gamma2_str = str(gamma2).replace('.', 'p')
        filename = f'axes_comparison_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.png'
        save_path = OUTPUT_DIR / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()
    
    return all_plotted_data

def save_comparison_data(all_plotted_data):
    """Save all plotted data to text files"""
    
    print("\n  Saving all plotted data...")
    gamma_pairs = sorted(set((g1, g2) for g1, g2, _ in all_plotted_data.keys()))
    
    for gamma1, gamma2 in gamma_pairs:
        gamma1_str = str(gamma1).replace('.', 'p')
        gamma2_str = str(gamma2).replace('.', 'p')
        data_file = OUTPUT_DIR / f'axes_comparison_data_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.txt'
        
        with open(data_file, 'w') as f:
            f.write(f"Grid Configuration Comparison Data: γ₁={gamma1}π, γ₂={gamma2}π\n")
            f.write("=" * 90 + "\n")
            f.write(f"Grid spacing: dx = dy = dz = {dx}\n")
            f.write(f"Seed: {seed}\n")
            f.write("=" * 90 + "\n\n")
            
            # Write cubic grid data
            key_cubic = (gamma1, gamma2, 'cubic')
            if key_cubic in all_plotted_data:
                f.write(f"\n128³ (Cubic Grid):\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Separation (real units)':>25}  {'Energy':>25}\n")
                f.write("-" * 90 + "\n")
                
                separations = all_plotted_data[key_cubic]['separations']
                energies = all_plotted_data[key_cubic]['energies']
                
                for sep, energy in zip(separations, energies):
                    f.write(f"{sep:25.6f}  {energy:25.12e}\n")
                
                f.write("\n")
            
            # Write extended grid data
            key_extended = (gamma1, gamma2, 'extended')
            if key_extended in all_plotted_data:
                f.write(f"\n128²×256 (Extended z-axis):\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Separation (real units)':>25}  {'Energy':>25}\n")
                f.write("-" * 90 + "\n")
                
                separations = all_plotted_data[key_extended]['separations']
                energies = all_plotted_data[key_extended]['energies']
                
                for sep, energy in zip(separations, energies):
                    f.write(f"{sep:25.6f}  {energy:25.12e}\n")
                
                f.write("\n")
        
        print(f"    Saved comparison data to: {data_file.name}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("GRID CONFIGURATION COMPARISON: 128³ vs 128²×256")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 3
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Grid configurations: 128³ (cubic) and 128²×256 (extended z)")
    print(f"Grid spacing: dx = dy = dz = {dx}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Loading data for both grid configurations...")
    
    base_file, extended_file = find_energy_files(128)
    
    base_data = None
    extended_data = None
    
    if base_file:
        print(f"\nLoading cubic grid data (128³)...")
        base_data = load_master_energy_data(base_file)
    
    if extended_file:
        print(f"\nLoading extended grid data (128²×256)...")
        extended_data = load_master_energy_data(extended_file)
    
    if base_data is None and extended_data is None:
        print("ERROR: No valid data found for either configuration!")
        exit()
    
    print_progress(3, total_steps, "Creating comparison plots...")
    all_plotted_data = plot_comparison(base_data, extended_data, 128)
    
    if not all_plotted_data:
        print("ERROR: No data to plot!")
        exit()
    
    # Save comparison data to files
    save_comparison_data(all_plotted_data)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("GRID CONFIGURATION COMPARISON COMPLETE")
    print("="*60)
