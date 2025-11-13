import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Simulation parameters
DATA_DIR = Path("/share/centaurus_nas/mkza/Week_6/2gamma_loop_large/")
OUTPUT_DIR = Path("/share/centaurus_nas/mkza/Plots/")
nx, ny, nz = 512, 512, 512  # Grid dimensions from C++ code
dx, dy, dz = 0.5, 0.5, 0.5  # Grid spacings
seed = 73  # From C++ code

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def find_master_energy_file():
    """Find the master energy file with gamma1, gamma2, and separation data"""
    files = list(DATA_DIR.glob("master_energy_gamma1_gamma2_sep_*.csv"))
    
    print(f"  Found {len(files)} master energy files")
    if files:
        print(f"    Using: {files[0].name}")
    
    return files[0] if files else None

def load_master_energy_data(filepath):
    """Load master energy data with gamma1, gamma2, separation, and energy"""
    try:
        data = pd.read_csv(filepath)
        print(f"  Loaded {len(data)} data points")
        print(f"  Columns: {list(data.columns)}")
        
        # Check unique gamma combinations
        gamma_pairs = data.groupby(['gamma_mult_1', 'gamma_mult_2']).size()
        print(f"  Found {len(gamma_pairs)} unique gamma combinations:")
        for (g1, g2), count in gamma_pairs.items():
            print(f"    γ₁={g1}π, γ₂={g2}π: {count} separations")
        
        return data
    except Exception as e:
        print(f"Error loading master energy data from {filepath}: {e}")
        return None

def extract_parameters_from_filename(filename):
    """Extract simulation parameters from master file name"""
    nx_match = re.search(r'nx=(\d+)', filename)
    grid_size = int(nx_match.group(1)) if nx_match else nx
    
    seed_match = re.search(r'seed=(\d+)', filename)
    seed_val = int(seed_match.group(1)) if seed_match else seed
    
    return grid_size, seed_val

def determine_kappa(separations, energies):
    """
    Determine kappa parameter based on where energy is maximized:
    kappa = (max_separation - min_separation) / (max_separation - min_separation)
    
    kappa = 0.0: Maximum at smallest separation (left edge)
    kappa = 0.5: Maximum at center
    kappa = 1.0: Maximum at largest separation (right edge)
    """
    max_idx = np.argmax(energies)
    max_separation = separations[max_idx]
    
    # Calculate normalized position
    sep_min = separations[0]
    sep_max = separations[-1]
    sep_range = sep_max - sep_min
    
    if sep_range == 0:
        # All separations are the same, return 0.5
        return 0.5
    
    kappa = (max_separation - sep_min) / sep_range
    
    return kappa

def plot_energy_vs_separation_grouped_by_gamma1(data, grid_size, seed_val):
    """
    Plot energy vs separation curves grouped by gamma_1 values.
    Each plot contains all gamma_2 curves for a fixed gamma_1.
    Returns kappa values for each (gamma1, gamma2) pair.
    """
    
    # Get unique gamma values
    gamma1_values = sorted(data['gamma_mult_1'].unique())
    gamma2_values = sorted(data['gamma_mult_2'].unique())
    
    print(f"  Found {len(gamma1_values)} unique γ₁ values: {gamma1_values}")
    print(f"  Found {len(gamma2_values)} unique γ₂ values: {gamma2_values}")
    
    # Store kappa values for heatmap
    kappa_dict = {}
    
    # Create colormap for gamma2 values
    colors = cm.viridis(np.linspace(0, 1, len(gamma2_values)))
    
    # Loop over each gamma1 value and create a plot
    for gamma1 in gamma1_values:
        print(f"\n  Processing γ₁={gamma1}π")
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Plot all gamma2 curves for this gamma1
        for idx, gamma2 in enumerate(gamma2_values):
            # Filter data for this gamma combination
            mask = (data['gamma_mult_1'] == gamma1) & (data['gamma_mult_2'] == gamma2)
            group_data = data[mask]
            
            if len(group_data) == 0:
                print(f"    No data for γ₂={gamma2}π")
                continue
            
            # Get separation and energy
            separations = group_data['separation'].values
            energies = group_data['total_energy'].values
            
            # Convert to real separation
            real_separations = 2 * separations * dz * nz
            
            # Sort by separation
            sort_idx = np.argsort(real_separations)
            real_separations = real_separations[sort_idx]
            energies = energies[sort_idx]
            
            # Determine kappa for this combination
            kappa = determine_kappa(real_separations, energies)
            kappa_dict[(gamma1, gamma2)] = kappa
            
            # Find maximum energy location
            max_idx = np.argmax(energies)
            max_sep = real_separations[max_idx]
            max_energy = energies[max_idx]
            
            print(f"    γ₂={gamma2}π: max at sep={max_sep:.3f}, κ={kappa:.3f}")
            
            # Plot curve
            ax.plot(real_separations, energies, 'o-', 
                   color=colors[idx], linewidth=2, markersize=6,
                   label=f'γ₂={gamma2}π (κ={kappa:.2f})')
            
            # Mark the maximum with a star
            ax.plot(max_sep, max_energy, '*', 
                   color=colors[idx], markersize=15, 
                   markeredgecolor='black', markeredgewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('Monopole-Antimonopole Separation', fontsize=13)
        ax.set_ylabel('Total Energy', fontsize=13)
        ax.set_title(f'Energy vs Separation for γ₁={gamma1}π\n'
                    f'Grid: {grid_size}³, Seed: {seed_val}',
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create legend with two columns
        ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
        
        # Add info text box
        info_text = (f'Grid: {grid_size}³\n'
                    f'γ₁ = {gamma1}π\n'
                    f'Seed: {seed_val}\n'
                    f'κ = 0.0: max at left\n'
                    f'κ = 0.5: max in middle\n'
                    f'κ = 1.0: max at right')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save plot
        filename = f'energy_curves_gamma1_{gamma1}pi_nx{grid_size}_seed{seed_val}.png'
        save_path = OUTPUT_DIR / filename
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()
    
    return kappa_dict

def create_kappa_heatmap(kappa_dict, grid_size, seed_val):
    """Create heatmap of kappa parameter vs gamma1 and gamma2"""
    
    print(f"\n  Creating κ parameter heatmap...")
    
    # Extract unique gamma values
    gamma1_values = sorted(set(g1 for g1, g2 in kappa_dict.keys()))
    gamma2_values = sorted(set(g2 for g1, g2 in kappa_dict.keys()))
    
    print(f"    γ₁ values: {gamma1_values}")
    print(f"    γ₂ values: {gamma2_values}")
    
    # Create 2D array for heatmap
    kappa_array = np.zeros((len(gamma2_values), len(gamma1_values)))
    
    for i, gamma2 in enumerate(gamma2_values):
        for j, gamma1 in enumerate(gamma1_values):
            if (gamma1, gamma2) in kappa_dict:
                kappa_array[i, j] = kappa_dict[(gamma1, gamma2)]
            else:
                kappa_array[i, j] = np.nan
    
    # Create heatmap with continuous colormap
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Use RdYlBu colormap: blue (0) -> yellow (0.5) -> red (1)
    cmap = plt.cm.RdYlBu_r  # reversed so blue is left, red is right
    
    # Plot heatmap
    im = ax.imshow(kappa_array, cmap=cmap, aspect='auto', 
                   interpolation='nearest', origin='lower',
                   vmin=0.0, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(range(len(gamma1_values)))
    ax.set_yticks(range(len(gamma2_values)))
    ax.set_xticklabels([f'{g:.1f}π' for g in gamma1_values], fontsize=11)
    ax.set_yticklabels([f'{g:.1f}π' for g in gamma2_values], fontsize=11)
    
    ax.set_xlabel('γ₁', fontsize=15, fontweight='bold')
    ax.set_ylabel('γ₂', fontsize=15, fontweight='bold')
    ax.set_title(f'Energy Maximum Location Parameter (κ) vs γ₁ and γ₂\n'
                f'κ = 0.0: max at smallest sep | κ = 0.5: max at center | κ = 1.0: max at largest sep\n'
                f'Grid: {grid_size}³, Seed: {seed_val}',
                fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('κ parameter (normalized position of maximum)', fontsize=13, fontweight='bold')
    
    # Add text annotations with kappa values
    for i in range(len(gamma2_values)):
        for j in range(len(gamma1_values)):
            value = kappa_array[i, j]
            if not np.isnan(value):
                # Determine text color based on kappa value for readability
                text_color = 'black' if 0.3 < value < 0.7 else 'white'
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", 
                             color=text_color, fontsize=11, fontweight='bold')
    
    # Statistics text box
    valid_kappa = kappa_array[~np.isnan(kappa_array)]
    
    # Categorize for statistics
    n_left = np.sum(valid_kappa < 0.25)
    n_center = np.sum((valid_kappa >= 0.25) & (valid_kappa <= 0.75))
    n_right = np.sum(valid_kappa > 0.75)
    
    stats_text = (f'Statistics:\n'
                 f'κ < 0.25 (left):   {n_left} ({100*n_left/len(valid_kappa):.1f}%)\n'
                 f'0.25 ≤ κ ≤ 0.75:   {n_center} ({100*n_center/len(valid_kappa):.1f}%)\n'
                 f'κ > 0.75 (right):  {n_right} ({100*n_right/len(valid_kappa):.1f}%)\n'
                 f'Mean κ: {np.mean(valid_kappa):.3f}\n'
                 f'Std κ:  {np.std(valid_kappa):.3f}\n'
                 f'Total: {len(valid_kappa)} combinations')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Save heatmap
    filename = f'kappa_heatmap_nx{grid_size}_seed{seed_val}.png'
    save_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()
    
    # Also save kappa data to CSV
    kappa_df = pd.DataFrame(kappa_array, 
                           index=[f'{g}pi' for g in gamma2_values],
                           columns=[f'{g}pi' for g in gamma1_values])
    csv_filename = f'kappa_values_nx{grid_size}_seed{seed_val}.csv'
    csv_path = OUTPUT_DIR / csv_filename
    kappa_df.to_csv(csv_path)
    print(f"    Saved kappa data: {csv_filename}")
    
    # Print detailed statistics
    print(f"\n  Kappa distribution:")
    print(f"    κ < 0.25 (left region):    {n_left:2d} cases ({100*n_left/len(valid_kappa):5.1f}%)")
    print(f"    0.25 ≤ κ ≤ 0.75 (center):  {n_center:2d} cases ({100*n_center/len(valid_kappa):5.1f}%)")
    print(f"    κ > 0.75 (right region):   {n_right:2d} cases ({100*n_right/len(valid_kappa):5.1f}%)")
    print(f"    Mean κ: {np.mean(valid_kappa):.3f}")
    print(f"    Std κ:  {np.std(valid_kappa):.3f}")
    print(f"    Min κ:  {np.min(valid_kappa):.3f}")
    print(f"    Max κ:  {np.max(valid_kappa):.3f}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("GAMMA PLANE ANALYSIS 2 - ENERGY PEAK LOCATION")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 4
    
    print_progress(1, total_steps, "Initializing analysis...")
    print(f"Looking for files in: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Simulation parameters: Grid={nx}³, Seed={seed}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Loading master energy file...")
    master_file = find_master_energy_file()
    if not master_file:
        print("ERROR: No master energy file found!")
        exit()
    
    data = load_master_energy_data(master_file)
    if data is None:
        print("ERROR: Could not load master energy data!")
        exit()
    
    grid_size, seed_val = extract_parameters_from_filename(master_file.name)
    
    print_progress(3, total_steps, "Plotting energy curves grouped by γ₁...")
    kappa_dict = plot_energy_vs_separation_grouped_by_gamma1(data, grid_size, seed_val)
    
    if not kappa_dict:
        print("ERROR: No kappa values obtained!")
        exit()
    
    print_progress(4, total_steps, "Creating κ parameter heatmap...")
    create_kappa_heatmap(kappa_dict, grid_size, seed_val)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("GAMMA PLANE ANALYSIS 2 COMPLETE")
    print("="*60)
