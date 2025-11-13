import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
from scipy.optimize import curve_fit
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

def quadratic_function(x, a, b, c):
    """Quadratic function: E(s) = a*s^2 + b*s + c"""
    return a * x**2 + b * x + c

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

def plot_energy_vs_separation_with_fits(data, grid_size, seed_val):
    """Plot energy vs separation for each gamma combination with quadratic fits"""
    
    # Calculate vacuum energy correction
    vacuum_energy = (1/8) * ((grid_size) * dx)**3
    print(f"\n  Vacuum energy correction: {vacuum_energy:.6f}")
    
    # Group data by gamma combinations
    gamma_groups = data.groupby(['gamma_mult_1', 'gamma_mult_2'])
    
    # Store fit coefficients for later heatmap
    fit_coefficients = {}
    
    # Create plots for each gamma combination
    for (gamma1, gamma2), group_data in gamma_groups:
        print(f"\n  Processing γ₁={gamma1}π, γ₂={gamma2}π")
        
        # Get separation and energy
        separations = group_data['separation'].values
        energies = group_data['total_energy'].values
        
        # Convert to real separation
        real_separations = 2 * separations * dz * nz
        
        # Sort by separation
        sort_idx = np.argsort(real_separations)
        real_separations = real_separations[sort_idx]
        energies = energies[sort_idx]
        
        print(f"    Data points: {len(real_separations)}")
        print(f"    Separation range: {real_separations[0]:.3f} to {real_separations[-1]:.3f}")
        print(f"    Energy range: {energies.min():.6f} to {energies.max():.6f}")
        
        # Fit quadratic function
        try:
            popt, pcov = curve_fit(quadratic_function, real_separations, energies)
            a, b, c = popt
            fit_coefficients[(gamma1, gamma2)] = {'a': a, 'b': b, 'c': c}
            
            # Calculate R-squared
            residuals = energies - quadratic_function(real_separations, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((energies - np.mean(energies))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"    Quadratic fit: E(s) = {a:.6e}·s² + {b:.6e}·s + {c:.6e}")
            print(f"    R² = {r_squared:.6f}")
            
            # Create smooth curve for plotting
            s_smooth = np.linspace(real_separations.min(), real_separations.max(), 200)
            e_smooth = quadratic_function(s_smooth, *popt)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot data points
            plt.plot(real_separations, energies, 'bo', markersize=8, label='Simulation data')
            
            # Plot fit
            plt.plot(s_smooth, e_smooth, 'r-', linewidth=2, label='Quadratic fit')
            
            plt.xlabel('Monopole-Antimonopole Separation', fontsize=12)
            plt.ylabel('Total Energy', fontsize=12)
            plt.title(f'Energy vs Separation with Quadratic Fit\n'
                     f'γ₁={gamma1}π, γ₂={gamma2}π, Grid: {grid_size}³, Seed: {seed_val}',
                     fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Create legend with fit parameters
            legend_text = (f'Quadratic fit:\n'
                          f'a = {a:.4e}\n'
                          f'b = {b:.4e}\n'
                          f'c = {c:.4e}\n'
                          f'R² = {r_squared:.6f}')
            plt.legend([plt.Line2D([0], [0], color='b', marker='o', linestyle=''),
                       plt.Line2D([0], [0], color='r', linestyle='-')],
                      ['Simulation data', legend_text],
                      fontsize=10, loc='best')
            
            # Add info text box
            info_text = (f'Grid: {grid_size}³\n'
                        f'Seed: {seed_val}\n'
                        f'Data points: {len(real_separations)}')
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Save plot
            filename = f'energy_fit_gamma1_{gamma1}pi_gamma2_{gamma2}pi_nx{grid_size}_seed{seed_val}.png'
            save_path = OUTPUT_DIR / filename
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved: {filename}")
            plt.close()
            
        except Exception as e:
            print(f"    Error fitting quadratic: {e}")
            fit_coefficients[(gamma1, gamma2)] = {'a': np.nan, 'b': np.nan, 'c': np.nan}
    
    return fit_coefficients

def create_quadratic_coefficient_heatmap(fit_coefficients, grid_size, seed_val):
    """Create heatmap of quadratic coefficients (a) vs gamma1 and gamma2"""
    
    print(f"\n  Creating quadratic coefficient heatmap...")
    
    # Extract unique gamma values
    gamma1_values = sorted(set(g1 for g1, g2 in fit_coefficients.keys()))
    gamma2_values = sorted(set(g2 for g1, g2 in fit_coefficients.keys()))
    
    print(f"    γ₁ values: {gamma1_values}")
    print(f"    γ₂ values: {gamma2_values}")
    
    # Create 2D array for heatmap
    coeff_array = np.zeros((len(gamma2_values), len(gamma1_values)))
    
    for i, gamma2 in enumerate(gamma2_values):
        for j, gamma1 in enumerate(gamma1_values):
            if (gamma1, gamma2) in fit_coefficients:
                coeff_array[i, j] = fit_coefficients[(gamma1, gamma2)]['a']
            else:
                coeff_array[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    im = ax.imshow(coeff_array, cmap='RdBu_r', aspect='auto',
                   interpolation='nearest', origin='lower')
    
    # Set ticks and labels
    ax.set_xticks(range(len(gamma1_values)))
    ax.set_yticks(range(len(gamma2_values)))
    ax.set_xticklabels([f'{g:.1f}π' for g in gamma1_values])
    ax.set_yticklabels([f'{g:.1f}π' for g in gamma2_values])
    
    ax.set_xlabel('γ₁', fontsize=14)
    ax.set_ylabel('γ₂', fontsize=14)
    ax.set_title(f'Quadratic Coefficient (a) vs γ₁ and γ₂\n'
                f'E(s) = a·s² + b·s + c\n'
                f'Grid: {grid_size}³, Seed: {seed_val}',
                fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Quadratic coefficient (a)', fontsize=12)
    
    # Add text annotations with coefficient values
    for i in range(len(gamma2_values)):
        for j in range(len(gamma1_values)):
            value = coeff_array[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.2e}',
                             ha="center", va="center", color="white" if abs(value) > np.nanmax(np.abs(coeff_array))/2 else "black",
                             fontsize=9)
    
    # Statistics text box
    valid_coeffs = coeff_array[~np.isnan(coeff_array)]
    stats_text = (f'Statistics:\n'
                 f'Min: {np.min(valid_coeffs):.4e}\n'
                 f'Max: {np.max(valid_coeffs):.4e}\n'
                 f'Mean: {np.mean(valid_coeffs):.4e}\n'
                 f'Std: {np.std(valid_coeffs):.4e}')
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save heatmap
    filename = f'quadratic_coefficient_heatmap_nx{grid_size}_seed{seed_val}.png'
    save_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {filename}")
    plt.close()
    
    # Also save coefficient data to CSV
    coeff_df = pd.DataFrame(coeff_array, 
                           index=[f'{g}pi' for g in gamma2_values],
                           columns=[f'{g}pi' for g in gamma1_values])
    csv_filename = f'quadratic_coefficients_nx{grid_size}_seed{seed_val}.csv'
    csv_path = OUTPUT_DIR / csv_filename
    coeff_df.to_csv(csv_path)
    print(f"    Saved coefficient data: {csv_filename}")

# Main analysis code
if __name__ == "__main__":
    print("="*60)
    print("GAMMA PLANE ANALYSIS - QUADRATIC FITS")
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
    
    print_progress(3, total_steps, "Fitting quadratic functions to energy vs separation...")
    fit_coefficients = plot_energy_vs_separation_with_fits(data, grid_size, seed_val)
    
    if not fit_coefficients:
        print("ERROR: No fit coefficients obtained!")
        exit()
    
    print_progress(4, total_steps, "Creating quadratic coefficient heatmap...")
    create_quadratic_coefficient_heatmap(fit_coefficients, grid_size, seed_val)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("GAMMA PLANE ANALYSIS COMPLETE")
    print("="*60)
