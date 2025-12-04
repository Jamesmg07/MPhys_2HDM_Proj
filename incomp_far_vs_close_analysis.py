import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path


# Configuration
OUTPUT_DIR = "/share/centaurus_nas/jmg_temp/binding_energy_study/"
LOCAL_OUTPUT_DIR = "./output/binding_energy_study/"

# Reference separation for non-interacting limit
REFERENCE_SEPARATION = 112  # Largest separation used as E_M + E_A reference


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    # Try network path first, fall back to local
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return OUTPUT_DIR
    except:
        os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
        return LOCAL_OUTPUT_DIR


def read_parameters_file(param_file_path):
    """Read simulation parameters from the parameters file"""
    params = {}
    try:
        with open(param_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
        return params
    except FileNotFoundError:
        print(f"Warning: Parameters file not found: {param_file_path}")
        return {}


def read_energy_data(data_file_path):
    """
    Read energy data from CSV file produced by C++ simulation
    Expected columns: energy_type, separation, total_energy
    """
    try:
        df = pd.read_csv(data_file_path)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_file_path}")
        return None


def calculate_binding_energies(df):
    """
    Calculate binding energies from the data
    Uses large separation as reference for E_M + E_A
    """
    # Extract individual energies
    E_monopole = df[df['energy_type'] == 'monopole']['total_energy'].values[0]
    E_antimonopole = df[df['energy_type'] == 'antimonopole']['total_energy'].values[0]
    E_sum = E_monopole + E_antimonopole
    
    print(f"Monopole energy: {E_monopole:.6e}")
    print(f"Antimonopole energy: {E_antimonopole:.6e}")
    print(f"Sum (E_M + E_A): {E_sum:.6e}")
    
    # Extract combined energies at different separations
    combined_df = df[df['energy_type'] == 'combined'].copy()
    
    # Calculate binding energy: ΔE = (E_M + E_A) - E_C
    combined_df['binding_energy'] = E_sum - combined_df['total_energy']
    combined_df['E_monopole'] = E_monopole
    combined_df['E_antimonopole'] = E_antimonopole
    combined_df['E_sum'] = E_sum
    
    return combined_df


def plot_binding_energy(df, output_dir, params):
    """
    Create plots of binding energy vs separation
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Energies vs separation
    ax1.plot(df['separation'], df['E_sum'], 'b--', 
             label='E_monopole + E_antimonopole', linewidth=2)
    ax1.plot(df['separation'], df['total_energy'], 'ro-', 
             label='E_combined', markersize=8, linewidth=2)
    ax1.axhline(y=df['E_sum'].iloc[0], color='b', linestyle=':', 
                alpha=0.5, label='Sum of individual energies')
    ax1.set_xlabel('Separation (lattice units)', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Total Energy vs Separation', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Binding energy vs separation
    ax2.plot(df['separation'], df['binding_energy'], 'go-', 
             markersize=8, linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='No binding')
    ax2.set_xlabel('Separation (lattice units)', fontsize=12)
    ax2.set_ylabel('Binding Energy: (E_M + E_A) - E_C', fontsize=12)
    ax2.set_title('Binding Energy vs Separation\n(Positive = Attraction, Negative = Repulsion)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation text
    mean_binding = df['binding_energy'].mean()
    if mean_binding > 0:
        interpretation = "Overall: ATTRACTIVE (bound state preferred)"
        color = 'green'
    else:
        interpretation = "Overall: REPULSIVE (separated state preferred)"
        color = 'red'
    
    ax2.text(0.05, 0.95, interpretation, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    seed = params.get('seed', 'unknown')
    plot_path = os.path.join(output_dir, f'binding_energy_plot_seed={seed}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()


def save_results(df, output_dir, params):
    """Save results to CSV with summary"""
    if df is None or len(df) == 0:
        return
    
    seed = params.get('seed', 'unknown')
    csv_path = os.path.join(output_dir, f'binding_energy_analysis_seed={seed}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Grid size: {params.get('grid_size', 'unknown')}")
    print(f"dx: {params.get('dx', 'unknown')}")
    print(f"Gamma parameters: γ₁={params.get('gamma_mult_1', 'unknown')}, γ₂={params.get('gamma_mult_2', 'unknown')}")
    print(f"Monopole energy: {df['E_monopole'].iloc[0]:.6e}")
    print(f"Antimonopole energy: {df['E_antimonopole'].iloc[0]:.6e}")
    print(f"Sum (E_M + E_A): {df['E_sum'].iloc[0]:.6e}")
    print("\nBinding energies:")
    for _, row in df.iterrows():
        sign = "ATTRACTIVE" if row['binding_energy'] > 0 else "REPULSIVE"
        print(f"  Separation {row['separation']:3.0f}: ΔE = {row['binding_energy']:+.6e} ({sign})")


def main():
    """Main execution function"""
    print("Binding Energy Analysis")
    print("="*60)
    
    # Determine output directory
    if os.path.exists(OUTPUT_DIR):
        search_path = OUTPUT_DIR
    elif os.path.exists(LOCAL_OUTPUT_DIR):
        search_path = LOCAL_OUTPUT_DIR
    else:
        print("No output directory found.")
        return
    
    # Find data files
    csv_pattern = os.path.join(search_path, "binding_energy_data_seed=*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process:")
    for file in csv_files:
        print(f"  {os.path.basename(file)}")
    
    # Process each file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(csv_file)}")
        print('='*60)
        
        # Read parameter file
        param_file = csv_file.replace('binding_energy_data', 'binding_energy_parameters')
        param_file = param_file.replace('.csv', '.txt')
        params = read_parameters_file(param_file)
        
        # Read energy data
        df_raw = read_energy_data(csv_file)
        if df_raw is None:
            continue
        
        # Calculate binding energies
        df_results = calculate_binding_energies(df_raw)
        
        # Save results
        save_results(df_results, search_path, params)
        
        # Create plots
        plot_binding_energy(df_results, search_path, params)
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("="*60)


if __name__ == "__main__":
    main()
