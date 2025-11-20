import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob

# Configuration
DATA_DIR = Path("/share/centaurus_nas/jmg_temp/dedr_vs_r/")
OUTPUT_DIR = Path("/share/centaurus_nas/jmg_temp/dedr_vs_r/")
seed = 73

# Box sizes to compare
BOX_SIZES = [256, 512, 1024]

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def find_force_files():
    """Find all binding force CSV files and organize by gamma values"""
    files_by_gamma = {}
    
    for box_size in BOX_SIZES:
        pattern = f"binding_force_gamma1_*_gamma2_*_box{box_size}_seed{seed}.csv"
        files = list(DATA_DIR.glob(pattern))
        
        print(f"Found {len(files)} files for box size {box_size}³")
        
        for file in files:
            # Extract gamma values from filename
            parts = file.stem.split('_')
            gamma1_idx = parts.index('gamma1') + 1
            gamma2_idx = parts.index('gamma2') + 1
            
            gamma1_str = parts[gamma1_idx].replace('pi', '')
            gamma2_str = parts[gamma2_idx].replace('pi', '')
            
            gamma_key = (gamma1_str, gamma2_str)
            
            if gamma_key not in files_by_gamma:
                files_by_gamma[gamma_key] = {}
            
            files_by_gamma[gamma_key][box_size] = file
    
    return files_by_gamma

def plot_force_comparison(gamma_key, file_dict):
    """Plot dE/dR vs R for all box sizes for a given gamma combination"""
    gamma1_str, gamma2_str = gamma_key
    
    # Check if this is a single gamma case (γ₁ = γ₂)
    is_single_gamma = (gamma1_str == gamma2_str)
    
    if is_single_gamma:
        print(f"\nPlotting for γ₁ = γ₂ = {gamma1_str}π")
    else:
        print(f"\nPlotting for γ₁={gamma1_str}π, γ₂={gamma2_str}π")
    
    # Define colors and markers
    box_colors = {128: 'blue', 256: 'green', 512: 'orange', 1024: 'red'}
    box_markers = {128: 'o', 256: 's', 512: '^', 1024: 'D'}
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Track data for deviation analysis
    deviation_data = {}
    
    for box_size in sorted(file_dict.keys()):
        filepath = file_dict[box_size]
        
        try:
            data = pd.read_csv(filepath)
            R = data['R_real'].values
            E = data['E_total'].values
            dE_dR = data['dE_dR'].values
            
            # Plot 1: dE/dR vs R
            ax1.plot(R, dE_dR, color=box_colors[box_size], marker=box_markers[box_size],
                    linewidth=2, markersize=5, label=f'{box_size}³', alpha=0.8)
            
            # Plot 2: Energy vs R
            ax2.plot(R, E, color=box_colors[box_size], marker=box_markers[box_size],
                    linewidth=2, markersize=5, label=f'{box_size}³', alpha=0.8)
            
            deviation_data[box_size] = {'R': R, 'dE_dR': dE_dR, 'E': E}
            
            print(f"  Box {box_size}³: {len(R)} data points")
            
        except Exception as e:
            print(f"  Error loading data for box {box_size}³: {e}")
    
    # Customize plot 1
    ax1.set_xlabel('Separation R (real units)', fontsize=13)
    ax1.set_ylabel('dE/dR (Binding Force)', fontsize=13)
    
    if is_single_gamma:
        title = f'Binding Force vs Separation\nγ₁ = γ₂ = {gamma1_str}π'
    else:
        title = f'Binding Force vs Separation\nγ₁={gamma1_str}π, γ₂={gamma2_str}π'
    
    ax1.set_title(title, fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Customize plot 2
    ax2.set_xlabel('Separation R (real units)', fontsize=13)
    ax2.set_ylabel('Total Energy', fontsize=13)
    ax2.set_title('Total Energy vs Separation', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=11)
    
    # Add info text
    if is_single_gamma:
        info_text = (f'γ₁ = γ₂ = {gamma1_str}π\n'
                    f'Seed: {seed}\n'
                    f'Box sizes: {", ".join([str(b)+"³" for b in sorted(file_dict.keys())])}')
    else:
        info_text = (f'γ₁ = {gamma1_str}π\n'
                    f'γ₂ = {gamma2_str}π\n'
                    f'Seed: {seed}\n'
                    f'Box sizes: {", ".join([str(b)+"³" for b in sorted(file_dict.keys())])}')
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'binding_force_comparison_gamma1_{gamma1_str}pi_gamma2_{gamma2_str}pi_seed{seed}.png'
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    # Analyze deviations (find where smaller grids diverge from largest)
    if 1024 in deviation_data:
        analyze_deviations(gamma_key, deviation_data)

def analyze_deviations(gamma_key, deviation_data):
    """Analyze where smaller grids deviate from the largest grid"""
    gamma1_str, gamma2_str = gamma_key
    
    if 1024 not in deviation_data:
        print("  No 1024³ data for deviation analysis")
        return
    
    reference = deviation_data[1024]
    R_ref = reference['R']
    dE_dR_ref = reference['dE_dR']
    
    print("\n  Deviation analysis (relative to 1024³):")
    print("  " + "="*60)
    
    # Define deviation threshold (e.g., 5% relative deviation)
    threshold = 0.05
    
    for box_size in sorted([b for b in deviation_data.keys() if b != 1024]):
        data = deviation_data[box_size]
        R = data['R']
        dE_dR = data['dE_dR']
        
        # Interpolate reference onto this grid
        dE_dR_ref_interp = np.interp(R, R_ref, dE_dR_ref)
        
        # Calculate relative deviation
        # Avoid division by near-zero values
        mask = np.abs(dE_dR_ref_interp) > 1e-10
        rel_deviation = np.zeros_like(dE_dR)
        rel_deviation[mask] = np.abs((dE_dR[mask] - dE_dR_ref_interp[mask]) / dE_dR_ref_interp[mask])
        
        # Find first point where deviation exceeds threshold
        exceed_idx = np.where(rel_deviation > threshold)[0]
        
        if len(exceed_idx) > 0:
            first_exceed = exceed_idx[0]
            R_diverge = R[first_exceed]
            print(f"  Box {box_size}³: Diverges at R = {R_diverge:.4f} "
                  f"(deviation = {rel_deviation[first_exceed]*100:.2f}%)")
        else:
            print(f"  Box {box_size}³: No significant deviation found")
    
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
    
    # Find all force files
    files_by_gamma = find_force_files()
    
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
