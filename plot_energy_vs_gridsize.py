import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.optimize import curve_fit


output_path = "/share/centaurus_nas/jmg_temp/energy_vs_gridsize_1/"


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

def fitting_function(N, a, b):
    """
    Fitting function: E(N) = a - b/N^2
    For a -1/r^2 potential, energy converges quadratically with grid size
    a: asymptotic (infinite resolution) energy
    b: convergence coefficient (positive for decreasing energy with increasing N)
    """
    return a - b / (N ** 2)

def fitting_function_power(N, a, b, c):
    """
    General power law fitting: E(N) = a - b/N^c
    c is fit from the data to determine actual convergence rate
    """
    return a - b / (N ** c)

def log_space_fitting(N, E):
    """
    Fit in log space to find power law: log(E_max - E) ~ -c*log(N) + log(b)
    Returns the power c and coefficient b
    """
    # Subtract from maximum to get convergence error
    E_max = E.max()
    delta_E = E_max - E
    
    # Only use positive values for log fit
    valid = delta_E > 0
    if valid.sum() < 2:
        return None, None
    
    log_N = np.log(N[valid])
    log_delta_E = np.log(delta_E[valid])
    
    # Linear fit in log space: log(delta_E) = -c*log(N) + log(b)
    coeffs = np.polyfit(log_N, log_delta_E, 1)
    c = -coeffs[0]  # Negative because we want E = a - b/N^c form
    b = np.exp(coeffs[1])
    
    return c, b

def plot_energy_vs_gridsize(data_file_path, output_dir=None):
    """Plot energy vs grid size with separate lines for different dx values"""
    
    # Read the CSV data
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_file_path}")
        return
    
    # Extract seed from filename for parameter file lookup
    filename = os.path.basename(data_file_path)
    seed = None
    if 'seed=' in filename:
        seed_part = filename.split('seed=')[1].split('.')[0]
        try:
            seed = int(seed_part)
        except ValueError:
            seed = None
    
    # Try to read parameters file
    param_file_path = data_file_path.replace('.csv', '.txt').replace('energy_vs_gridsize_dx_study', 'simulation_parameters')
    params = read_parameters_file(param_file_path)
    
    # Extract parameters with defaults
    physical_separation = float(params.get('physical_separation', 64))
    gamma_mult_1 = float(params.get('gamma_mult_1', 0.0))
    gamma_mult_2 = float(params.get('gamma_mult_2', 0.0))
    
    # Convert gamma multipliers to actual gamma values
    gamma_1 = gamma_mult_1 * np.pi
    gamma_2 = gamma_mult_2 * np.pi
    
    # Get unique dx values and sort them
    dx_values = sorted(df['dx'].unique())
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for different dx values
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Storage for fit parameters
    fit_params_dict = {}
    fit_params_power_dict = {}
    
    # Plot each dx value as a separate line
    for i, dx in enumerate(dx_values):
        dx_data = df[df['dx'] == dx].sort_values('grid_size')
        
        plt.plot(dx_data['grid_size'], dx_data['total_energy'], 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)],
                markersize=8, linewidth=2, 
                label=f'dx = {dx}')
        
        # Method 1: Log-space fitting to find power law exponent
        N_array = dx_data['grid_size'].values
        E_array = dx_data['total_energy'].values
        c_log, b_log = log_space_fitting(N_array, E_array)
        
        # Method 2: Fit with free power exponent
        try:
            # Initial guess
            energy_diff = dx_data['total_energy'].max() - dx_data['total_energy'].min()
            
            if c_log is not None:
                # Use log-space result as initial guess
                p0 = [dx_data['total_energy'].max(), b_log, c_log]
            else:
                # Default guess
                p0 = [dx_data['total_energy'].max(), energy_diff * dx_data['grid_size'].min()**2, 2.0]
            
            # Fit with bounds
            lower_bounds = [dx_data['total_energy'].min(), 0, 0.5]
            upper_bounds = [dx_data['total_energy'].max() * 1.5, np.inf, 4.0]
            
            popt, pcov = curve_fit(fitting_function_power, dx_data['grid_size'], 
                                   dx_data['total_energy'], p0=p0, 
                                   bounds=(lower_bounds, upper_bounds),
                                   maxfev=10000)
            
            fit_params_power_dict[dx] = popt
            
            # Generate smooth curve for plotting
            N_smooth = np.linspace(dx_data['grid_size'].min(), 
                                   dx_data['grid_size'].max(), 200)
            E_fit = fitting_function_power(N_smooth, *popt)
            
            # Plot fitted curve
            plt.plot(N_smooth, E_fit, 
                    color=colors[i % len(colors)], 
                    linestyle='--', linewidth=2, alpha=0.5,
                    label=f'dx = {dx} (fit: c={popt[2]:.2f})')
            
        except Exception as e:
            print(f"Warning: Could not fit data for dx = {dx}: {e}")
    
    # Formatting
    plt.xlabel('Grid Size (N)', fontsize=14)  
    plt.ylabel('Total Energy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    
    # Create simplified title with function form
    title_lines = [
        f'Energy vs Grid Size Study',
        f'Physical Separation = {physical_separation}, γ₁ = {gamma_1:.3f}, γ₂ = {gamma_2:.3f}',
        f'Fit Function: E(N) = a - b/N^c'
    ]
    
    plt.title('\n'.join(title_lines), fontsize=12, pad=20)
    
    # Adjust layout to prevent title cutoff
    plt.tight_layout()
    
    # Save the plot
    if output_dir is None:
        output_dir = os.path.dirname(data_file_path)  
    
    output_filename = filename.replace('.csv', '_plot.png')
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Fitting Function: E(N) = a - b/N^c")
    print("(c is determined from log-log fit)")
    print("-" * 50)
    for dx in dx_values:
        dx_data = df[df['dx'] == dx]
        print(f"dx = {dx}:")
        print(f"  Grid sizes: {sorted(dx_data['grid_size'].unique())}")
        print(f"  Energy range: {dx_data['total_energy'].min():.6e} - {dx_data['total_energy'].max():.6e}")
        print(f"  Energy ratio (max/min): {dx_data['total_energy'].max()/dx_data['total_energy'].min():.2f}")
        
        # Print fit parameters if available
        if dx in fit_params_power_dict:
            a, b, c = fit_params_power_dict[dx]
            print(f"  Fit parameters: a = {a:.6e}, b = {b:.6e}, c = {c:.3f}")
            print(f"  Asymptotic energy (N→∞): {a:.6e}")
            print(f"  Convergence rate: ~1/N^{c:.2f}")
        print()

def main():
    """Main function to process all CSV files in the output directory"""
    
    # Default paths
    
    local_path = "./output/"  # Alternative local path
    
    # Check which path exists
    if os.path.exists(output_path):
        search_path = output_path
    elif os.path.exists(local_path):
        search_path = local_path
    else:
        print("No output directory found. Please specify the correct path.")
        return
    
    # Find all CSV files matching the pattern
    csv_pattern = os.path.join(search_path, "energy_vs_gridsize_dx_study_seed=*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        print("Available files in directory:")
        for file in os.listdir(search_path):
            if file.endswith('.csv'):
                print(f"  {file}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  {os.path.basename(file)}")
    
    # Process each file
    for csv_file in csv_files:
        print(f"\nProcessing: {os.path.basename(csv_file)}")
        try:
            plot_energy_vs_gridsize(csv_file, search_path)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    # You can also call this directly with a specific file:
    # plot_energy_vs_gridsize("/path/to/your/energy_vs_gridsize_dx_study_seed=73.csv")
    main()
