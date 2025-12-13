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

def fitting_function_linear(N, a, b):
    """
    Linear fitting: E(N) = a + b*N
    For cases where energy scales linearly with grid size
    """
    return a + b * N

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
    fit_type_dict = {}  # Track which fit type was used
    
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
        
        # Try both linear and power law fits
        fit_success = False
        best_fit_type = None
        best_r_squared = -np.inf
        
        # Try linear fit first
        try:
            popt_linear, _ = curve_fit(fitting_function_linear, N_array, E_array)
            E_fit_linear = fitting_function_linear(N_array, *popt_linear)
            residuals_linear = E_array - E_fit_linear
            ss_res_linear = np.sum(residuals_linear**2)
            ss_tot = np.sum((E_array - np.mean(E_array))**2)
            r_squared_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
            
            if r_squared_linear > best_r_squared:
                best_r_squared = r_squared_linear
                best_fit_type = 'linear'
                best_popt = popt_linear
                fit_success = True
        except Exception as e:
            print(f"Linear fit failed for dx = {dx}: {e}")
        
        # Try power law fit
        try:
            # Use more robust initial guess and bounds
            E_min, E_max = E_array.min(), E_array.max()
            E_range = E_max - E_min
            
            # Initial guess
            if c_log is not None and c_log > 0:
                p0 = [E_max, b_log, c_log]
            else:
                # Conservative guess: assume quadratic convergence
                p0 = [E_max, E_range * N_array.min()**2, 2.0]
            
            # More permissive bounds to avoid infeasibility
            lower_bounds = [E_min - abs(E_range), 0, 0.1]  # Allow c down to 0.1
            upper_bounds = [E_max + abs(E_range), np.inf, 6.0]  # Allow higher powers
            
            # Check if initial guess is within bounds
            p0 = np.clip(p0, lower_bounds, upper_bounds)
            
            popt_power, _ = curve_fit(fitting_function_power, N_array, E_array, 
                                      p0=p0, bounds=(lower_bounds, upper_bounds),
                                      maxfev=10000)
            
            E_fit_power = fitting_function_power(N_array, *popt_power)
            residuals_power = E_array - E_fit_power
            ss_res_power = np.sum(residuals_power**2)
            r_squared_power = 1 - (ss_res_power / ss_tot) if ss_tot > 0 else 0
            
            if r_squared_power > best_r_squared:
                best_r_squared = r_squared_power
                best_fit_type = 'power'
                best_popt = popt_power
                fit_success = True
                
        except Exception as e:
            print(f"Power law fit failed for dx = {dx}: {e}")
        
        # Plot the best fit
        if fit_success:
            fit_type_dict[dx] = best_fit_type
            N_smooth = np.linspace(N_array.min(), N_array.max(), 200)
            
            if best_fit_type == 'linear':
                E_fit = fitting_function_linear(N_smooth, *best_popt)
                b_str = format_to_3sf(best_popt[1])
                fit_label = f'dx = {dx} (linear: E ∝ {b_str}L)'
                fit_params_dict[dx] = best_popt
            else:  # power law
                E_fit = fitting_function_power(N_smooth, *best_popt)
                c_value = best_popt[2]
                c_str = format_to_3sf(c_value)
                # For power law E = a - b/L^c, this means E converges as 1/L^c
                # So we show E ∝ L^(-c)
                fit_label = f'dx = {dx} (power law: E ∝ L^${{-{c_str}}}$)'
                fit_params_power_dict[dx] = best_popt
            
            plt.plot(N_smooth, E_fit, 
                    color=colors[i % len(colors)], 
                    linestyle='--', linewidth=2, alpha=0.5,
                    label=fit_label)
        else:
            print(f"Warning: Both fits failed for dx = {dx}")
    
    # Formatting
    plt.xlabel('Grid Size (L)', fontsize=16)  
    plt.ylabel('Total Energy', fontsize=16)
    plt.xticks([256, 512, 1024, 2048, 4096], fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Create simplified title with function form
    title_lines = [
        f'Energy vs Grid Size Study',
        f'Constant Physical Separation, γ₁ = γ₂ = $\\pi$ ',
      
    ]
    
    plt.title('\n'.join(title_lines), fontsize=16, pad=20)
    
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
    print(f"Auto-fitted: Linear (E=a+bN) or Power Law (E=a-b/N^c)")
    print("-" * 50)
    for dx in dx_values:
        dx_data = df[df['dx'] == dx]
        print(f"dx = {dx}:")
        print(f"  Grid sizes: {sorted(dx_data['grid_size'].unique())}")
        E_min_str = format_to_3sf(dx_data['total_energy'].min())
        E_max_str = format_to_3sf(dx_data['total_energy'].max())
        print(f"  Energy range: {E_min_str} - {E_max_str}")
        print(f"  Energy ratio (max/min): {dx_data['total_energy'].max()/dx_data['total_energy'].min():.2f}")
        
        # Print fit parameters based on type
        if dx in fit_type_dict:
            if fit_type_dict[dx] == 'linear' and dx in fit_params_dict:
                a, b = fit_params_dict[dx]
                a_str = format_to_3sf(a)
                b_str = format_to_3sf(b)
                print(f"  Fit type: LINEAR")
                print(f"  Fit parameters: E(N) = {a_str} + {b_str}*N")
            elif fit_type_dict[dx] == 'power' and dx in fit_params_power_dict:
                a, b, c = fit_params_power_dict[dx]
                a_str = format_to_3sf(a)
                b_str = format_to_3sf(b)
                print(f"  Fit type: POWER LAW")
                print(f"  Fit parameters: a = {a_str}, b = {b_str}, c = {c:.3f}")
                print(f"  Asymptotic energy (N→∞): {a_str}")
                print(f"  Convergence rate: ~1/N^{c:.2f}")
        print()

def format_to_3sf(value):
    """Format a number to 3 significant figures without scientific notation"""
    if value == 0:
        return "0"
    
    # Determine the order of magnitude
    magnitude = int(np.floor(np.log10(abs(value))))
    
    # Round to 3 significant figures
    rounded = round(value, -magnitude + 2)
    
    # Format without scientific notation
    if magnitude >= 2 or magnitude < -2:
        # For large or small numbers, format with appropriate decimal places
        decimals = max(0, 2 - magnitude)
        return f"{rounded:.{decimals}f}"
    else:
        # For numbers close to 1, use standard formatting
        return f"{rounded:.3g}".replace('e', 'E')  # In case .3g still uses sci notation

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
