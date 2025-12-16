import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path

# Data directory
DATA_DIR = Path(r".\Report_Data\half_pi")
OUTPUT_DIR = Path(r".\Report_Plots")

def load_simulation_parameters(param_file):
    """Load simulation parameters from a single parameter file"""
    params = {}
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert to appropriate type
                        if key in ['nx', 'ny', 'nz', 'nt', 'seed', 'sep_SaveFreq', 'R_saveFreq']:
                            params[key] = int(value)
                        elif key in ['dx', 'dy', 'dz', 'dt', 'gamma_mult_1', 'gamma_mult_2', 'gamma_mult',
                                   'offset_from_centre', 'monopole1_vx', 'monopole1_vy', 'monopole1_vz',
                                   'monopole2_vx', 'monopole2_vy', 'monopole2_vz']:
                            params[key] = float(value)
                        else:
                            params[key] = value
        
        # Handle single gamma_mult parameter (use for both gamma1 and gamma2)
        if 'gamma_mult' in params and 'gamma_mult_1' not in params:
            params['gamma_mult_1'] = params['gamma_mult']
            params['gamma_mult_2'] = params['gamma_mult']
        
        return params
        
    except Exception as e:
        print(f"Error reading parameter file {param_file.name}: {e}")
        return None

def match_tracking_to_params(tracking_files, param_files):
    """Match each monopole tracking file to its corresponding parameter file"""
    matched_data = []
    
    for tracking_file in tracking_files:
        # Extract outTag from tracking filename
        # Pattern: monopole_tracking_{outTag}.csv
        match = re.search(r'monopole_tracking_(.+)\.csv', tracking_file.name)
        if not match:
            print(f"Warning: Could not extract outTag from {tracking_file.name}")
            continue
        
        out_tag = match.group(1)
        
        # Find matching parameter file
        param_file = None
        for pf in param_files:
            if out_tag in pf.name:
                param_file = pf
                break
        
        if param_file is None:
            print(f"Warning: No parameter file found for {tracking_file.name}")
            continue
        
        # Load parameters
        params = load_simulation_parameters(param_file)
        if params is None:
            continue
        
        matched_data.append({
            'tracking_file': tracking_file,
            'param_file': param_file,
            'params': params,
            'out_tag': out_tag
        })
    
    return matched_data

def load_monopole_tracking_data(filepath):
    """Load monopole tracking data from file"""
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        print(f"Error loading monopole tracking data from {filepath}: {e}")
        return None

def calculate_separation(tracking_data, params):
    """Calculate separation for each timestep"""
    separations = []
    valid_timesteps = []
    
    for _, row in tracking_data.iterrows():
        # Check if both monopoles were found
        if (row['x1_center'] != -1 and row['y1_center'] != -1 and row['z1_center'] != -1 and
            row['x2_center'] != -1 and row['y2_center'] != -1 and row['z2_center'] != -1):
            
            # Calculate 3D distance
            dx = row['x2_center'] - row['x1_center']
            dy = row['y2_center'] - row['y1_center']
            dz = row['z2_center'] - row['z1_center']
            separation = np.sqrt(dx**2 + dy**2 + dz**2)
            
            separations.append(separation)
            valid_timesteps.append(row['timestep'])
    
    # Convert to numpy arrays
    valid_timesteps = np.array(valid_timesteps)
    separations = np.array(separations)
    
    # Convert timesteps to physical time
    dt = params.get('dt', 0.1)
    time_values = valid_timesteps * dt
    
    return time_values, separations, valid_timesteps

def plot_separation_comparison(matched_data):
    """Plot separation vs time for all simulations on a single plot"""
    fig, ax = plt.subplots(figsize=(10, 6.6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(matched_data)))
    
    # Collect all simulation data with initial separations for sorting
    sim_data_list = []
    
    for idx, sim in enumerate(matched_data):
        tracking_file = sim['tracking_file']
        params = sim['params']
        out_tag = sim['out_tag']
        
        # Extract gamma values for this simulation
        gamma1_sim = params.get('gamma_mult_1', 0.0)
        gamma2_sim = params.get('gamma_mult_2', 1.0)
        
        print(f"\nProcessing simulation {idx+1}/{len(matched_data)}")
        print(f"  File: {tracking_file.name}")
        print(f"  γ₁ = {gamma1_sim}π, γ₂ = {gamma2_sim}π")
        
        # Load tracking data
        tracking_data = load_monopole_tracking_data(tracking_file)
        if tracking_data is None or len(tracking_data) == 0:
            print(f"  Warning: No valid data in {tracking_file.name}")
            continue
        
        # Calculate separation
        time_values, separations, valid_timesteps = calculate_separation(tracking_data, params)
        
        if len(separations) == 0:
            print(f"  Warning: No valid monopole pairs found")
            continue
        
        # Cut off data after t = 50
        mask = time_values <= 50.0
        time_values = time_values[mask]
        separations = separations[mask]
        
        # Get initial separation
        initial_sep = separations[0]
        
        # Print statistics
        print(f"  Valid timesteps: {len(separations)}")
        print(f"  Initial separation: {initial_sep:.4f}")
        print(f"  Final separation: {separations[-1]:.4f}")
        print(f"  Change: {separations[-1] - initial_sep:.4f}")
        if abs(initial_sep) > 1e-10:
            print(f"  Relative change: {(separations[-1] - initial_sep)/initial_sep*100:.2f}%")
        
        # Store data for sorting
        sim_data_list.append({
            'initial_sep': initial_sep,
            'time_values': time_values,
            'separations': separations,
            'color': colors[idx]
        })
    
    # Sort by initial separation
    sim_data_list.sort(key=lambda x: x['initial_sep'])
    
    # Plot in sorted order
    for idx, sim_data in enumerate(sim_data_list):
        # Legend label with initial separation (1 decimal place)
        label = f"$R_i$ = {sim_data['initial_sep']:.1f}"
        
        # For the two smallest initial separations, cut off data after minimum
        if idx < 2:
            # Find minimum separation index
            min_idx = np.argmin(sim_data['separations'])
            # Cut off data after the minimum
            time_to_plot = sim_data['time_values'][:min_idx+1]
            sep_to_plot = sim_data['separations'][:min_idx+1]
        else:
            time_to_plot = sim_data['time_values']
            sep_to_plot = sim_data['separations']
        
        ax.plot(time_to_plot, sep_to_plot, 
               color=sim_data['color'], linewidth=2, marker='o', markersize=3,
               label=label, alpha=0.8)
    
    # Mark minimum points for the two smallest initial separations
    for i in range(min(2, len(sim_data_list))):
        sim_data = sim_data_list[i]
        # Find minimum separation
        min_idx = np.argmin(sim_data['separations'])
        min_time = sim_data['time_values'][min_idx]
        min_sep = sim_data['separations'][min_idx]
        
        ax.plot(min_time, min_sep, 'rx', markersize=12, markeredgewidth=3, 
               label='Collapsed' if i == 0 else '', zorder=10)
    
    ax.set_xlabel('$t$', fontsize=16)
    ax.set_ylabel('$R$', fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=16)
    
    plt.tight_layout()
    
    # Save plot
    save_path = OUTPUT_DIR / f'monopole_separation_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("MONOPOLE SEPARATION COMPARISON ACROSS SIMULATIONS")
    print("="*60)
    
    # Set matplotlib to non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    
    print(f"\nLooking for files in: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    # Find all parameter files and tracking files
    print("\nSearching for parameter files...")
    param_files = list(DATA_DIR.glob("simulation_parameters_*.txt"))
    print(f"Found {len(param_files)} parameter files")
    
    print("\nSearching for monopole tracking files...")
    tracking_files = list(DATA_DIR.glob("monopole_tracking_*.csv"))
    print(f"Found {len(tracking_files)} monopole tracking files")
    
    if len(param_files) == 0 or len(tracking_files) == 0:
        print("ERROR: No parameter or tracking files found!")
        exit()
    
    # Match tracking files to parameter files
    print("\nMatching tracking files to parameter files...")
    matched_data = match_tracking_to_params(tracking_files, param_files)
    print(f"Successfully matched {len(matched_data)} simulations")
    
    if len(matched_data) == 0:
        print("ERROR: Could not match any tracking files to parameter files!")
        exit()
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_separation_comparison(matched_data)
    
    print(f"\nPlot saved to: {OUTPUT_DIR}")
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
