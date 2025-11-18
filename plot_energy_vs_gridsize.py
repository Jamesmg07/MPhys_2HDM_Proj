import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


output_path = "/share/centaurus_nas/jmg_temp/energy_vs_gridsize/"


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
    
    # Plot each dx value as a separate line
    for i, dx in enumerate(dx_values):
        dx_data = df[df['dx'] == dx].sort_values('grid_size')
        
        plt.plot(dx_data['grid_size'], dx_data['total_energy'], 
                color=colors[i % len(colors)], 
                marker=markers[i % len(markers)],
                markersize=8, linewidth=2, 
                label=f'dx = {dx}')
    
    # Formatting
    plt.xlabel('Grid Size (N)', fontsize=14)  
    plt.ylabel('Total Energy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Create simplified title
    title_lines = [
        f'Energy vs Grid Size Study',
        f'Physical Separation = {physical_separation}, γ₁ = {gamma_1:.3f}, γ₂ = {gamma_2:.3f}'
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
    for dx in dx_values:
        dx_data = df[df['dx'] == dx]
        print(f"dx = {dx}:")
        print(f"  Grid sizes: {sorted(dx_data['grid_size'].unique())}")
        print(f"  Energy range: {dx_data['total_energy'].min():.6e} - {dx_data['total_energy'].max():.6e}")
        print(f"  Energy ratio (max/min): {dx_data['total_energy'].max()/dx_data['total_energy'].min():.2f}")
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
