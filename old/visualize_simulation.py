import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import re
from pathlib import Path

# Simulation parameters
DATA_DIR = Path("./Data/")
OUTPUT_DIR = Path("./Plots/")
nx, ny, nz = 256, 256, 1
nPos = nx * ny * nz

# Create output directory for plots
OUTPUT_DIR.mkdir(exist_ok=True)

def print_progress(step, total_steps, message):
    """Print progress with step counter"""
    print(f"[{step}/{total_steps}] {message}")

def find_output_files():
    """Find all output files from the simulation"""
    files = list(DATA_DIR.glob("*_nx*_nt*_seed*_Z2.txt"))
    
    # Categorize files by type
    file_categories = {
        'final': [f for f in files if "finalFields" in f.name],
        'vals': [f for f in files if "valsPerLoop" in f.name],
        'timestep': [f for f in files if "fields_timestep" in f.name]
    }
    
    return file_categories['final'], file_categories['vals'], file_categories['timestep']

def load_field_data(filepath):
    """Load field data and reshape to grid"""
    try:
        data = pd.read_csv(filepath, sep=' ')
        fields = {}
        
        for col in data.columns:
            if col.strip() and len(data[col].values) == nPos:
                field_data = data[col].values
                # Reshape based on dimensions
                shape = (nx, ny, nz) if nz > 1 else (nx, ny)
                fields[col.strip()] = field_data.reshape(shape)
        
        return fields
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def save_and_close_plot(save_path, message):
    """Helper function to save and close plots"""
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(message)
    plt.close()

def plot_energy_evolution(vals_files):
    """Plot energy evolution over time"""
    if not vals_files:
        print("No valsPerLoop files found")
        return
    
    print(f"\nProcessing {len(vals_files)} energy evolution files...")
    
    for i, vals_file in enumerate(vals_files):
        print(f"  Processing file {i+1}/{len(vals_files)}: {vals_file.name}")
        
        try:
            data = pd.read_csv(vals_file, sep=' ')
            timesteps = np.arange(len(data))
            
            # Define plot configurations
            plot_configs = [
                ('Energy', 'Total Energy', 'Energy Evolution', (0, 0)),
                ('NDW', 'Number of Domain Walls', 'Domain Wall Count', (0, 1)),
                ('ADW_Simple', 'Domain Wall Area (Simple)', 'Domain Wall Area Evolution', (1, 0)),
                ('ADW_Full', 'Domain Wall Area (Full)', 'Domain Wall Area Evolution (Full)', (1, 1))
            ]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Simulation Evolution - {vals_file.stem}', fontsize=14)
            
            # Plot each available column
            for col_name, ylabel, title, pos in plot_configs:
                if col_name in data.columns:
                    ax = axes[pos]
                    ax.plot(timesteps, data[col_name])
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel(ylabel)
                    ax.set_title(title)
                    ax.grid(True)
                    print(f"      ✓ {title} created")
            
            save_and_close_plot(OUTPUT_DIR / f'energy_evolution_{vals_file.stem}.png',
                               f"    Saved: energy_evolution_{vals_file.stem}.png")
            
        except Exception as e:
            print(f"    Error loading {vals_file}: {e}")

def plot_field_snapshot(fields, title="Field Configuration", save_name=None):
    """Plot 2D snapshots of various fields"""
    if not fields:
        return
    
    field_names = list(fields.keys())
    n_fields = len(field_names)
    
    # Determine subplot layout
    if n_fields <= 4:
        nrows, ncols = 2, 2
    elif n_fields <= 6:
        nrows, ncols = 2, 3
    elif n_fields <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 4, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(title, fontsize=16)
    
    # Custom colormap
    colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    for i, (field_name, field_data) in enumerate(fields.items()):
        if i >= nrows * ncols:
            break
        
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        
        # Plot field data
        plot_data = field_data if nz == 1 else field_data[:, :, nz//2]
        im = ax.imshow(plot_data, cmap=cmap, origin='lower', interpolation='bilinear')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_title(f'{field_name}')
        ax.set_xlabel('y')
        ax.set_ylabel('x')
    
    # Hide unused subplots
    for i in range(n_fields, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)
    
    if save_name:
        save_and_close_plot(OUTPUT_DIR / f'{save_name}.png', 
                           f"    Saved: {save_name}.png")
    else:
        plt.show()

def create_animation(timestep_files):
    """Create animated gif from timestep files"""
    if not timestep_files:
        print("No timestep files found for animation")
        return
    
    print(f"\nCreating animation from {len(timestep_files)} timestep files...")
    
    # Sort files by timestep
    def extract_timestep(filename):
        match = re.search(r'timestep=(\d+)', filename.name)
        return int(match.group(1)) if match else 0
    
    timestep_files.sort(key=extract_timestep)
    
    # Load all timestep data efficiently
    print("  Loading timestep data...")
    all_data, timesteps = [], []
    
    for i, file in enumerate(timestep_files):
        if i % max(1, len(timestep_files) // 10) == 0:
            print(f"    Progress: {i+1}/{len(timestep_files)} files loaded ({100*(i+1)/len(timestep_files):.1f}%)")
        
        fields = load_field_data(file)
        if fields:
            all_data.append(fields)
            timesteps.append(extract_timestep(file))
    
    if not all_data:
        print("No valid timestep data found")
        return
    
    # Choose field for animation
    field_to_animate = 'R1' if 'R1' in all_data[0] else list(all_data[0].keys())[0]
    print(f"  Creating animation for field: {field_to_animate}")
    
    # Set up animation
    fig, ax = plt.subplots(figsize=(8, 8))
    data_range = [
        min(np.min(data[field_to_animate]) for data in all_data),
        max(np.max(data[field_to_animate]) for data in all_data)
    ]
    
    im = ax.imshow(all_data[0][field_to_animate], 
                  vmin=data_range[0], vmax=data_range[1],
                  cmap='RdBu_r', origin='lower', interpolation='bilinear')
    
    ax.set_title(f'{field_to_animate} Evolution')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    plt.colorbar(im, ax=ax)
    
    def animate(frame):
        im.set_array(all_data[frame][field_to_animate])
        ax.set_title(f'{field_to_animate} at timestep {timesteps[frame]}')
        return [im]
    
    # Create and save animation
    print("  Generating animation frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(all_data), 
                                 interval=200, blit=True, repeat=True)
    
    gif_path = OUTPUT_DIR / f'{field_to_animate}_evolution.gif'
    print(f"  Saving animation to: {gif_path}")
    anim.save(gif_path, writer='pillow', fps=5)
    print(f"  Animation saved successfully!")
    
    plt.close()

# Main visualization code
if __name__ == "__main__":
    print("="*60)
    print("SIMULATION VISUALIZATION STARTING")
    print("="*60)
    
    # Set non-interactive backend for efficiency
    import matplotlib
    matplotlib.use('Agg')
    
    total_steps = 6
    
    print_progress(1, total_steps, "Initializing visualization...")
    print(f"Looking for files in: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory {DATA_DIR} does not exist!")
        exit()
    
    print_progress(2, total_steps, "Searching for output files...")
    final_files, vals_files, timestep_files = find_output_files()
    
    print(f"Found {len(final_files)} final field files")
    print(f"Found {len(vals_files)} values per loop files") 
    print(f"Found {len(timestep_files)} timestep files")
    
    # Execute visualization steps
    if vals_files:
        print_progress(3, total_steps, "Creating energy evolution plots...")
        plot_energy_evolution(vals_files)
    else:
        print_progress(3, total_steps, "Skipping energy plots (no data files found)")
    
    if final_files:
        print_progress(4, total_steps, "Creating final field configuration plots...")
        for i, final_file in enumerate(final_files):
            print(f"  Processing final field file {i+1}/{len(final_files)}: {final_file.name}")
            fields = load_field_data(final_file)
            if fields:
                plot_field_snapshot(fields, 
                                   title=f"Final Field Configuration - {final_file.stem}",
                                   save_name=f"final_fields_{final_file.stem}")
            else:
                print(f"    Skipped: Could not load {final_file.name}")
    else:
        print_progress(4, total_steps, "Skipping final field plots (no data files found)")
    
    if timestep_files:
        print_progress(5, total_steps, "Creating animations...")
        create_animation(timestep_files)
    else:
        print_progress(5, total_steps, "Skipping animations (no timestep files found)")
    
    print_progress(6, total_steps, "Visualization completed!")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)