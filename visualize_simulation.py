import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import glob
import os
import re
from pathlib import Path

class SimulationVisualizer:
    def __init__(self, data_dir="./Data/", nx=256, ny=256, nz=1):
        self.data_dir = Path(data_dir)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nPos = nx * ny * nz
        
        # Create output directory for plots
        self.output_dir = Path("./Plots/")
        self.output_dir.mkdir(exist_ok=True)
        
    def find_output_files(self):
        """Find all output files from the simulation"""
        # Find files with the pattern
        pattern = "*_nx*_nt*_seed*_Z2.txt"
        files = list(self.data_dir.glob(pattern))
        
        # Separate different types of files
        final_files = [f for f in files if "finalFields" in f.name]
        vals_files = [f for f in files if "valsPerLoop" in f.name]
        timestep_files = [f for f in files if "fields_timestep" in f.name]
        
        return final_files, vals_files, timestep_files
    
    def load_field_data(self, filepath):
        """Load field data and reshape to 2D grid"""
        try:
            # Read the data
            data = pd.read_csv(filepath, sep=' ')
            
            # Extract field data (assuming all points are in order)
            fields = {}
            for col in data.columns:
                if col.strip():  # Skip empty columns
                    field_1d = data[col].values
                    if len(field_1d) == self.nPos:
                        # Reshape 1D array back to 2D (or 3D if nz > 1)
                        if self.nz == 1:
                            field_2d = field_1d.reshape(self.nx, self.ny)
                        else:
                            field_2d = field_1d.reshape(self.nx, self.ny, self.nz)
                        fields[col.strip()] = field_2d
            
            return fields
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def load_energy_data(self, filepath):
        """Load energy and wall detection data"""
        try:
            data = pd.read_csv(filepath, sep=' ')
            return data
        except Exception as e:
            print(f"Error loading energy data {filepath}: {e}")
            return None
    
    def plot_energy_evolution(self, vals_files):
        """Plot energy evolution over time"""
        if not vals_files:
            print("No valsPerLoop files found")
            return
            
        for vals_file in vals_files:
            data = self.load_energy_data(vals_file)
            if data is None:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Simulation Evolution - {vals_file.stem}', fontsize=14)
            
            timesteps = np.arange(len(data))
            
            # Energy plot
            if 'Energy' in data.columns:
                axes[0,0].plot(timesteps, data['Energy'])
                axes[0,0].set_xlabel('Timestep')
                axes[0,0].set_ylabel('Total Energy')
                axes[0,0].set_title('Energy Evolution')
                axes[0,0].grid(True)
            
            # Domain wall number
            if 'NDW' in data.columns:
                axes[0,1].plot(timesteps, data['NDW'])
                axes[0,1].set_xlabel('Timestep')
                axes[0,1].set_ylabel('Number of Domain Walls')
                axes[0,1].set_title('Domain Wall Count')
                axes[0,1].grid(True)
            
            # Domain wall area (simple)
            if 'ADW_Simple' in data.columns:
                axes[1,0].plot(timesteps, data['ADW_Simple'])
                axes[1,0].set_xlabel('Timestep')
                axes[1,0].set_ylabel('Domain Wall Area (Simple)')
                axes[1,0].set_title('Domain Wall Area Evolution')
                axes[1,0].grid(True)
            
            # Domain wall area (full)
            if 'ADW_Full' in data.columns:
                axes[1,1].plot(timesteps, data['ADW_Full'])
                axes[1,1].set_xlabel('Timestep')
                axes[1,1].set_ylabel('Domain Wall Area (Full)')
                axes[1,1].set_title('Domain Wall Area Evolution (Full)')
                axes[1,1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'energy_evolution_{vals_file.stem}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_field_snapshot(self, fields, title="Field Configuration", save_name=None):
        """Plot 2D snapshots of various fields"""
        if not fields:
            return
            
        # Determine number of subplots needed
        field_names = list(fields.keys())
        n_fields = len(field_names)
        
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
        
        # Custom colormap for better visualization
        colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
        cmap = LinearSegmentedColormap.from_list('custom', colors)
        
        for i, (field_name, field_data) in enumerate(fields.items()):
            if i >= nrows * ncols:
                break
                
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            
            if self.nz == 1:
                im = ax.imshow(field_data, cmap=cmap, origin='lower', interpolation='bilinear')
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                # For 3D data, plot a slice
                im = ax.imshow(field_data[:, :, self.nz//2], cmap=cmap, origin='lower', interpolation='bilinear')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
            ax.set_title(f'{field_name}')
            ax.set_xlabel('y')
            ax.set_ylabel('x')
        
        # Hide unused subplots
        for i in range(n_fields, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.output_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_animation(self, timestep_files):
        """Create animated gif from timestep files"""
        if not timestep_files:
            print("No timestep files found for animation")
            return
            
        # Sort files by timestep
        def extract_timestep(filename):
            match = re.search(r'timestep=(\d+)', filename.name)
            return int(match.group(1)) if match else 0
        
        timestep_files.sort(key=extract_timestep)
        
        # Load all timestep data
        all_data = []
        timesteps = []
        
        for file in timestep_files:
            fields = self.load_field_data(file)
            if fields:
                all_data.append(fields)
                timesteps.append(extract_timestep(file))
        
        if not all_data:
            print("No valid timestep data found")
            return
        
        # Create animation for R1 field (domain wall indicator)
        field_to_animate = 'R1'  # This shows domain walls well
        if field_to_animate not in all_data[0]:
            field_to_animate = list(all_data[0].keys())[0]  # Use first available field
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up the plot
        data_range = [np.min([np.min(data[field_to_animate]) for data in all_data]),
                     np.max([np.max(data[field_to_animate]) for data in all_data])]
        
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
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(all_data), 
                                     interval=200, blit=True, repeat=True)
        
        # Save as gif
        gif_path = self.output_dir / f'{field_to_animate}_evolution.gif'
        anim.save(gif_path, writer='pillow', fps=5)
        print(f"Animation saved as: {gif_path}")
        
        plt.show()
    
    def run_visualization(self):
        """Main function to run all visualizations"""
        print("Starting visualization...")
        print(f"Looking for files in: {self.data_dir}")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist!")
            print("Make sure your C++ simulation has run and created output files.")
            return
        
        # Find output files
        final_files, vals_files, timestep_files = self.find_output_files()
        
        print(f"Found {len(final_files)} final field files")
        print(f"Found {len(vals_files)} values per loop files") 
        print(f"Found {len(timestep_files)} timestep files")
        
        # Plot energy evolution
        if vals_files:
            print("Creating energy evolution plots...")
            self.plot_energy_evolution(vals_files)
        
        # Plot final field configurations
        if final_files:
            print("Creating final field configuration plots...")
            for final_file in final_files:
                fields = self.load_field_data(final_file)
                if fields:
                    self.plot_field_snapshot(fields, 
                                           title=f"Final Field Configuration - {final_file.stem}",
                                           save_name=f"final_fields_{final_file.stem}")
        
        # Create animations
        if timestep_files:
            print("Creating animations...")
            self.create_animation(timestep_files)
        
        print(f"All plots saved to: {self.output_dir}")

if __name__ == "__main__":
    # Initialize visualizer with your simulation parameters
    viz = SimulationVisualizer(data_dir="./Data/", nx=256, ny=256, nz=1)
    
    # Run all visualizations
    viz.run_visualization()
