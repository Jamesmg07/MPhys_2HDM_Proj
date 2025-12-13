import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the equation: 5 + 2cos(γ₁) + 2cos(γ₂) - cos(γ₁-γ₂) = 6
# Rearranged: 2cos(γ₁) + 2cos(γ₂) - cos(γ₁-γ₂) = 1
def equation(gamma2, gamma1):
    return 2*np.cos(gamma1) + 2*np.cos(gamma2) - np.cos(gamma1 - gamma2) - 1

# Solve for gamma2 given gamma1 values
gamma1_vals = np.linspace(-2*np.pi, 2*np.pi, 16000)
gamma2_solutions = []

for g1 in gamma1_vals:
    # Find multiple solutions for each gamma1 (there may be multiple gamma2 values)
    initial_guesses = np.linspace(-2*np.pi, 2*np.pi, 100)
    sols = []
    for guess in initial_guesses:
        try:
            sol = fsolve(equation, guess, args=(g1,), full_output=True)
            if sol[2] == 1 and -2*np.pi <= sol[0][0] <= 2*np.pi:  # Check if converged
                # Check if this solution is unique (not already found)
                is_new = True
                for existing_sol in sols:
                    if np.abs(sol[0][0] - existing_sol) < 0.01:
                        is_new = False
                        break
                if is_new:
                    sols.append(sol[0][0])
        except:
            pass
    
    for sol in sols:
        gamma2_solutions.append((g1, sol))

# Convert to arrays for plotting
if gamma2_solutions:
    gamma1_plot = np.array([point[0] for point in gamma2_solutions])
    gamma2_plot = np.array([point[1] for point in gamma2_solutions])
    
    # Sort by gamma1 to get smooth lines
    sorted_indices = np.argsort(gamma1_plot)
    gamma1_sorted = gamma1_plot[sorted_indices]
    gamma2_sorted = gamma2_plot[sorted_indices]

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the solution curve as a line
ax.plot(gamma1_sorted, gamma2_sorted, 'b.', markersize=0.8, alpha=1, label=r'$\Delta E = 0$')

# Add gamma1 = gamma2 dashed line
gamma_line = np.linspace(0, 2*np.pi, 100)
ax.plot(gamma_line, gamma_line, 'k--', linewidth=1.5, alpha=0.7, label=r'$\gamma_1 = \gamma_2$')

# Add scatter points with different colors
points = [
    (0, np.pi, 'red', r'$(0, \pi)$'),
    (np.pi, 0, 'darkorange', r'$(\pi, 0)$'),
    (np.pi, np.pi, 'gold', r'$(\pi, \pi)$'),
    (0, 0, 'lime', r'$(0, 0)$'),
    (np.pi/2, np.pi/2, 'blueviolet', r'$(\pi/2, \pi/2)$'),
    (np.pi/3, np.pi/3, 'deeppink', r'$(\pi/3, \pi/3)$')
]

for x, y, color, label in points:
    ax.scatter(x, y, c=color, s=100, zorder=5, linewidths=0, label=label)

# Set axis labels and title
ax.set_xlabel(r'$\gamma_1$', fontsize=14)
ax.set_ylabel(r'$\gamma_2$', fontsize=14)
ax.set_title(r'$\gamma_1$-$\gamma_2$ plane for $\Delta E = 0$', fontsize=16)

# Set ticks in multiples of π
pi_ticks = np.array([0, 1, 2]) * np.pi
pi_labels = [r'$0$', r'$\pi$', r'$2\pi$']
ax.set_xticks(pi_ticks)
ax.set_xticklabels(pi_labels)
ax.set_yticks(pi_ticks)
ax.set_yticklabels(pi_labels)

# Set axis limits
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(0, 2*np.pi)

# Add grid
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add legend outside the plot on the right
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('gamma_plane_E0.png', dpi=300, bbox_inches='tight')
plt.show()
