#!/usr/bin/env python3
"""
Lattice Boltzmann Method - Lid-Driven Cavity Flow
==================================================

Classical CFD benchmark: flow in a square cavity with moving top lid.
Visualizes velocity magnitude and streamlines.

Theory:
- Square domain with no-slip walls on 3 sides
- Top wall moves with constant horizontal velocity
- Creates characteristic recirculating flow patterns
- Benchmark case from Ghia et al. (1982)

References:
- Ghia, U., Ghia, K.N., & Shin, C.T. (1982): High-Re solutions for incompressible flow
- Krüger et al. (2017): The Lattice Boltzmann Method: Principles and Practice
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import time

class LBM_Cavity:
    """2D Lattice Boltzmann Method for lid-driven cavity"""
    
    def __init__(self, n, tau, u_lid=0.1):
        self.n = n  # Grid size (n × n)
        self.tau = tau  # Relaxation time
        self.u_lid = u_lid  # Lid velocity
        
        # D2Q9 velocity set
        self.e = np.array([
            [0,  1,  0, -1,  0,  1, -1, -1,  1],  # x-components
            [0,  0,  1,  0, -1,  1,  1, -1, -1]   # y-components
        ], dtype=np.float32)
        
        # D2Q9 weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
        
        # Opposite directions
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Initialize distribution functions
        self.f = np.ones((9, n, n), dtype=np.float32) * self.w[:, None, None]
        self.f_new = np.ones((9, n, n), dtype=np.float32)
        
        # Macroscopic fields
        self.rho = np.ones((n, n), dtype=np.float32)
        self.u = np.zeros((2, n, n), dtype=np.float32)
        self.u_mag = np.zeros((n, n), dtype=np.float32)
        
        # Wall masks
        self.create_walls()
        
    def create_walls(self):
        """Create wall boundary masks"""
        n = self.n
        
        # Wall boundaries
        self.wall_bottom = np.zeros((n, n), dtype=bool)
        self.wall_top = np.zeros((n, n), dtype=bool) 
        self.wall_left = np.zeros((n, n), dtype=bool)
        self.wall_right = np.zeros((n, n), dtype=bool)
        
        self.wall_bottom[0, :] = True  # Bottom wall
        self.wall_top[-1, :] = True    # Top wall (moving lid)
        self.wall_left[:, 0] = True    # Left wall
        self.wall_right[:, -1] = True  # Right wall
        
        # Combined wall mask
        self.walls = self.wall_bottom | self.wall_left | self.wall_right
        self.moving_lid = self.wall_top
        
    def compute_macroscopic(self):
        """Compute density and velocity from distribution functions"""
        self.rho = np.sum(self.f, axis=0)
        
        # Avoid division by zero
        rho_safe = np.maximum(self.rho, 1e-12)
        
        # Compute velocity
        for i in range(2):
            self.u[i] = np.sum(self.f * self.e[i][:, None, None], axis=0) / rho_safe
        
        # Velocity magnitude
        self.u_mag = np.sqrt(self.u[0]**2 + self.u[1]**2)
        
    def compute_equilibrium(self):
        """Compute equilibrium distribution functions"""
        # Compute e_i · u for each direction
        e_u = np.zeros((9, self.n, self.n), dtype=np.float32)
        for i in range(9):
            e_u[i] = self.e[0, i] * self.u[0] + self.e[1, i] * self.u[1]  # e_i · u
        
        u_sq = np.sum(self.u**2, axis=0)  # |u|²
        
        # Maxwell-Boltzmann equilibrium
        f_eq = np.zeros_like(self.f)
        for i in range(9):
            f_eq[i] = self.w[i] * self.rho * (
                1 + 3*e_u[i] + 4.5*e_u[i]**2 - 1.5*u_sq
            )
        
        return f_eq
    
    def collision(self):
        """BGK collision step"""
        f_eq = self.compute_equilibrium()
        
        # BGK collision: f_new = f - omega * (f - f_eq)
        omega = 1.0 / self.tau
        self.f_new = self.f - omega * (self.f - f_eq)
    
    def streaming(self):
        """Stream distribution functions"""
        for i in range(9):
            # Periodic streaming (will be overridden by boundary conditions)
            self.f[i] = np.roll(np.roll(self.f_new[i], self.e[0, i], axis=1), self.e[1, i], axis=0)
    
    def boundary_conditions(self):
        """Apply boundary conditions"""
        # Bottom, left, and right walls: bounce-back
        for i in range(9):
            self.f[i, self.walls] = self.f_new[self.opp[i], self.walls]
        
        # Moving lid (top wall): bounce-back with velocity
        # This is a simplified moving wall BC
        rho_wall = 1.0
        u_wall = np.array([self.u_lid, 0.0])
        
        # Apply moving wall BC at top boundary
        y = self.n - 1  # Top row
        for x in range(self.n):
            # Bounce-back with added momentum for moving wall
            # Simplified approach: add 2 * rho * u_wall · e_i to reflected distributions
            for i in range(9):
                if self.moving_lid[y, x]:
                    momentum_transfer = 2 * rho_wall * np.dot(u_wall, self.e[:, i])
                    self.f[i, y, x] = self.f_new[self.opp[i], y, x] + self.w[i] * momentum_transfer
    
    def step(self):
        """Single LBM time step"""
        # 1. Collision
        self.collision()
        
        # 2. Streaming  
        self.streaming()
        
        # 3. Boundary conditions
        self.boundary_conditions()
        
        # 4. Update macroscopic quantities
        self.compute_macroscopic()
    
    def get_streamlines_data(self, density=1.5):
        """Compute streamlines for visualization"""
        # Create coordinate grids
        x = np.linspace(0, self.n-1, self.n)
        y = np.linspace(0, self.n-1, self.n)
        X, Y = np.meshgrid(x, y)
        
        # Velocity components (transpose for correct orientation)
        U = self.u[0].T
        V = self.u[1].T
        
        # Starting points for streamlines
        start_x = np.linspace(1, self.n-2, int(density * self.n/10))
        start_y = np.linspace(1, self.n-2, int(density * self.n/10))
        
        return X, Y, U, V, start_x, start_y

def run_cavity_simulation():
    """Run lid-driven cavity simulation with streamline visualization"""
    print("🌊 Starting Lattice Boltzmann Lid-Driven Cavity Simulation")
    print("=" * 65)
    
    # Simulation parameters
    n = 128  # Grid size (n × n)
    Re = 400  # Reynolds number
    u_lid = 0.05  # Lid velocity
    
    # LBM parameters  
    nu = u_lid * n / Re  # Kinematic viscosity
    tau = 3*nu + 0.5  # Relaxation time
    
    print(f"Grid: {n} × {n}")
    print(f"Reynolds number: {Re}")
    print(f"Lid velocity: {u_lid:.3f}")
    print(f"Relaxation time: {tau:.3f}")
    print(f"Kinematic viscosity: {nu:.6f}")
    
    # Initialize simulation
    lbm = LBM_Cavity(n, tau, u_lid)
    
    # Animation setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    
    # Left plot: Velocity magnitude
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, n)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Velocity Magnitude', fontsize=14, fontweight='bold')
    
    # Right plot: Streamlines
    ax2.set_xlim(0, n)
    ax2.set_ylim(0, n)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Streamlines', fontsize=14, fontweight='bold')
    
    # Initial plots
    vmax = 0.15
    im1 = ax1.imshow(lbm.u_mag, extent=[0, n, 0, n], cmap='viridis',
                     vmin=0, vmax=vmax, origin='lower', interpolation='bicubic')
    
    # Add walls visualization
    wall_patch1 = patches.Rectangle((0, 0), n, 1, linewidth=0, 
                                   facecolor='gray', alpha=0.3, zorder=5)
    ax1.add_patch(wall_patch1)
    wall_patch2 = patches.Rectangle((0, 0), 1, n, linewidth=0,
                                   facecolor='gray', alpha=0.3, zorder=5)  
    ax1.add_patch(wall_patch2)
    wall_patch3 = patches.Rectangle((n-1, 0), 1, n, linewidth=0,
                                   facecolor='gray', alpha=0.3, zorder=5)
    ax1.add_patch(wall_patch3)
    
    # Moving lid
    lid_patch1 = patches.Rectangle((0, n-1), n, 1, linewidth=2,
                                  facecolor='red', alpha=0.5, zorder=5)
    ax1.add_patch(lid_patch1)
    ax1.annotate('Moving Lid →', xy=(n/2, n-0.5), xytext=(n/2, n-10),
                ha='center', va='center', fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Repeat for streamlines plot
    for ax in [ax2]:
        wall_patches = [
            patches.Rectangle((0, 0), n, 1, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((0, 0), 1, n, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((n-1, 0), 1, n, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((0, n-1), n, 1, linewidth=2, facecolor='red', alpha=0.5, zorder=5)
        ]
        for patch in wall_patches:
            ax.add_patch(patch)
        
        ax.annotate('Moving Lid →', xy=(n/2, n-0.5), xytext=(n/2, n-10),
                   ha='center', va='center', fontsize=10, color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Colorbar
    cbar = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label('Velocity |u| [m/s]', fontsize=12)
    
    # Time text
    time_text = fig.suptitle('Lattice Boltzmann Method — Lid-Driven Cavity Flow', 
                            fontsize=16, fontweight='bold')
    step_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Animation function
    def animate(frame):
        print(f"\rFrame {frame:4d}/400", end='', flush=True)
        
        # Run multiple steps per frame
        for _ in range(50):
            lbm.step()
        
        # Update velocity magnitude plot
        im1.set_array(lbm.u_mag)
        
        # Clear and redraw streamlines
        ax2.clear()
        ax2.set_xlim(0, n)
        ax2.set_ylim(0, n)
        ax2.set_aspect('equal')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Streamlines', fontsize=14, fontweight='bold')
        
        # Get streamlines data
        X, Y, U, V, start_x, start_y = lbm.get_streamlines_data(density=2.0)
        
        # Plot streamlines
        ax2.streamplot(X, Y, U, V, color='blue', density=1.5, linewidth=1, 
                      arrowsize=1, arrowstyle='->', start_points=None)
        
        # Re-add walls
        wall_patches = [
            patches.Rectangle((0, 0), n, 1, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((0, 0), 1, n, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((n-1, 0), 1, n, linewidth=0, facecolor='gray', alpha=0.3, zorder=5),
            patches.Rectangle((0, n-1), n, 1, linewidth=2, facecolor='red', alpha=0.5, zorder=5)
        ]
        for patch in wall_patches:
            ax2.add_patch(patch)
        
        ax2.annotate('Moving Lid →', xy=(n/2, n-0.5), xytext=(n/2, n-10),
                    ha='center', va='center', fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Update step counter
        step_text.set_text(f'Time step: {frame*50:,}')
        
        return [im1]
    
    print("\n🎬 Creating animation...")
    start_time = time.time()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=400, 
                                  interval=100, blit=False, repeat=False)
    
    # Save as MP4
    output_file = "lid_cavity.mp4"
    print(f"\n💾 Saving to {output_file}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='FLUX LBM'), bitrate=8000)
    anim.save(output_file, writer=writer)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Animation complete! ({elapsed:.1f}s)")
    print(f"📁 Output: {output_file}")
    
    # Save final frame
    plt.tight_layout()
    plt.savefig("lid_cavity_final.png", dpi=300, bbox_inches='tight')
    print("📸 Final frame saved as lid_cavity_final.png")

if __name__ == "__main__":
    run_cavity_simulation()