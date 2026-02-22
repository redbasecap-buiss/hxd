#!/usr/bin/env python3
"""
Lattice Boltzmann Method - Vortex Street Simulation
===================================================

Beautiful visualization of flow around a cylinder using D2Q9 LBM.
Generates the classic von Kármán vortex street.

Theory:
- 2D Lattice Boltzmann Method with D2Q9 velocity set
- BGK collision operator with single relaxation time
- Bounce-back boundary conditions for solid obstacles
- Zou-He velocity boundary conditions at inlet

References:
- Krüger et al. (2017): The Lattice Boltzmann Method: Principles and Practice
- Zou & He (1997): On pressure and velocity boundary conditions for LBM BGK model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time

class LBM_D2Q9:
    """2D Lattice Boltzmann Method with D2Q9 velocity set"""
    
    def __init__(self, nx, ny, tau, u_inlet=0.1):
        self.nx, self.ny = nx, ny
        self.tau = tau  # Relaxation time
        self.u_inlet = u_inlet
        
        # D2Q9 velocity set (9 velocities in 2D)
        self.e = np.array([
            [0,  1,  0, -1,  0,  1, -1, -1,  1],  # x-components
            [0,  0,  1,  0, -1,  1,  1, -1, -1]   # y-components
        ], dtype=np.float32)
        
        # D2Q9 weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
        
        # Opposite directions (for bounce-back)
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Initialize distribution functions
        self.f = np.ones((9, ny, nx), dtype=np.float32)
        self.f_new = np.ones((9, ny, nx), dtype=np.float32)
        
        # Macroscopic fields
        self.rho = np.ones((ny, nx), dtype=np.float32)
        self.u = np.zeros((2, ny, nx), dtype=np.float32)
        self.vorticity = np.zeros((ny, nx), dtype=np.float32)
        
        # Create cylinder obstacle
        self.obstacle = self.create_cylinder()
        
        # Initialize equilibrium
        self.compute_macroscopic()
        self.f = self.compute_equilibrium()
        
    def create_cylinder(self, cx=None, cy=None, radius=None):
        """Create circular obstacle (cylinder)"""
        if cx is None: cx = self.nx // 4  # Cylinder at x = nx/4
        if cy is None: cy = self.ny // 2  # Cylinder at y = ny/2  
        if radius is None: radius = self.ny // 10  # Radius = ny/10
        
        x, y = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        obstacle = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Store cylinder parameters for visualization
        self.cyl_x, self.cyl_y, self.cyl_r = cx, cy, radius
        
        return obstacle
    
    def compute_macroscopic(self):
        """Compute density and velocity from distribution functions"""
        self.rho = np.sum(self.f, axis=0)
        
        # Avoid division by zero
        rho_safe = np.maximum(self.rho, 1e-12)
        
        # Compute velocity
        for i in range(2):
            self.u[i] = np.sum(self.f * self.e[i][:, None, None], axis=0) / rho_safe
    
    def compute_equilibrium(self):
        """Compute equilibrium distribution functions and return them"""
        # Expand arrays for vectorized computation  
        e_u = np.zeros((9, self.ny, self.nx), dtype=np.float32)
        for i in range(9):
            e_u[i] = self.e[0, i] * self.u[0] + self.e[1, i] * self.u[1]  # e_i · u
        
        u_sq = np.sum(self.u**2, axis=0)  # |u|²
        
        # Maxwell-Boltzmann equilibrium (truncated to 2nd order)
        f_eq = np.zeros_like(self.f)
        for i in range(9):
            f_eq[i] = self.w[i] * self.rho * (
                1 + 3*e_u[i] + 4.5*e_u[i]**2 - 1.5*u_sq
            )
        
        return f_eq
    
    def collision(self):
        """BGK collision step"""
        # Compute equilibrium distribution
        f_eq = self.compute_equilibrium()
        
        # BGK collision: f_new = f - (f - f_eq) / tau
        omega = 1.0 / self.tau
        self.f_new = self.f - omega * (self.f - f_eq)
    
    def streaming(self):
        """Stream distribution functions to neighboring nodes"""
        for i in range(9):
            # Stream in direction e_i
            self.f[i] = np.roll(np.roll(self.f_new[i], self.e[0, i], axis=1), self.e[1, i], axis=0)
    
    def boundary_conditions(self):
        """Apply boundary conditions"""
        # Bounce-back on obstacle
        for i in range(9):
            self.f[i, self.obstacle] = self.f_new[self.opp[i], self.obstacle]
        
        # Velocity inlet (left boundary) - Zou-He method
        rho_inlet = 1.0
        u_inlet = np.array([self.u_inlet, 0.0])
        
        # Left boundary (x=0)
        x = 0
        for y in range(self.ny):
            if not self.obstacle[y, x]:
                # Unknown distributions: f1, f5, f8
                # Known: f0, f2, f3, f4, f6, f7
                self.f[1, y, x] = self.f[3, y, x] + (2/3) * rho_inlet * u_inlet[0]
                self.f[5, y, x] = self.f[7, y, x] + (1/6) * rho_inlet * u_inlet[0] - 0.5 * (self.f[2, y, x] - self.f[4, y, x])
                self.f[8, y, x] = self.f[6, y, x] + (1/6) * rho_inlet * u_inlet[0] + 0.5 * (self.f[2, y, x] - self.f[4, y, x])
        
        # Open boundary (right boundary) - Zero gradient
        self.f[:, :, -1] = self.f[:, :, -2]
        
        # Wall boundaries (top/bottom) - Bounce-back
        for i in range(9):
            self.f[i, 0, :] = self.f_new[self.opp[i], 0, :]  # Bottom wall
            self.f[i, -1, :] = self.f_new[self.opp[i], -1, :]  # Top wall
    
    def compute_vorticity(self):
        """Compute vorticity ω = ∂v/∂x - ∂u/∂y"""
        # Use central differences with periodic boundaries
        dudx = np.gradient(self.u[0], axis=1)
        dudy = np.gradient(self.u[0], axis=0)
        dvdx = np.gradient(self.u[1], axis=1)
        dvdy = np.gradient(self.u[1], axis=0)
        
        self.vorticity = dvdx - dudy
    
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
        self.compute_vorticity()

def run_simulation():
    """Run the vortex street simulation and create animation"""
    print("🌊 Starting Lattice Boltzmann Vortex Street Simulation")
    print("=" * 60)
    
    # Simulation parameters
    nx, ny = 400, 100  # Grid size
    Re = 100  # Reynolds number
    u_inlet = 0.05  # Inlet velocity (low Mach number)
    
    # LBM parameters
    nu = u_inlet * (ny//10) / Re  # Kinematic viscosity
    tau = 3*nu + 0.5  # Relaxation time
    
    print(f"Grid: {nx} × {ny}")
    print(f"Reynolds number: {Re}")
    print(f"Inlet velocity: {u_inlet:.3f}")
    print(f"Relaxation time: {tau:.3f}")
    print(f"Kinematic viscosity: {nu:.6f}")
    
    # Initialize simulation
    lbm = LBM_D2Q9(nx, ny, tau, u_inlet)
    
    # Animation setup
    fig, ax = plt.subplots(figsize=(16, 6), dpi=120)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Colormap for vorticity
    vmax = 0.1
    im = ax.imshow(lbm.vorticity, extent=[0, nx, 0, ny], 
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax, 
                   origin='lower', interpolation='bicubic')
    
    # Add cylinder
    cylinder = Circle((lbm.cyl_x, lbm.cyl_y), lbm.cyl_r, 
                     color='black', zorder=10)
    ax.add_patch(cylinder)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Vorticity ω [1/s]', fontsize=12)
    
    # Title
    title = ax.set_title('Lattice Boltzmann Method — von Kármán Vortex Street', 
                        fontsize=14, fontweight='bold')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Animation function
    def animate(frame):
        print(f"\rFrame {frame:4d}/500", end='', flush=True)
        
        # Run multiple LBM steps per frame
        for _ in range(20):
            lbm.step()
        
        # Update visualization
        im.set_array(lbm.vorticity)
        time_text.set_text(f'Time step: {frame*20:,}')
        
        return [im, time_text]
    
    print("\n🎬 Creating animation...")
    start_time = time.time()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=500, 
                                  interval=50, blit=False, repeat=False)
    
    # Save as MP4
    output_file = "vortex_street.mp4"
    print(f"\n💾 Saving to {output_file}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='FLUX LBM'), bitrate=8000)
    anim.save(output_file, writer=writer)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Animation complete! ({elapsed:.1f}s)")
    print(f"📁 Output: {output_file}")
    
    # Show final frame
    plt.tight_layout()
    plt.savefig("vortex_street_final.png", dpi=300, bbox_inches='tight')
    print("📸 Final frame saved as vortex_street_final.png")

if __name__ == "__main__":
    run_simulation()