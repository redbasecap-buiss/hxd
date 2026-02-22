#!/usr/bin/env python3
"""
Simple LBM test with Taylor-Green vortex for validation.
Uses periodic boundaries to avoid complex boundary condition issues.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleLBM:
    """Simple 2D LBM with periodic boundaries"""
    
    def __init__(self, nx, ny, tau):
        self.nx, self.ny = nx, ny
        self.tau = tau
        
        # D2Q9 lattice
        self.e = np.array([
            [0,  1,  0, -1,  0,  1, -1, -1,  1],
            [0,  0,  1,  0, -1,  1,  1, -1, -1]
        ], dtype=np.float64)
        
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Initialize with Taylor-Green vortex
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        kx, ky = 2*np.pi/nx, 2*np.pi/ny
        
        self.rho = np.ones((ny, nx)) + 0.01*np.cos(kx*x)*np.cos(ky*y)
        self.u = np.zeros((2, ny, nx))
        self.u[0] = 0.1*np.sin(kx*x)*np.cos(ky*y)  # u
        self.u[1] = -0.1*np.cos(kx*x)*np.sin(ky*y)  # v
        
        # Initialize distributions
        self.f = np.zeros((9, ny, nx))
        self.f_new = np.zeros((9, ny, nx))
        
        # Set to equilibrium
        for i in range(9):
            eu = self.e[0,i]*self.u[0] + self.e[1,i]*self.u[1]
            usq = self.u[0]**2 + self.u[1]**2
            self.f[i] = self.w[i] * self.rho * (1 + 3*eu + 4.5*eu**2 - 1.5*usq)
        
    def step(self):
        """Single LBM step with periodic boundaries"""
        # Collision
        self.rho = np.sum(self.f, axis=0)
        self.u[0] = np.sum(self.f * self.e[0,:,None,None], axis=0) / self.rho
        self.u[1] = np.sum(self.f * self.e[1,:,None,None], axis=0) / self.rho
        
        # Equilibrium
        for i in range(9):
            eu = self.e[0,i]*self.u[0] + self.e[1,i]*self.u[1]
            usq = self.u[0]**2 + self.u[1]**2
            feq = self.w[i] * self.rho * (1 + 3*eu + 4.5*eu**2 - 1.5*usq)
            self.f_new[i] = self.f[i] - (self.f[i] - feq)/self.tau
        
        # Streaming with periodic boundaries
        for i in range(9):
            self.f[i] = np.roll(np.roll(self.f_new[i], self.e[0,i], axis=1), self.e[1,i], axis=0)

def test_simple_lbm():
    """Test simple LBM with Taylor-Green vortex decay"""
    print("🧪 Testing Simple LBM (Taylor-Green Vortex)...")
    
    lbm = SimpleLBM(64, 64, 1.0)
    
    initial_energy = np.sum(lbm.u[0]**2 + lbm.u[1]**2)
    initial_rho_avg = np.mean(lbm.rho)
    
    for step in range(100):
        lbm.step()
        
        if step % 20 == 0:
            energy = np.sum(lbm.u[0]**2 + lbm.u[1]**2)
            rho_avg = np.mean(lbm.rho)
            max_u = np.max(np.sqrt(lbm.u[0]**2 + lbm.u[1]**2))
            print(f"  Step {step}: Energy={energy:.6f}, <ρ>={rho_avg:.6f}, max|u|={max_u:.6f}")
    
    final_energy = np.sum(lbm.u[0]**2 + lbm.u[1]**2) 
    final_rho_avg = np.mean(lbm.rho)
    
    # Energy should decay due to viscosity
    assert final_energy < initial_energy, "Energy should decay"
    
    # Mass should be conserved
    assert abs(final_rho_avg - initial_rho_avg) < 0.01, "Mass not conserved"
    
    # Check for reasonable values (no blowup)
    assert np.all(np.isfinite(lbm.rho)), "Density contains NaN/Inf"
    assert np.all(np.isfinite(lbm.u)), "Velocity contains NaN/Inf"
    
    print(f"  ✅ Simple LBM OK: Energy decay {initial_energy:.6f}→{final_energy:.6f}")
    return True

if __name__ == "__main__":
    test_simple_lbm()
    print("🎉 Simple LBM test passed!")