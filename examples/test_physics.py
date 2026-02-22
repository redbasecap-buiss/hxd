#!/usr/bin/env python3
"""
Quick physics validation test for the LBM simulations.
Runs shortened versions to verify the implementations produce reasonable results.
"""

import numpy as np
import sys
import os

# Add current directory to path to import the simulation modules
sys.path.append(os.path.dirname(__file__))

def test_vortex_street():
    """Test that vortex street simulation produces expected flow patterns"""
    print("🧪 Testing Vortex Street Physics...")
    
    # Import and modify the LBM class for testing
    from vortex_street import LBM_D2Q9
    
    # Smaller grid for faster testing
    nx, ny = 100, 25
    Re = 100
    u_inlet = 0.05
    nu = u_inlet * (ny//10) / Re
    tau = 3*nu + 0.5
    
    lbm = LBM_D2Q9(nx, ny, tau, u_inlet)
    
    # Run simulation for a short time
    for step in range(1000):
        lbm.step()
        u_mag = np.sqrt(lbm.u[0]**2 + lbm.u[1]**2)
        if step % 200 == 0:
            print(f"  Step {step}: max|u|={np.max(u_mag):.4f}, max|ω|={np.max(np.abs(lbm.vorticity)):.4f}")
    
    # Check physics
    max_velocity = np.max(u_mag)
    max_vorticity = np.max(np.abs(lbm.vorticity))
    
    # Should have reasonable velocity magnitudes
    assert 0.001 < max_velocity < 0.2, f"Velocity magnitude {max_velocity} out of expected range"
    
    # Should develop some vorticity around the cylinder
    assert max_vorticity > 0.001, f"Vorticity {max_vorticity} too small - no flow development?"
    
    # Mass conservation check (density should stay close to 1)
    avg_density = np.mean(lbm.rho)
    assert 0.95 < avg_density < 1.05, f"Density {avg_density} deviates too much from 1"
    
    print(f"  ✅ Vortex street physics OK: |u|_max={max_velocity:.4f}, |ω|_max={max_vorticity:.4f}")
    return True

def test_lid_cavity():
    """Test that lid-driven cavity produces expected recirculation"""
    print("🧪 Testing Lid-Driven Cavity Physics...")
    
    # Import and modify the cavity class
    from lid_cavity import LBM_Cavity
    
    # Smaller grid for testing
    n = 64
    Re = 100
    u_lid = 0.05
    nu = u_lid * n / Re
    tau = 3*nu + 0.5
    
    lbm = LBM_Cavity(n, tau, u_lid)
    
    # Run simulation 
    for step in range(2000):
        lbm.step()
        if step % 400 == 0:
            print(f"  Step {step}: max|u|={np.max(lbm.u_mag):.4f}")
    
    # Check physics
    max_velocity = np.max(lbm.u_mag)
    
    # Should have reasonable velocity 
    assert 0.001 < max_velocity < 0.15, f"Velocity {max_velocity} out of expected range"
    
    # Check that top boundary has horizontal velocity (moving lid)
    top_row_u = lbm.u[0, -1, :]
    avg_top_u = np.mean(top_row_u[5:-5])  # Exclude corners
    assert avg_top_u > 0.02, f"Top wall velocity {avg_top_u} too low"
    
    # Check mass conservation
    avg_density = np.mean(lbm.rho)
    assert 0.95 < avg_density < 1.05, f"Density {avg_density} deviates too much"
    
    # Check for recirculation (velocity should change sign vertically)
    center_col = lbm.u[1, :, n//2]  # v-velocity at center
    has_recirculation = np.any(center_col > 0.001) and np.any(center_col < -0.001)
    assert has_recirculation, "No recirculation detected in cavity"
    
    print(f"  ✅ Cavity physics OK: |u|_max={max_velocity:.4f}, top_u={avg_top_u:.4f}")
    return True

def main():
    """Run physics validation tests"""
    print("🌊 FLUX LBM Physics Validation")
    print("=" * 40)
    
    try:
        test_vortex_street()
        test_lid_cavity()
        
        print("\n🎉 All physics tests passed!")
        print("✨ The LBM implementations produce physically reasonable results.")
        return True
        
    except Exception as e:
        print(f"\n❌ Physics test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)