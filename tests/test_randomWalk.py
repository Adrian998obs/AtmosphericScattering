import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation

def test_RW():
    """Test the random walk simulation of photon packets in the atmosphere."""
    atmosphere = simulation.Atmosphere(shape=(100,100,100), cell_size=1e4, number_density=1.4e24)
    blue_photon = simulation.PhotonPacket(position=np.array([atmosphere.shape()[0]/2 * atmosphere.cell_size(),
                                                       atmosphere.shape()[1]/2 * atmosphere.cell_size(),
                                                       atmosphere.shape()[2]/2 * atmosphere.cell_size()]), 
                                                       wavelength=400e-9, 
                                                       number_density=atmosphere.number_density())
    while atmosphere.in_box(blue_photon.position()):
        blue_photon.random_walk()
        blue_photon.move()
    
    red_photon = simulation.PhotonPacket(position=np.array([atmosphere.shape()[0]/2 * atmosphere.cell_size(),
                                                       atmosphere.shape()[1]/2 * atmosphere.cell_size(),
                                                       atmosphere.shape()[2]/2 * atmosphere.cell_size()]), 
                                                       wavelength=700e-9, 
                                                       number_density=atmosphere.number_density())
    while atmosphere.in_box(red_photon.position()):
        red_photon.random_walk()
        red_photon.move()

    cell_size = atmosphere.cell_size()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(blue_photon.trajectory()[:,0],
        blue_photon.trajectory()[:,1],
        blue_photon.trajectory()[:,2],
        color='b', linewidth=2, label='Blue Photon Path'
    )
    ax.plot(red_photon.trajectory()[:,0],
        red_photon.trajectory()[:,1],
        red_photon.trajectory()[:,2],
        color='r', linewidth=2, label='Red Photon Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape()[0]*cell_size)
    ax.set_ylim(0, atmosphere.shape()[1]*cell_size)
    ax.set_zlim(0, atmosphere.shape()[2]*cell_size)
    plt.savefig('./figures/test_randomWalk.png')
    plt.show()

def test_mfp():
    """Test the mean free path calculation and luminosity deposition."""
    lengths = np.array([])
    N = 100000
    for _ in range(N):
        photon = simulation.PhotonPacket(position=np.array([0,0,5.5]), wavelength=380e-9, number_density=1.4e24)
        photon.random_walk()
        length = photon.optical_length()
        lengths = np.append(lengths, length)
    mean_optical_depth = np.mean(lengths * photon.scattering_coefficient())
    mean_length = np.mean(lengths)
    theoretical_mfp = 1 / photon.scattering_coefficient()
    print(f"Theoretical Mean Free Path of Photon with wavelength {photon.wavelength()*1e9:.0f} nm: {theoretical_mfp*1e-3:.4f} km")
    print(f"Mean free path length of {N} realizations: {mean_length*1e-3:.4f} km")
    print(f"Mean optical depth of {N} realizations: {mean_optical_depth:.4f}")
    print(f"Scattering Coefficient: alpha={photon.scattering_coefficient():.0e} m^-1")
    assert np.isclose(mean_length, theoretical_mfp, rtol=0.1), "Mean free path does not match theoretical value within 10%."
    
if __name__ == "__main__":
    test_RW()
    print("Random Walk test completed successfully.")
    test_mfp()
    print("Mean Free Path test completed successfully.")