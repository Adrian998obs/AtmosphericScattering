import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation

def test_RW():
    """Test the random walk simulation of photon packets in the atmosphere."""
    atmosphere = simulation.Atmosphere()
    photon = simulation.PhotonPacket(position=np.array([5,5,5]))
    while atmosphere.in_box(photon.position()):
        photon.random_walk()
        photon.move()

    cell_size = atmosphere.cell_size()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(photon.trajectory()[:,0],
        photon.trajectory()[:,1],
        photon.trajectory()[:,2],
        color='r', linewidth=2, label='Photon Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape()[0]*cell_size)
    ax.set_ylim(0, atmosphere.shape()[1]*cell_size)
    ax.set_zlim(0, atmosphere.shape()[2]*cell_size)
    plt.savefig('/home/localuser/Documents/MC_RAD/AtmosphericScattering/figures/test_randomWalk.png')
    plt.close()

def test_mfp():

    """Test the mean free path calculation and luminosity deposition."""
    lengths = np.array([])
    N = 10000
    for _ in range(N):
        photon = simulation.PhotonPacket(position=np.array([0,0,5.5]), wavelength=650e-9)
        photon.random_walk()
        length = photon.optical_length()
        lengths = np.append(lengths, length)

    mean_length = np.mean(lengths)
    theoretical_mfp = 1 / photon.scattering_coefficient()
    print(f"Theoretical Mean Free Path of Photon with wavelength {photon.wavelength()*1e9:.0f} nm: {theoretical_mfp*1e-3:.4f} km")
    print(f"Mean free path length of {N} realizations: {mean_length*1e-3:.4f} km")
    assert np.isclose(mean_length, theoretical_mfp, rtol=0.1), "Mean free path does not match theoretical value within 10%."
    
if __name__ == "__main__":
    test_RW()
    print("Random Walk test completed successfully.")
    test_mfp()
    print("Mean Free Path test completed successfully.")