import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation


def test_length_ray():
    """Test the computation of lengths in atmosphere cells for a vertical ray."""
    # Create a simple atmosphere
    atmosphere = simulation.Atmosphere()
    # Create a photon packet
    photon = simulation.PhotonPacket()
    
    # Vertical ray
    photon.optical_depth = 100
    photon.theta = 0
    photon.phi = 0
    
    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    expected_length = 10
    assert np.sum(lengths) == expected_length, f"Sum of lengths in cells should equal the photon's optical depth, got {np.sum(lengths)} instead of {expected_length}"

    # Diagonal ray
    photon.position = np.array([0, 0, 0])
    photon.theta = np.pi / 4 
    photon.phi = np.pi / 4
    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    expected_length = np.sqrt(2) * 10
    assert np.abs(np.sum(lengths) - expected_length) < 1e-6, f"Sum of lengths in cells should equal the photon's optical depth for diagonal ray, got {np.sum(lengths)} instead of {expected_length}"

def test_plot_atmosphere():
    """Test the plotting of the voxels with lengths traversed by a photon."""
    atmosphere = simulation.Atmosphere()
    photon = simulation.PhotonPacket()
    photon.theta = 0.9553166181245092 # ~54.7356 degrees
    photon.phi = np.pi / 4
    dir = photon.direction_in_cartesian(photon.theta, photon.phi)
    photon.optical_depth = atmosphere.distance_to_boundary(photon.position, dir)

    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    photon.move()
    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.min(lengths), vmax=np.max(lengths))
    facecolors = cm.rainbow_r(norm(lengths))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(lengths, facecolors=facecolors, edgecolor='k', alpha=0.5)


    ax.plot(photon.trajectory[:,0],
        photon.trajectory[:,1],
        photon.trajectory[:,2],
        color='r', linewidth=3, label='Ray Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape[0])
    ax.set_ylim(0, atmosphere.shape[1])
    ax.set_zlim(0, atmosphere.shape[2])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='Length in Cell')
    plt.savefig('../figures/test_plot_atmosphere.png')
    plt.close()

def test_plot_sourceFunction():
    """Test the plotting of the source function."""
    atmosphere = simulation.Atmosphere()
    photon = simulation.PhotonPacket(position=np.array([0,0,5.5]))
    photon.theta = np.pi / 2
    
    dir = photon.direction_in_cartesian(photon.theta, photon.phi)
    photon.optical_depth = np.min([atmosphere.distance_to_boundary(photon.position, dir), photon.maximum_optical_depth()])

    atmosphere.deposit_energy(photon)
    photon.move()
    source_function = atmosphere.source_function
    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.min(source_function), vmax=np.max(source_function))
    facecolors = cm.rainbow_r(norm(source_function))

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(source_function, facecolors=facecolors, edgecolor='k', alpha=0.5)


    ax.plot(photon.trajectory[:,0],
        photon.trajectory[:,1],
        photon.trajectory[:,2],
        color='r', linewidth=3, label='Ray Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape[0])
    ax.set_ylim(0, atmosphere.shape[1])
    ax.set_zlim(0, atmosphere.shape[2])
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='intensity')
    plt.savefig('../figures/test_plot_sourceFunction.png')
    plt.close()


if __name__ == "__main__":
    test_length_ray()
    print("test_length_ray passed.")
    test_plot_atmosphere()
    print("test_plot_atmosphere passed.")
    test_plot_sourceFunction()
    print("test_plot_sourceFunction passed.")

