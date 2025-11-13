import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation


def test_length_ray():
    """Test the computation of lengths in atmosphere cells for a vertical ray."""
    # Create a simple atmosphere
    atmosphere = simulation.Atmosphere(cell_size=2)
    # Create a photon packet
    photon = simulation.PhotonPacket()
    
    # Vertical ray
    photon._optical_depth = 100
    photon._theta = 0
    photon._phi = 0
    
    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    expected_length = atmosphere.cell_size() * atmosphere.shape[0]
    assert np.sum(lengths) == expected_length, f"Sum of lengths in cells should equal the photon's optical depth, got {np.sum(lengths)} instead of {expected_length}"

    # Diagonal ray
    photon._position = np.array([0, 0, 0])
    photon._theta = np.pi / 4
    photon._phi = np.pi / 4
    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    expected_length = np.sqrt(2) * atmosphere.cell_size() * atmosphere.shape[0]
    assert np.abs(np.sum(lengths) - expected_length) < 1e-6, f"Sum of lengths in cells should equal the photon's optical depth for diagonal ray, got {np.sum(lengths)} instead of {expected_length}"

def test_plot_atmosphere():
    """Test the plotting of the voxels with lengths traversed by a photon."""
    atmosphere = simulation.Atmosphere(shape=(10, 10, 10), cell_size=2)
    photon = simulation.PhotonPacket()
    photon._theta = 0.9553166181245092  # ~54.7356 degrees
    photon._phi = np.pi / 4
    dir = photon.direction_in_cartesian(photon._theta, photon._phi)
    photon._optical_length = atmosphere.distance_to_boundary(photon._position, dir)

    lengths = atmosphere.deposit_energy(photon, return_lengths=True)
    photon.move()

    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.min(lengths), vmax=np.max(lengths))
    facecolors = cm.rainbow_r(norm(lengths))

    nx, ny, nz = lengths.shape
    cell_size = atmosphere.cell_size()

    x = np.arange(0, (nx + 1) * cell_size, cell_size)
    y = np.arange(0, (ny + 1) * cell_size, cell_size)
    z = np.arange(0, (nz + 1) * cell_size, cell_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(X, Y, Z, lengths > 0, facecolors=facecolors, edgecolor='k', alpha=0.5)

    ax.plot(
        photon.trajectory()[:, 0],
        photon.trajectory()[:, 1],
        photon.trajectory()[:, 2],
        color='r', linewidth=3, label='Ray Path'
    )
    Lx, Ly, Lz = np.array(atmosphere.shape) * cell_size
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_box_aspect([Lx, Ly, Lz])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap='rainbow_r'),
        ax=ax, shrink=0.5, aspect=5, label='Length in Cell'
    )

    plt.close()

def test_plot_sourceFunction():
    """Test the plotting of the source function."""
    atmosphere = simulation.Atmosphere(cell_size=1e4)
    photon = simulation.PhotonPacket(position=np.array([0,0,10.5]))
    photon._theta = np.pi / 2

    dir = photon.direction_in_cartesian(photon._theta, photon._phi)
    photon._optical_length = np.min([atmosphere.distance_to_boundary(photon.position(), dir), photon.maximum_optical_depth()/photon._scattering_coefficient])
    print("Distance to boundary :",atmosphere.distance_to_boundary(photon.position(), dir), " maximal ", photon.maximum_optical_depth(), "optical length:", photon._optical_length)
    atmosphere.deposit_energy(photon)
    photon.move()
    source_function = atmosphere.source_function()
    print(source_function[source_function>0])
    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.min(source_function), vmax=np.max(source_function))
    facecolors = cm.rainbow_r(norm(source_function))

    nx, ny, nz = source_function.shape
    cell_size = atmosphere.cell_size()

    x = np.arange(0, (nx + 1) * cell_size, cell_size)
    y = np.arange(0, (ny + 1) * cell_size, cell_size)
    z = np.arange(0, (nz + 1) * cell_size, cell_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(X, Y, Z, source_function, facecolors=facecolors, edgecolor='k', alpha=0.5)


    ax.plot(photon.trajectory()[:,0],
        photon.trajectory()[:,1],
        photon.trajectory()[:,2],
        color='r', linewidth=3, label='Ray Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape[0] * atmosphere._cell_sizes)
    ax.set_ylim(0, atmosphere.shape[1] * atmosphere._cell_sizes)
    ax.set_zlim(0, atmosphere.shape[2] * atmosphere._cell_sizes)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='intensity')
    plt.savefig('/home/localuser/Documents/MC_RAD/AtmosphericScattering/figures/test_plot_sourceFunction.png')
    plt.show()


if __name__ == "__main__":
    test_length_ray()
    print("test_length_ray passed.")
    test_plot_atmosphere()
    print("test_plot_atmosphere passed.")
    test_plot_sourceFunction()
    print("test_plot_sourceFunction passed.")

