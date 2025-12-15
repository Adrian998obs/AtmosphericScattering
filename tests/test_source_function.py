import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation


def test_length_ray():
    """Test the computation of lengths in atmosphere cells for a vertical ray."""
    # Create a simple atmosphere
    atmosphere = simulation.Atmosphere(shape=(10, 10, 10), cell_size=2)
    # Create a photon packet
    photon = simulation.PhotonPacket(position=np.array([0, 0, 0]))
    
    # Vertical ray
    photon.set_optical_length(100)
    photon._theta = 0
    photon._phi = 0

    lengths = atmosphere.deposit_luminosity(photon, return_lengths=True)
    expected_length = atmosphere.cell_size() * atmosphere.shape()[0]
    assert np.sum(lengths) == expected_length, f"Sum of lengths in cells should equal the photon's optical depth, got {np.sum(lengths)} instead of {expected_length}"

    # Diagonal ray
    photon._position = np.array([0, 0, 0])
    photon._theta = np.pi / 4
    photon._phi = np.pi / 4
    lengths = atmosphere.deposit_luminosity(photon, return_lengths=True)
    expected_length = np.sqrt(2) * atmosphere.cell_size() * atmosphere.shape()[0]
    assert np.abs(np.sum(lengths) - expected_length) < 1e-6, f"Sum of lengths in cells should equal the photon's optical depth for diagonal ray, got {np.sum(lengths)} instead of {expected_length}"

def test_plot_atmosphere():
    """Test the plotting of the voxels with lengths traversed by a photon."""
    atmosphere = simulation.Atmosphere(shape=(10, 10, 10), cell_size=2)
    photon = simulation.PhotonPacket()
    photon._theta = 0.9553166181245092  # ~54.7356 degrees
    photon._phi = np.pi / 4
    dir = simulation.direction_in_cartesian(photon._theta, photon._phi)
    photon._optical_length = atmosphere.distance_to_boundary(photon.position(), dir)

    lengths = atmosphere.deposit_luminosity(photon, return_lengths=True)
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
    Lx, Ly, Lz = np.array(atmosphere.shape()) * cell_size
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
    atmosphere = simulation.Atmosphere(cell_size=1e5, shape=(10,10,10))
    photon_blue = simulation.PhotonPacket(position=np.array([0,0,atmosphere.shape()[0] * atmosphere.cell_size()/3]), wavelength=400e-9)
    photon_blue._theta = np.pi / 2

    dir = simulation.direction_in_cartesian(photon_blue._theta, photon_blue._phi)
    photon_blue._optical_length = np.min([atmosphere.distance_to_boundary(photon_blue.position(), dir), photon_blue.maximum_optical_depth()/photon_blue._scattering_coefficient])
    atmosphere.deposit_luminosity(photon_blue)
    photon_blue.move()

    photon_red = simulation.PhotonPacket(position=np.array([0,0,atmosphere.shape()[0] * atmosphere.cell_size() * 2/3]), wavelength=700e-9)
    photon_red._theta = np.pi / 2
    dir = simulation.direction_in_cartesian(photon_red._theta, photon_red._phi)
    photon_red._optical_length = np.min([atmosphere.distance_to_boundary(photon_red.position(), dir), photon_red.maximum_optical_depth()/photon_red._scattering_coefficient])
    atmosphere.deposit_luminosity(photon_red)
    photon_red.move()

    source_function = atmosphere.source_function_integrated()
    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=np.min(source_function), vmax=np.max(source_function))
    facecolors = cm.viridis(norm(source_function))

    nx, ny, nz = source_function.shape
    cell_size = atmosphere.cell_size()

    x = np.arange(0, (nx + 1) * cell_size, cell_size)
    y = np.arange(0, (ny + 1) * cell_size, cell_size)
    z = np.arange(0, (nz + 1) * cell_size, cell_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(X*1e-3, Y*1e-3, Z*1e-3, source_function, facecolors=facecolors, edgecolor='k', alpha=0.4)


    ax.plot(photon_blue.trajectory()[:,0]*1e-3,
        photon_blue.trajectory()[:,1]*1e-3,
        photon_blue.trajectory()[:,2]*1e-3,
        color='b', linewidth=3, label='Ray Path'
    )
    ax.plot(photon_red.trajectory()[:,0]*1e-3,
        photon_red.trajectory()[:,1]*1e-3,
        photon_red.trajectory()[:,2]*1e-3,
        color='r', linewidth=3, label='Ray Path'
    )
    ax.set_xlabel('[km]')
    ax.set_ylabel('[km]')
    ax.set_zlabel('[km]')
    ax.set_xlim(0, atmosphere.shape()[0] * atmosphere.cell_size()*1e-3)
    ax.set_ylim(0, atmosphere.shape()[1] * atmosphere.cell_size()*1e-3)
    ax.set_zlim(0, atmosphere.shape()[2] * atmosphere.cell_size()*1e-3)
    cbar=plt.colorbar(cm.ScalarMappable( cmap='viridis'), ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(r'$S_\nu^{\rm norm}$', fontsize=16)
    plt.savefig('./figures/test_plot_sourceFunction.png')
    plt.show()

def test_total_luminosity():
    """Test that the total luminosity does not depend on the photon number"""
    total_deposited=[]
    for i in range(2):
        star = simulation.Star(model='Sun',direction=(np.pi/1.2, 0))
        atmosphere = simulation.Atmosphere(shape=(10, 10, 10), cell_size=1e4)
        sim1 = simulation.Simulation(atm=atmosphere, star=star, N=10000 * (i +1) )
        sim1.run()

        total_deposited.append(np.sum(atmosphere.source_function_integrated()))

    assert np.isclose(total_deposited[0], total_deposited[1], rtol=1e-2), f"Total deposited luminosity should be similar for different photon numbers, got {total_deposited[0]} and {total_deposited[1]}"


if __name__ == "__main__":
    test_length_ray()
    print("test_length_ray passed.")
    test_plot_atmosphere()
    print("test_plot_atmosphere passed.")
    test_plot_sourceFunction()
    print("test_plot_sourceFunction passed.")
    test_total_luminosity()
    print("test_total_luminosity passed.")

