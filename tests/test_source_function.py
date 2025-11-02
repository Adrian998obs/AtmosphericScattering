import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation


def test_length_ray():
    """Test the computation of lengths in atmosphere cells for a vertical ray."""
    # Create a simple atmosphere
    atmosphere = simulation.Atmosphere((10, 10, 10))
    # Create a photon packet
    photon = simulation.PhotonPacket(position=np.array([0,0,0]), energy=1.0)
    
    # Vertical ray
    photon.optical_depth = 100
    photon.theta = 0
    photon.phi = 0
    
    lengths = atmosphere.compute_length_in_cells(photon.position, photon.optical_depth, photon.direction_in_cartesian(photon.theta, photon.phi))
    expected_length = 10
    assert np.sum(lengths) == expected_length, "Sum of lengths in cells should equal the photon's optical depth"

    # Diagonal ray
    photon.position = np.array([0, 0, 0])
    photon.theta = np.pi / 4  # Horizontal ray
    photon.phi = np.pi / 4
    lengths = atmosphere.compute_length_in_cells(photon.position, photon.optical_depth, photon.direction_in_cartesian(photon.theta, photon.phi))
    expected_length = np.sqrt(2) * 10
    assert np.sum(lengths)== expected_length, "Sum of lengths in cells should equal the photon's optical depth for diagonal ray"


if __name__ == "__main__":
    test_length_ray()
    print("test_length_ray passed.")

