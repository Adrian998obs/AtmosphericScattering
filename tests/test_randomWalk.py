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
    while atmosphere.in_box(photon.position):
        photon.random_walk()
        photon.move()
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(photon.trajectory[:,0],
        photon.trajectory[:,1],
        photon.trajectory[:,2],
        color='r', linewidth=2, label='Photon Path'
    )
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(0, atmosphere.shape[0])
    ax.set_ylim(0, atmosphere.shape[1])
    ax.set_zlim(0, atmosphere.shape[2])
    plt.savefig('../figures/test_randomWalk.png')
    plt.close()

if __name__ == "__main__":
    test_RW()
    print("Random Walk test completed successfully.")