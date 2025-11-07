import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import scipy.constants as constant

class Atmosphere:
    def __init__(self, shape):
        self.shape = shape
        self.source_function = np.zeros(shape)
        self.albedo = 1.0

    def in_box(self, position):
        return all(0 <= position[i] < self.shape[i] for i in range(3))
    
    def distance_to_boundary(self, position, direction):
        """
        Calculate the distance to the boundary of the 3D grid from a given position in a given direction.
        position: (x, y, z) coordinates of the starting point
        direction: (dx, dy, dz) direction vector
        Returns: distance to the boundary
        """
        distances = []
        for i in range(3):
            if direction[i] > 0:
                boundary = self.shape[i]
                distance = (boundary - position[i]) / direction[i]
            elif direction[i] < 0:
                boundary = 0
                distance = (boundary - position[i]) / direction[i]
            else:
                distance = float('inf')  # No movement in this direction
            distances.append(distance)
        return min(distances)
    
    def distance_to_planes(self, position, direction):
        """
        Calculate the parametric distances to the next planes in a 3D grid.
        position: (x, y, z) coordinates of the starting point
        direction: (dx, dy, dz) direction vector
        Returns: (tMaxX, tMaxY, tMaxZ) parametric distances to the next planes
        """
        
        x, y, z = position
        # Determine the next face coordinates based on direction
        faceX = np.floor(x) + 1 if direction[0] > 0 else np.floor(x)
        faceY = np.floor(y) + 1 if direction[1] > 0 else np.floor(y)
        faceZ = np.floor(z) + 1 if direction[2] > 0 else np.floor(z)

        tMaxX = (faceX - x) / direction[0] if direction[0] != 0 else float('inf')
        tMaxY = (faceY - y) / direction[1] if direction[1] != 0 else float('inf')
        tMaxZ = (faceZ - z) / direction[2] if direction[2] != 0 else float('inf')

        return tMaxX, tMaxY, tMaxZ

    def parametric_distance_in_cell(self, direction, cell_sizes=(1,1,1)):
        """
        Calculate the parametric distances to the cell boundaries.
        direction: (dx, dy, dz) direction vector
        cell_sizes: (sx, sy, sz) sizes of the cells in each dimension
        Returns: (tDeltaX, tDeltaY, tDeltaZ) parametric distances to the cell boundaries
        """

        tDeltaX = cell_sizes[0] / abs(direction[0]) if direction[0] != 0 else float('inf')
        tDeltaY = cell_sizes[1] / abs(direction[1]) if direction[1] != 0 else float('inf')
        tDeltaZ = cell_sizes[2] / abs(direction[2]) if direction[2] != 0 else float('inf')

        return tDeltaX, tDeltaY, tDeltaZ

    def compute_length_in_cells(self, initial_position, depth, direction):
        """
        Compute the length of the ray in each cell it traverses.
        position: (x, y, z) coordinates of the starting point
        direction: (dx, dy, dz) direction vector
        Returns: 3D array of lengths in each cell
        """
        length_in_cell = np.zeros(self.shape)
        if not self.in_box(initial_position):
            return length_in_cell
        # Determine step directions
        stepX = 1 if direction[0] > 0 else -1
        stepY = 1 if direction[1] > 0 else -1
        stepZ = 1 if direction[2] > 0 else -1

        # Initial distances to the next planes
        tMaxX, tMaxY, tMaxZ = self.distance_to_planes(initial_position, direction)
        t_curr = np.min([tMaxX, tMaxY, tMaxZ])
        length_in_cell[np.floor(initial_position).astype(int)[0], np.floor(initial_position).astype(int)[1], np.floor(initial_position).astype(int)[2]] += t_curr

        # Parametric distances to cross a cell
        tDeltaX, tDeltaY, tDeltaZ = self.parametric_distance_in_cell(direction)
        # Current cell indices
        Xcell, Ycell, Zcell = np.floor(initial_position).astype(int)

        while self.in_box([Xcell, Ycell, Zcell]) and t_curr < depth: 
            if tMaxX < tMaxY and tMaxX < tMaxZ:
                t_next = tMaxX
            elif tMaxY < tMaxX and tMaxY < tMaxZ:
                t_next = tMaxY
            else:
                t_next = tMaxZ

            delta = min(t_next, depth) - t_curr
            length_in_cell[Xcell, Ycell, Zcell] += delta

            if t_next == tMaxX:
                tMaxX += tDeltaX
                Xcell += stepX
            elif t_next == tMaxY:
                tMaxY += tDeltaY
                Ycell += stepY
            else:
                tMaxZ += tDeltaZ
                Zcell += stepZ

            t_curr = min(t_next, depth)
        
        return length_in_cell

    def deposit_energy(self, photon):

        initial_position = photon.position
        depth = photon.optical_depth
        direction = photon.direction_in_cartesian(photon.theta, photon.phi)
        energy = photon.energy
        if not self.in_box(initial_position + direction * depth):
            depth = self.distance_to_boundary(initial_position, direction)

        lengths = self.compute_length_in_cells(initial_position, depth, direction)

        self.source_function += lengths * energy * self.albedo / (4 * np.pi)

class PhotonPacket:

    def __init__(self, position, energy):
        self.position = position
        self.energy = energy
        self.trajectory = np.array([position])
        self.energy_threshold = 1e-30
        self.optical_depth = 0
        self.theta = 0
        self.phi = 0

    def direction_in_cartesian(self, theta, phi):
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        return np.array([dx, dy, dz])
    
    def maximum_optical_depth(self):
        return -np.log(self.energy_threshold / self.energy)
    
    
    def random_walk(self, return_all=False):
        random_optical_depth = -np.log(np.random.random())
        optical_depth = min(random_optical_depth, self.maximum_optical_depth())
        phi = 2 * np.pi * np.random.random()
        theta = np.arccos(2 * np.random.random() - 1)

        self.optical_depth = optical_depth 
        self.theta = theta 
        self.phi = phi

        if return_all:
            return optical_depth, theta, phi

    def move(self):

        direction = self.direction_in_cartesian(self.theta, self.phi)
        new_position = self.position + self.optical_depth * direction
        self.position = new_position
        self.trajectory = np.append(self.trajectory, [new_position], axis=0)

    def energy_loss(self):
        self.energy *= np.exp(-self.optical_depth)
        
    def get_random_walk(self):

        return self.optical_depth, self.theta, self.phi


class Star:
    def __init__(self, T, R, D):
        self.T = T
        self.R = R
        self.D = D
        

    def bb_shape_energy_pdf(self, x):
        """
        Energy PDF shape in x = h*nu/(kB*T):  f(x) ∝ x^3 / (exp(x) - 1)
        Stable near x=0 by using the series limit.
        """
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        small = x < 1e-6
        out[small]  = x[small]**2
        out[~small] = x[~small]**3 / (np.exp(x[~small]) - 1.0)
        return out  # unnormalized, fine for rejection
    
    def sample_blackbody_x(self, T, N, x_max=20.0, y_max=1.6):
        """
        Rejection sample x ~ energy PDF. Returns N samples of x.
        NOTE: we track the *number accepted*, not the number of batches.
        """
        kept = []
        total = 0
        batch = max(1000, N // 5)
        while total < N:
            x = np.random.uniform(0.0, x_max, size=batch)
            y = np.random.uniform(0.0, y_max, size=batch)
            f = self.bb_shape_energy_pdf(x)
            accept = x[y < f]
            if accept.size:
                take = min(N - total, accept.size)
                kept.append(accept[:take])
                total += take
        return np.concatenate(kept, axis=0)
        
    def createPhotonPackets(self, initial, N, use_physical_units=False, area=1.0, dt=1.0):
        """
        Option A: equal-energy packets.
        - We sample *color* (x, hence lambda) from the energy PDF.
        - We give every packet the same weight (energy).
        """
        h = constant.h
        c = constant.c
        kB = constant.k
        
        x_samples = self.sample_blackbody_x(self.T, N)
        x_samples = np.clip(x_samples, 1e-9, None)
        lam_samples = (h*c) / (kB*self.T * x_samples)  # store if you need lambda-dependent opacities
    
        if use_physical_units:
            # You can later replace this with: weight = (∫F_lambda dλ * area * dt)/N
            weight = 1.0 / N
        else:
            weight = 1.0  # relative units: every packet identical
    
        photons = []
        for i in range(N):
            p = PhotonPacket(initial, weight)
            # Optional: keep color for later physics
            p.lambda_m = float(lam_samples[i])
            photons.append(p)
        return photons

class Simulation:
    def __init__(self, atmosphere, star, N=10):
        
        self.atmosphere = atmosphere
        self.star = star
        self.N = N

        x, y, z = self.atmosphere.shape[0]/2, self.atmosphere.shape[1]/2, self.atmosphere.shape[2]-0.001
        initial = np.array([x, y, z])

        self.photons = self.star.createPhotonPackets(initial,N)

    def run(self):
        
        for photon in self.photons:
            while photon.energy > photon.energy_threshold and self.atmosphere.in_box(photon.position):

                photon.random_walk()
                self.atmosphere.deposit_energy(photon)
                photon.energy_loss()
                photon.move()

    def plot(self):

        norm = colors.Normalize(vmin=np.min(self.atmosphere.source_function), vmax=np.max(self.atmosphere.source_function))
        facecolors = cm.rainbow_r(norm(self.atmosphere.source_function))

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.voxels(self.atmosphere.source_function > 0, facecolors=facecolors, edgecolor='k', alpha=0.5)
        for i in range(self.N):
            ax.plot(
                self.photons[i].trajectory[:,0],
                self.photons[i].trajectory[:,1],
                self.photons[i].trajectory[:,2],
                color='r', linewidth=3, label='Ray Path'
        )
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, boxsize[0])
        ax.set_ylim(0, boxsize[1])
        ax.set_zlim(0, boxsize[2])
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='Intensity')
        plt.savefig('../figures/simulation_output.png')
        plt.show()

if __name__ == "__main__":
    boxsize = (10, 10, 10)
    T = 5800  # Temperature in Kelvin
    R = 696.340  # Radius in Mm
    D = 1.5e11  # Distance in meters
    N = 1  # Number of photon packets
    star = Star(T, R, D)
    atm = Atmosphere(boxsize)
    sim = Simulation(atm, star, N)
    sim.run()
    sim.plot()



    