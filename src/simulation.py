import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from astropy.constants import c, h, k_B

class Atmosphere:
    def __init__(self, shape = (10,10,10), cell_size=1.0):
        self._shape = shape
        self._source_function = np.zeros(shape)
        self._albedo = 1.0 # only scattering
        self._cell_size = cell_size 

    def in_box(self, position, index=False):
        if index:
            return all(0 <= position[i] < self._shape[i] for i in range(3))
        else:
            return all(0 <= position[i] < self._shape[i] * self._cell_size for i in range(3))

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
                boundary = self._shape[i] * self._cell_size
                distance = (boundary - position[i]) / direction[i]
            elif direction[i] < 0:
                boundary = 0
                distance = (boundary - position[i]) / direction[i]
            else:
                distance = float('inf')  # No movement in this direction
            distances.append(distance)
        return min(distances)
    
    def distance_to_planes(self, position, direction):
        x, y, z = position

        # Pour chaque axe, calcule la prochaine face physique
        if direction[0] > 0:
            faceX = (np.floor(x / self._cell_size) + 1) * self._cell_size
        else:
            faceX = (np.floor(x / self._cell_size) * self._cell_size)
        if direction[1] > 0:
            faceY = (np.floor(y / self._cell_size) + 1) * self._cell_size
        else:
            faceY = (np.floor(y / self._cell_size)) * self._cell_size
        if direction[2] > 0:
            faceZ = (np.floor(z / self._cell_size) + 1) * self._cell_size
        else:
            faceZ = (np.floor(z / self._cell_size)) * self._cell_size

        tMaxX = (faceX - x) / direction[0] if direction[0] != 0 else float('inf')
        tMaxY = (faceY - y) / direction[1] if direction[1] != 0 else float('inf')
        tMaxZ = (faceZ - z) / direction[2] if direction[2] != 0 else float('inf')

        return tMaxX, tMaxY, tMaxZ

    def parametric_distance_in_cell(self, direction):
        """
        Calculate the parametric distances to the cell boundaries.
        direction: (dx, dy, dz) direction vector
        Returns: (tDeltaX, tDeltaY, tDeltaZ) parametric distances to the cell boundaries
        """

        tDeltaX = self._cell_size / abs(direction[0]) if direction[0] != 0 else float('inf')
        tDeltaY = self._cell_size / abs(direction[1]) if direction[1] != 0 else float('inf')
        tDeltaZ = self._cell_size / abs(direction[2]) if direction[2] != 0 else float('inf')

        return tDeltaX, tDeltaY, tDeltaZ

    def deposit_luminosity(self, photon, return_lengths=False):
        """
        Compute the length of the ray in each cell it traverses.
        position: (x, y, z) coordinates of the starting point
        direction: (dx, dy, dz) direction vector
        Returns: 3D array of lengths in each cell
        """
        length_in_cell = np.zeros(self._shape)

        initial_position = photon.position()
        depth, theta, phi = photon.get_random_walk()
        length = photon.optical_length()
        direction = photon.direction_in_cartesian(theta, phi)
        if not self.in_box(initial_position + direction * length):
            length = self.distance_to_boundary(initial_position, direction)
        
        # Determine step directions
        stepX = 1 if direction[0] > 0 else -1
        stepY = 1 if direction[1] > 0 else -1
        stepZ = 1 if direction[2] > 0 else -1

        # Initial distances to the next planes
        tMaxX, tMaxY, tMaxZ = self.distance_to_planes(initial_position, direction)
        t_curr = 0
        # Parametric distances to cross a cell
        tDeltaX, tDeltaY, tDeltaZ = self.parametric_distance_in_cell(direction)
        # Current cell indices
        Xcell, Ycell, Zcell = np.floor(initial_position/self._cell_size).astype(int)
        print(Xcell, Ycell, Zcell)
        while self.in_box([Xcell, Ycell, Zcell], index=True) and t_curr < length: 
            t_next = min(tMaxX, tMaxY, tMaxZ) # distance to next boundary crossing (from initial position)
            delta = min(t_next, length) - t_curr # length traveled in this cell 
            length_in_cell[Xcell, Ycell, Zcell] += delta
            self._source_function[Xcell, Ycell, Zcell] += delta * photon.luminosity() * self._albedo / (4 * np.pi * self._cell_size ** 3)
            photon.luminosity_loss(delta)

            if t_next == tMaxX:
                tMaxX += tDeltaX
                Xcell += stepX
            elif t_next == tMaxY:
                tMaxY += tDeltaY
                Ycell += stepY
            else:
                tMaxZ += tDeltaZ
                Zcell += stepZ
            #position = initial_position + direction * t_curr
            #Xcell, Ycell, Zcell = np.floor(position / self._cell_size).astype(int)
            print(Xcell, Ycell, Zcell)
            t_curr = min(t_next, length)

        if return_lengths:
            return length_in_cell
    
    def compute_length_in_cells(self, initial_position, depth, direction):
        """
        Compute the length of the ray in each cell it traverses.
        initial_position: (x, y, z) coordinates of the starting point
        depth: distance to propagate
        direction: (dx, dy, dz) direction vector
        Returns: 3D array of lengths in each cell
        """
        length_in_cell = np.zeros(self._shape)

        # Step directions
        stepX = 1 if direction[0] > 0 else -1
        stepY = 1 if direction[1] > 0 else -1
        stepZ = 1 if direction[2] > 0 else -1

        # Initial distances to the next planes
        tMaxX, tMaxY, tMaxZ = self.distance_to_planes(initial_position, direction)
        t_curr = 0
        # Parametric distances to cross a cell
        tDeltaX, tDeltaY, tDeltaZ = self.parametric_distance_in_cell(direction)
        # Current cell indices
        Xcell, Ycell, Zcell = np.floor(initial_position/self._cell_size).astype(int)

        while self.in_box([Xcell, Ycell, Zcell], index=True) and t_curr < depth: 
            t_next = min(tMaxX, tMaxY, tMaxZ)
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

    def cell_size(self):
        return self._cell_size
    def source_function(self):
        return self._source_function
    def shape(self):
        return self._shape

class PhotonPacket:

    def __init__(self, position = np.array([0,0,0]), luminosity = 1.0, wavelength = 550e-9, number_density = 2.5e25, initial_theta=None, initial_phi=None):
        self._position = position
        self._lambda = wavelength  # in meters
        self._luminosity = luminosity  # in Watts
        self._trajectory = np.array([position])
        self._cross_section = 4.3e-56 / (self._lambda **4)  # Rayleigh scattering cross-section
        self._scattering_coefficient = self._cross_section * number_density
        self._luminosity_threshold = 1e-30
        self._optical_depth = 0
        self._optical_length = self._optical_depth / self._scattering_coefficient
        self._theta = 0
        self._phi = 0
        
        self._first_step = True
        self._initial_theta = initial_theta
        self._initial_phi = initial_phi

    def direction_in_cartesian(self, theta, phi):
        dx = np.sin(theta) * np.cos(phi)
        dy = np.sin(theta) * np.sin(phi)
        dz = np.cos(theta)
        return np.array([dx, dy, dz])
    
    def maximum_optical_depth(self):
        return -np.log(self._luminosity_threshold / self._luminosity)
    
    def random_walk(self):
        random_optical_depth = -np.log(np.random.random())
        optical_depth = min(random_optical_depth, self.maximum_optical_depth())
        if self._first_step and self._initial_theta is not None and self._initial_phi is not None:
            theta = self._initial_theta
            phi = self._initial_phi
            self._first_step = False
        else:
            phi = 2 * np.pi * np.random.random()
            theta = np.arccos(2 * np.random.random() - 1)

        self._optical_depth = optical_depth
        self._optical_length = optical_depth / self._scattering_coefficient
        self._theta = theta
        self._phi = phi

    def move(self):

        direction = self.direction_in_cartesian(self._theta, self._phi)
        new_position = self._position + self._optical_length * direction
        self._position = new_position
        self._trajectory = np.append(self._trajectory, [new_position], axis=0)

    def set_optical_length(self, length):
        self._optical_length = length
        self._optical_depth = length * self._scattering_coefficient

    def luminosity_loss(self, s=None):
        if s is None:
            s = self._optical_length
        self._luminosity *= np.exp(-self._scattering_coefficient * s)

    def get_random_walk(self):

        return self._optical_depth, self._theta, self._phi
    
    def optical_length(self):

        return self._optical_length
    def optical_depth(self):
        return self._optical_depth
    
    def position(self):
        return self._position
    
    def trajectory(self):
        return self._trajectory
    
    def luminosity(self):
        return self._luminosity

    def luminosity_threshold(self):
        return self._luminosity_threshold

    def scattering_coefficient(self):
        return self._scattering_coefficient
    def wavelength(self):
        return self._lambda

class Star:
    def __init__(self, T, R, D, direction= (np.pi, 0)):
        self.T = T
        self.R = R
        self.D = D
        self._direction = direction

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

        x_samples = self.sample_blackbody_x(self.T, N)
        x_samples = np.clip(x_samples, 1e-9, None)
        lam_samples = ((h*c) / k_B).value / (self.T * x_samples)  # store if you need lambda-dependent opacities
    
        if use_physical_units:
            # You can later replace this with: weight = (∫F_lambda dλ * area * dt)/N
            weight = 1.0 / N
        else:
            weight = 1.0  # relative units: every packet identical
    
        photons = []
        for i in range(N):
            p = PhotonPacket(position = initial[i], 
                             luminosity=weight, 
                             wavelength=lam_samples[i], 
                             initial_theta=self._direction[0], 
                             initial_phi=self._direction[1])
            photons.append(p)
        return photons

class Observer:
    """
    Integrate the source function of the star along all lines of sight to render an image
    """

    def __init__(self, atmosphere, star, position,
                 image_size=(200, 200), fov_deg=(10.0,10.0),
                 up=np.array([0.0, 1.0, 0.0]),
                 forward=np.array([0.0, 0.0, 1.0]),
                 star_direction=np.array([0.0, 0.0, 1.0])):
        """
        atmosphere : Atmosphere object
        star       : Star object 
        position   : Array, 3D coordinates i nside the atmosphere
        image_size : image size in pixels
        fov_deg    : field of view in degrees
        up, forward: camera orientation vectors
        star_direction : unit vector pointing from observer to star
        
        """
        self.atm = atmosphere
        self.star = star
        self.position = np.array(position, dtype=float)
        self.nx, self.ny = image_size
        self.fov_x = np.deg2rad(fov_deg[0])
        self.fov_y = np.deg2rad(fov_deg[0])

        # camera orientation
        self.forward = forward / np.linalg.norm(forward) #+Z by default
        self.up = up / np.linalg.norm(up) #Y by default
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)

        self.star_direction = star_direction / np.linalg.norm(star_direction)

    # -----------------------------------------------------------
    def ray_direction(self, i, j):
        """Return 3D ray direction for pixel (i, j) taking into account the fov
        https://en.wikipedia.org/wiki/Field_of_view_in_video_games"""
        x = (2*(i + 0.5) / self.nx - 1) * np.tan(self.fov_x/2) #horizontal (right) and vertical (up) fov
        y = (2*(j + 0.5) / self.ny - 1) * np.tan(self.fov_y/2) #(x,y) = position of pixel from the observer
        
        
        dir_cam = self.forward + x*self.right + y*self.up #https://en.wikipedia.org/wiki/Pinhole_camera_model / https://hedivision.github.io/Pinhole.html
        
        return dir_cam / np.linalg.norm(dir_cam)

    # -----------------------------------------------------------
    def star_angular_radius(self):
        """Compute the angular radius of the star""" 
        return np.arcsin(np.clip(self.star.R*1e6 / self.star.D, 0.0, 1.0)) 

    # -----------------------------------------------------------
    def render(self, include_star=True):
        """
        Integrate the source function along each ray
        Returns a 2D numpy array
        """
        img = np.zeros((self.ny, self.nx))
        star_ang = self.star_angular_radius()

        for j in range(self.ny):
            for i in range(self.nx):
                direction = self.ray_direction(i, j)
                depth = self.atm.distance_to_boundary(self.position, direction)
                if depth <= 0:
                    continue

                # integrate emission along this ray
                lengths = self.atm.compute_length_in_cells(self.position, depth, direction)
                intensity = np.sum(self.atm.source_function() * lengths) #to add the exp(-tau) attenuation factor depending on optical depth?

                #add star  if looking toward the star
                if include_star:
                    cosang = np.dot(direction, self.star_direction)
                    if cosang > np.cos(star_ang): 
                        intensity += 1.0  # put 1 as a placeholder, supposed to the intensity of the star

                img[j, i] = intensity

        return img

    # -----------------------------------------------------------
    def show(self, image, cmap='inferno', include_star=True):
        norm = colors.Normalize(vmin=np.min(image), vmax=np.max(image))
        plt.figure(figsize=(6,6))
        plt.imshow(image, origin='lower', cmap=cmap, norm=norm)
        plt.colorbar(label='Integrated intensity')
        plt.xlabel('x pixel')
        plt.ylabel('y pixel')
        if include_star==True:
            plt.savefig("../figures/render_w_star.png")
        else:
            plt.savefig("../figures/render_wo_star.png")
        plt.show()

class Simulation:
    def __init__(self, atmosphere, star, N=10):
        self.atmosphere = atmosphere
        self.star = star
        self.N = N
        initial = np.array([np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N),
                            np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N),
                            (self.atmosphere.shape()[2] * self.atmosphere.cell_size() - 0.001)*np.ones(N)]).T
        self.photons = self.star.createPhotonPackets(initial, N)

        obs_pos = [(self.atmosphere.shape()[0] * self.atmosphere.cell_size())/2, 
                   (self.atmosphere.shape()[1] * self.atmosphere.cell_size())/2,
                    (self.atmosphere.shape()[2] * self.atmosphere.cell_size())/2]
        self.observer = Observer( self.atmosphere, self.star, 
                            position=obs_pos, 
                            image_size=(200, 220), fov_deg=(30, 30), 
                            up=np.array([0.0, 1.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]),
                            star_direction=np.array([0.0, 0.0, 1.0])
                        )

    def run(self):
        
        for photon in self.photons:
            while photon.luminosity() > photon.luminosity_threshold() and self.atmosphere.in_box(photon.position()):
                print("Check")
                photon.random_walk()
                tau, theta, phi = photon.get_random_walk()
                
                length = photon.optical_length()
                print(f"Optical depth: {tau}, length: {length}, Theta: {theta*180/np.pi}, Phi: {phi*180/np.pi}")
                self.atmosphere.deposit_luminosity(photon)
                if self.atmosphere.in_box(photon.position() + photon.direction_in_cartesian(theta, phi) * length):
                    photon.move()
                else:
                    L = self.atmosphere.distance_to_boundary(photon.position(), photon.direction_in_cartesian(theta, phi))
                    photon.set_optical_length(L)
                    photon.move()
                    break

    def plot(self, rays=False):

        norm = colors.Normalize(vmin=np.min(self.atmosphere.source_function()), vmax=np.max(self.atmosphere.source_function()))
        facecolors = cm.rainbow_r(norm(self.atmosphere.source_function()))
        nx, ny, nz = self.atmosphere.shape()
        cell_size = self.atmosphere.cell_size()

        x = np.arange(0, (nx + 1) * cell_size, cell_size)
        y = np.arange(0, (ny + 1) * cell_size, cell_size)
        z = np.arange(0, (nz + 1) * cell_size, cell_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.voxels(X, Y, Z, self.atmosphere.source_function() > 0, facecolors=facecolors, edgecolor='k', alpha=0.5)
        if rays:
            for i in range(self.N):
                ax.plot(
                    self.photons[i].trajectory()[:,0],
                    self.photons[i].trajectory()[:,1],
                    self.photons[i].trajectory()[:,2],
                    color='r', linewidth=3, label='Ray Path'
            )
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_xlim(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size())
        ax.set_ylim(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size())
        ax.set_zlim(0, self.atmosphere.shape()[2] * self.atmosphere.cell_size())
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='Intensity')
        plt.savefig('/home/localuser/Documents/MC_RAD/AtmosphericScattering/figures/simulation_output.png')
        plt.show()

    def observe(self):

        image = self.observer.render(include_star=True)
        self.observer.show(image,include_star=True)

if __name__ == "__main__":
    boxsize = (10, 10, 10)
    cell_size = 1.0e4
    T = 5800
    R = 700
    D = 1.5e11
    N = 1
    
    star = Star(T, R, D, direction=(np.pi, 0))
    atm = Atmosphere(shape = boxsize, cell_size=cell_size)
    sim = Simulation(atm, star, N)
    sim.run()
    sim.plot(rays=True)
    sim.observe()

    





    