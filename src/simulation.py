import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

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
        

    def bb_shape(self, x):
    # Stable near x=0: use series x^2 when x is tiny
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        small = x < 1e-6
        out[small] = x[small]**2                     # limit as x->0
        out[~small] = x[~small]**3 / (np.exp(x[~small]) - 1.0)
        return out
    
    def sample_blackbody_E(self, T, N, x_max=20.0, y_max=1.5):
    # simple rejection sampling; vectorized in batches for speed
        k_B = 1.380649e-23  # J/K
        samples = []
        batch = max(1000, int(N/5))
        while len(samples) < N:
            x = np.random.uniform(0.0, x_max, size=batch)
            y = np.random.uniform(0.0, y_max, size=batch)
            f = self.bb_shape(x)
            keep = x[y < f]
            samples.append(keep)
        x_samples = np.concatenate(samples)[:N]
        E = x_samples * k_B * T
        
        return E
        
    def createPhotonPackets(self, initial, N):
        energy_samples = self.sample_blackbody_E(self.T, N)
        photons = [PhotonPacket(initial, energy_samples[i]) for i in range(N)]

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
                intensity = np.sum(self.atm.source_function * lengths) #to add the exp(-tau) attenuation factor depending on optical depth?

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

if __name__ == "__main__":
    boxsize = (40, 40, 40)
    T = 5800
    R = 700
    D = 1.5e11
    N = 100

    star = Star(T, R, D)
    atmosphere = Atmosphere(boxsize)
    sim = Simulation(atmosphere, star, N)
    sim.run()
    sim.plot()

    obs_pos = [boxsize[0]/2, boxsize[1]/2, boxsize[2]/2]
    observer = Observer( atmosphere, star, 
                         position=obs_pos, 
                         image_size=(200, 220), fov_deg=(30, 30), 
                         up=np.array([0.0, 1.0, 0.0]), forward=np.array([0.0, 0.0, 1.0]),
                         star_direction=np.array([0.0, 0.0, 1.0])
                       )

    image = observer.render(include_star=True)
    observer.show(image,include_star=True)





    