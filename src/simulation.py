import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from astropy.constants import c, h, k_B

def direction_in_cartesian(theta, phi):
    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)
    return np.array([dx, dy, dz])


class Atmosphere:
    def __init__(self, shape = (10,10,10), cell_size=1.0):
        self._shape = shape
        #self._source_function = np.zeros(shape)
        self._albedo = 1.0 # only scattering
        self._cell_size = cell_size 
        self._source_function = np.empty(self._shape, dtype=object)
        for idx in np.ndindex(self._shape):
            self._source_function[idx] = []  # liste vide par voxel

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
        direction = direction_in_cartesian(theta, phi)
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

        while self.in_box([Xcell, Ycell, Zcell], index=True) and t_curr < length: 
            t_next = min(tMaxX, tMaxY, tMaxZ) # distance to next boundary crossing (from initial position)
            delta = min(t_next, length) - t_curr # length traveled in this cell 
            length_in_cell[Xcell, Ycell, Zcell] += delta
            deposited = delta * photon.luminosity() * self._albedo / (4 * np.pi * self._cell_size ** 3)
            #self._source_function[Xcell, Ycell, Zcell] += deposited
            self._source_function[Xcell, Ycell, Zcell].append((photon.wavelength(), deposited))

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

            t_curr = min(t_next, length)

        if return_lengths:
            return length_in_cell
        
    def source_function_integrated(self):
        """Compute integrated (over λ) energy per cell on demand."""
        integrated = np.zeros(self._shape, dtype=float)
        for idx in np.ndindex(self._shape):
            ev = self._source_function[idx]
            if ev:
                integrated[idx] = sum(e for (_, e) in ev)
        return integrated
    
    def source_function_spectral(self, wavelength_bins):
        """Compute spectral energy per cell on demand."""
        n_bins = len(wavelength_bins) - 1
        spectral = np.zeros(self._shape + (n_bins,), dtype=float)
        for idx in np.ndindex(self._shape):
            ev = self._source_function[idx]
            if ev:
                for (lam, e) in ev:
                    ib = np.searchsorted(wavelength_bins, lam) - 1
                    ib = 0 if ib < 0 else (n_bins-1 if ib >= n_bins else ib)
                    spectral[idx + (ib,)] += e
        return spectral
    

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

        direction = direction_in_cartesian(self._theta, self._phi)
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
    def __init__(self, model= 'Sun',T=None, R=None, D=None, direction= (0, 0)):
        if model == 'Sun':
            self.T = 5778  # Kelvin
            self.R = 6.96e8  # meters
            self.D = 1.5e11  # meters
        else:
            if T is None or R is None or D is None:
                raise ValueError("For custom star model, T, R, and D must be provided.")
            self.T = T
            self.R = R
            self.D = D

        self._luminosity = 4 * np.pi * (self.R)**2 * 5.67e-8 * self.T**4  # Stefan-Boltzmann law, R in meters
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
    
    def lambda_sample(self, N):
        x_samples = self.sample_blackbody_x(self.T, N)
        lam_samples = ((h*c) / k_B).value / (self.T * x_samples)  # store if you need lambda-dependent opacities
        return lam_samples
    
    def createPhotonPackets(self, initial, N, use_physical_units=True, area=1.0, dt=1.0):
        """
        Option A: equal-energy packets.
        - We sample *color* (x, hence lambda) from the energy PDF.
        - We give every packet the same weight (energy).
        """

        lam_samples = self.lambda_sample(N)

        if use_physical_units:
            # flux at distance D (W/m^2)
            flux = self._luminosity / (4.0 * np.pi * (self.D ** 2))
            # power intercepted by the target area A (W)
            intercepted_power = flux * area
            # assign power per packet (W)
            weight = intercepted_power / float(N)
        else:
            weight = 1.0  # relative units: every packet identical
    
        photons = []
        for i in range(N):
            p = PhotonPacket(position = initial[i], 
                             luminosity=weight, 
                             wavelength=lam_samples[i], 
                             initial_theta= np.pi - self._direction[0], 
                             initial_phi= self._direction[1] + np.pi)
            photons.append(p)
        return photons
    
    def direction(self):
        return self._direction
    
    def luminosity(self):
        return self._luminosity

class Observer:
    """
    Integrate the source function of the star along all lines of sight to render an image.
    Simple spectral model: observer has 3 spectral bins (R,G,B) with configurable sensitivities.
    """
    def __init__(self, atmosphere, star, position,
                 image_size=(200, 200), fov_deg=(10.0,10.0),
                 up=np.array([0.0, 1.0, 0.0]),
                 forward=np.array([0.0, 0.0, 1.0]),
                 # spectral settings: bin edges in meters and per-bin efficiency
                 spectral_edges=None,
                 spectral_efficiency=None):
        self.atm = atmosphere
        self.star = star
        self.position = np.array(position, dtype=float)
        self.nx, self.ny = image_size
        self.fov_x = np.deg2rad(fov_deg[0])
        self.fov_y = np.deg2rad(fov_deg[1])

        # camera orientation
        self.forward = forward / np.linalg.norm(forward)
        self.up = up / np.linalg.norm(up)
        self.right = np.cross(self.forward, self.up)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.up /= np.linalg.norm(self.up)

        theta, phi = star.direction()
        self.star_direction = np.array([theta, phi])
        print(f"Star direction (theta, phi): {self.star_direction * 180/np.pi} degrees")

        # spectral bins: default visible-ish bins (meters)
        if spectral_edges is None:
            # edges: [blue_start, green_start, red_start, red_end] in meters
            self.spectral_edges = np.array([380e-9, 495e-9, 570e-9, 700e-9])
        else:
            self.spectral_edges = np.asarray(spectral_edges, dtype=float)

        # per-bin efficiency (sensitivity) for R,G,B order (len = 3)
        if spectral_efficiency is None:
            # default simple eye-like sensitives (relative)
            self.spectral_efficiency = np.array([0.6, 1.0, 0.9])
        else:
            self.spectral_efficiency = np.asarray(spectral_efficiency, dtype=float)

    # -----------------------------------------------------------
    def ray_direction(self, i, j):
        """Return 3D ray direction for pixel (i, j) taking into account the fov"""
        x = (2*(i + 0.5) / self.nx - 1) * np.tan(self.fov_x/2)
        y = (2*(j + 0.5) / self.ny - 1) * np.tan(self.fov_y/2)
        dir_cam = self.forward + x*self.right + y*self.up
        return dir_cam / np.linalg.norm(dir_cam)
    
    def pixels_to_angles(self, i, j, coord='cartesian'):
        """
        Very simple: map each pixel to a direction (theta, phi) uniformly over the sphere:
        - theta in (0, pi) (colatitude)
        - phi in (-pi, pi)
        Returns (theta_map, phi_map) with shape (ny, nx).
        """
        theta = np.pi * (j + 0.5) / self.ny        # colatitude 0..pi
        phi = 2.0 * np.pi * (i + 0.5) / self.nx - np.pi  # azimuth -pi..pi
        if coord == 'spherical':
            return theta, phi
        dir_vec = direction_in_cartesian(theta, phi)
        return dir_vec
    # -----------------------------------------------------------
    def star_angular_radius(self):
        return np.arctan(np.clip(self.star.R/ self.star.D, 0.0, 1.0))

    # -----------------------------------------------------------
    def compute_length_in_cells(self, initial_position, depth, direction, source_function_spectral, alpha):
        """
        Compute the length of the ray in each cell it traverses.
        initial_position: (x, y, z) coordinates of the starting point
        depth: distance to propagate
        direction: (dx, dy, dz) direction vector
        Returns: 3D array of lengths in each cell
        """
        length_in_cell = np.zeros((self.atm.shape())+ (self._n_bins,), dtype=float)

        # Step directions
        stepX = 1 if direction[0] > 0 else -1
        stepY = 1 if direction[1] > 0 else -1
        stepZ = 1 if direction[2] > 0 else -1

        # Initial distances to the next planes
        tMaxX, tMaxY, tMaxZ = self.atm.distance_to_planes(initial_position, direction)
        t_curr = 0
        # Parametric distances to cross a cell
        tDeltaX, tDeltaY, tDeltaZ = self.atm.parametric_distance_in_cell(direction)
        # Current cell indices
        Xcell, Ycell, Zcell = np.floor(initial_position/self.atm.cell_size()).astype(int)

        while self.atm.in_box([Xcell, Ycell, Zcell], index=True) and t_curr < depth: 
            t_next = min(tMaxX, tMaxY, tMaxZ)
            delta = min(t_next, depth) - t_curr
            length_in_cell[Xcell, Ycell, Zcell] += alpha * delta * source_function_spectral[Xcell, Ycell, Zcell] * np.exp(-alpha * (depth - t_curr))

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

        return np.sum(length_in_cell, axis = (0,1,2))
    
    def render(self, projection='fisheye', radius_ratio=1.0, include_star=True, rayleigh_n=2.5e25):
        """
        Render sky with discrete RT along rays.
        Returns numpy array (ny, nx, 3) with channels in order [B, G, R].
        projection: 'fisheye' (hemisphere) or 'equirect' (full sphere)
        rayleigh_n: number density used for simple Rayleigh extinction σ(λ)*n
        """
        img = np.zeros((self.ny, self.nx, 3), dtype=float)
        edges = self.spectral_edges  # edges length = 4 -> 3 bins
        self._n_bins = len(edges) - 1
        cx = (self.nx - 1) / 2.0
        cy = (self.ny - 1) / 2.0
        Rpix = min(self.nx, self.ny) / 2.0 * radius_ratio
        cell_size = self.atm.cell_size()
        cell_vol = cell_size**3
        # simple Rayleigh σ(λ) ~ 4.3e-56 / λ^4 (m^2) used if no atmosphere model per-cell
        sigma_prefactor = 4.3e-56
        lam_center = [0.5 * (edges[b] + edges[b+1]) for b in range(self._n_bins)]
        sigma = sigma_prefactor / (np.array(lam_center) ** 4)
        alpha = sigma * rayleigh_n 

        source_function_spectral = self.atm.source_function_spectral(edges)

        # loop pixels
        for j in range(self.ny):
            for i in range(self.nx):
                # build direction for this pixel
                if projection == 'equirect':
                    # map pixel to (theta, phi)
                    theta = np.pi * (j + 0.5) / self.ny       # 0..pi
                    phi = 2*np.pi * (i + 0.5) / self.nx - np.pi
                    dir_vec = np.cos(theta)*self.forward + np.sin(theta)*(np.cos(phi)*self.right + np.sin(phi)*self.up)
                    dir_vec /= np.linalg.norm(dir_vec)
                
                elif projection == 'pinhole':
                    dir_vec = self.pixels_to_angles(i, j, coord='cartesian')

                else:  # fisheye hemisphere equidistant
                    dx = (i - cx) / Rpix
                    dy = (j - cy) / Rpix
                    r = np.hypot(dx, dy)
                    if r > 1.0:
                        continue
                    theta = r * (np.pi/2.0)
                    phi = np.arctan2(dy, dx)
                    dir_vec = np.cos(theta)*self.forward + np.sin(theta)*(np.cos(phi)*self.right + np.sin(phi)*self.up)
                    dir_vec /= np.linalg.norm(dir_vec)

                depth = self.atm.distance_to_boundary(self.position, dir_vec)
                if depth <= 0:
                    continue
                pixel_spectral = self.compute_length_in_cells(self.position, depth, dir_vec, source_function_spectral, alpha)
                
                if include_star:
                    
                    # check if star is in this pixel
                    star_theta, star_phi = self.star_direction
                    star_dir = direction_in_cartesian(star_theta, star_phi)
                    star_dir /= np.linalg.norm(star_dir)
                    cos_angle = np.dot(dir_vec, star_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    #print("Stellar angular radius (rad):", self.star_angular_radius())
                    if angle < self.star_angular_radius():
                        print(f"Pixel ({i}, {j}) includes star at angle {angle*180/np.pi} deg")
                        # angular size and solid angle of the star
                        alpha_star = self.star_angular_radius()
                        omega_star = 2.0 * np.pi * (1.0 - np.cos(alpha_star))

                        # spectral fractions per bin (R,G,B order as used)
                        star_spec = np.array([0.125, 0.136, 0.103])
                        L_b = self.star.luminosity() * star_spec  # luminosity per band (W)

                        # pixel solid angle (approx) using camera FOV
                        dtheta = self.fov_y / self.ny
                        dphi = self.fov_x / self.nx
                        theta_pix = np.arccos(np.clip(np.dot(dir_vec, self.forward), -1.0, 1.0))
                        delta_omega = np.sin(theta_pix) * dtheta * dphi

                        # CORRECTION: fraction of the STAR's solid angle that the pixel covers
                        # If pixel is smaller than the star: frac = ΔΩ_pixel / Ω_star
                        # If pixel is larger: frac = 1 (pixel contains whole star disk)
                        if omega_star <= 0 or delta_omega <= 0:
                            frac = 0.0
                        else:
                            frac = min(1.0, delta_omega / omega_star)

                        # add band-by-band: F_b = L_b / (4πD^2), attenuated by atmosphere
                        for b in range(self._n_bins):
                            transmittance_star = np.exp(-alpha[b] * depth)
                            F_b = L_b[b] / (4.0 * np.pi * (self.star.D ** 2))
                            pixel_spectral[b] += F_b * frac * transmittance_star
                
                img[j, i, :] = pixel_spectral

        return img
    
    def show_truecolor(self, img):
        """Display rendered image as truecolor using spectral sensitivities."""
        # normalize per channel with efficiency
        rgb = np.zeros_like(img)
        for c in range(3):
            rgb[:, :, c] = img[:, :, 2 - c] * self.spectral_efficiency[c]
        # normalize to max
        max_val = np.max(rgb)
        if max_val > 0:
            rgb /= max_val
        plt.figure(figsize=(8, 8))
        extent = [360,0, 180, 0]
        plt.imshow(rgb, origin='upper', extent=extent)
        #plt.axis('off')
        plt.savefig('./figures/observer_output.png')
        plt.show()


class Simulation:
    def __init__(self, atmosphere, star, N=10):
        self.atmosphere = atmosphere
        self.star = star
        self.N = N
        initial = np.array([np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N),
                            np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N),
                            (self.atmosphere.shape()[2] * self.atmosphere.cell_size() - 0.001)*np.ones(N)]).T
        area = (self.atmosphere.shape()[0] * self.atmosphere.cell_size()) * (self.atmosphere.shape()[1] * self.atmosphere.cell_size())
        self.photons = self.star.createPhotonPackets(initial, N, use_physical_units=True, area=area)

        obs_pos = [(self.atmosphere.shape()[0] * self.atmosphere.cell_size())/2, 
                   (self.atmosphere.shape()[1] * self.atmosphere.cell_size())/2,
                    (self.atmosphere.shape()[2] * self.atmosphere.cell_size())/2]
        self.observer = Observer( self.atmosphere, self.star, 
                            position=obs_pos, 
                            image_size=(500, 500), fov_deg=(30, 30), 
                            up=np.array([0.0, 1.0, 0.0]), forward=np.array([0.0, 0.0, 1.0])
                        )

    def run(self):
        
        for photon in self.photons:
            while photon.luminosity() > photon.luminosity_threshold() and self.atmosphere.in_box(photon.position()):
                photon.random_walk()
                tau, theta, phi = photon.get_random_walk()
                
                length = photon.optical_length()
                self.atmosphere.deposit_luminosity(photon)
                if self.atmosphere.in_box(photon.position() + direction_in_cartesian(theta, phi) * length):
                    photon.move()
                else:
                    L = self.atmosphere.distance_to_boundary(photon.position(), direction_in_cartesian(theta, phi))
                    photon.set_optical_length(L)
                    photon.move()
                    break

    def plot(self, rays=False):

        norm = colors.Normalize(vmin=np.min(self.atmosphere.source_function_integrated()), vmax=np.max(self.atmosphere.source_function_integrated()))
        facecolors = cm.rainbow_r(norm(self.atmosphere.source_function_integrated()))
        nx, ny, nz = self.atmosphere.shape()
        cell_size = self.atmosphere.cell_size()

        x = np.arange(0, (nx + 1) * cell_size, cell_size)
        y = np.arange(0, (ny + 1) * cell_size, cell_size)
        z = np.arange(0, (nz + 1) * cell_size, cell_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.voxels(X, Y, Z, self.atmosphere.source_function_integrated() > 0, facecolors=facecolors, edgecolor='k', alpha=0.5)
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
        plt.savefig('./figures/simulation_output.png')
        plt.show()

    def observe(self):

        img = self.observer.render(include_star=True, projection='pinhole')

        self.observer.show_truecolor(img)


if __name__ == "__main__":
    boxsize = (10, 10, 10)
    cell_size = 5e3

    N = 10000
    star = Star(model='Sun', direction=(np.pi/10, np.pi/2))
    atm = Atmosphere(shape = boxsize, cell_size=cell_size)
    sim = Simulation(atm, star, N)
    sim.run()
    sim.plot(rays=False)
    sim.observe()

    





    