import os
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
    def __init__(self, shape = (10,10,10), cell_size=1.0, number_density=1.4e24):
        self._shape = shape
        #self._source_function = np.zeros(shape)
        self._albedo = 1.0 # only scattering
        self._number_density = number_density  # molecules per m^3 (approx at sea level)
        self._cell_size = cell_size 
        self._source_function = np.empty(self._shape, dtype=object)
        for idx in np.ndindex(self._shape):
            self._source_function[idx] = []  # liste vide par voxel

        self._source_function_integrated = None
        self._spectral_source_function = None

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
        
    def integrate_source_function(self):
        """Compute total luminosity per cell on demand."""
        total = np.zeros(self._shape, dtype=float)
        for idx in np.ndindex(self._shape):
            ev = self._source_function[idx]
            if ev:
                total[idx] = sum(e for (_, e) in ev)
        self._source_function_integrated = total
        return total
    
    def source_function_integrated(self):
        """Return integrated source function, computing if necessary."""
        if self._source_function_integrated is None:
            return self.integrate_source_function()
        return self._source_function_integrated

    def compute_source_function_spectral(self, wavelength_bins):
        """Compute spectral energy per cell on demand."""
        n_bins = len(wavelength_bins) - 1
        spectral = np.zeros(self._shape + (n_bins,), dtype=float)
        for idx in np.ndindex(self._shape):
            ev = self._source_function[idx]
            if ev:
                for (lam, e) in ev:
                    ib = np.searchsorted(wavelength_bins, lam) - 1
                    if ib < 0 or ib >= n_bins:
                        continue
                    #ib = 0 if ib < 0 else (n_bins-1 if ib >= n_bins else ib)
                    spectral[idx + (ib,)] += e
        self._spectral_source_function = spectral
        return spectral
    
    def spectral_source_function(self, wavelength_bins):
        """Return spectral source function, computing if necessary."""
        if self._spectral_source_function is None:
            return self.compute_source_function_spectral(wavelength_bins)
        return self._spectral_source_function

    def cell_size(self):
        return self._cell_size
    def source_function(self):
        return self._source_function
    def shape(self):
        return self._shape
    def number_density(self):
        return self._number_density

class PhotonPacket:

    def __init__(self, position = np.array([0,0,0]), luminosity = 1.0, wavelength = 550e-9, number_density = 1.4e24, initial_theta=None, initial_phi=None):
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
    def __init__(self, model= None,T=None, R=None, D=None, direction= (0, 0)):
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
        out[small] = x[small] ** 2
        out[~small] = x[~small] ** 3 / (np.exp(x[~small]) - 1.0)
        return out  # unnormalized, fine for rejection

    def sample_blackbody_x(self, N, x_max=20.0, y_max=1.6, lam_band_m=None):
        """
        Rejection- ample x = h*nu/(k_B*T) ~ energy PDF, from the blackbody *energy* PDF f(x) ~~ x^3/(exp(x)-1).
        Returns N samples of x with the wavelength band chosen.
        NOTE: we track the *number accepted*, not the number of batches.
        """
        
        kept = []
        total = 0
        batch = max(1000, N // 5)
    
        if lam_band_m is not None:
            lam_min, lam_max = lam_band_m
            # x = hc/(λ kT)
            x_low  = (h.value * c.value) / (k_B.value * self.T * lam_max)  # from λ_max
            x_high = (h.value * c.value) / (k_B.value * self.T * lam_min)  # from λ_min
            x_low  = max(0.0, x_low)
            x_high = min(x_max, x_high)
            if not (x_high > x_low):
                raise ValueError(f"Banded sampler: empty x-range. "
                                 f"Try increasing x_max or widening λ-band. "
                                 f"(x_low={x_low:.3g}, x_high={x_high:.3g})")
    
        while total < N:
            x = np.random.uniform(0.0, x_max, size=batch)
            y = np.random.uniform(0.0, y_max, size=batch)
            f = self.bb_shape_energy_pdf(x)
            acc = x[y < f]
            if lam_band_m is not None:
                acc = acc[(acc >= x_low) & (acc <= x_high)]
            take = min(N - total, acc.size)
            if take:
                kept.append(acc[:take]); total += take
        return np.concatenate(kept, axis=0)

    
    def planck_B_lambda(self, lam_m):
        """
        Planck's law for black body for then to be calculated into spectral irradiance
        """
        lam = np.asarray(lam_m, dtype=float)
        lam = np.clip(lam, 1e-20, None)
        a = 2.0 * h.value * c.value**2 / lam**5
        b = h.value * c.value / (lam * k_B.value * self.T)
        
        with np.errstate(over='ignore', under='ignore'):
            return a / np.expm1(b)

    
    def irradiance_F_lambda(self, lam_m):
        """
        Flux density from the star
        """
        return np.pi * (self.R / self.D)**2 * self.planck_B_lambda(lam_m)

    
    def createPhotonPackets(self, initial, N, use_physical_units=True, 
                            area=1.0, dt=1.0,
                            lam_band_m=(1e-9, 1_000_000.0e-9), 
                            number_density=1.4e24):            # 1 nm – 1 mm
        #lam_band_m = (lam_band_nm[0]*1e-9, lam_band_nm[1]*1e-9)
    
        x_samples = self.sample_blackbody_x(N, x_max=20.0, y_max=1.6, lam_band_m=lam_band_m)
        x_samples = np.clip(x_samples, 1e-12, None)
        lam_samples = ((h*c) / k_B).value / (self.T * x_samples)
    
        if use_physical_units:
            lam_grid = np.linspace(lam_band_m[0], lam_band_m[1], 5000)
            F = self.irradiance_F_lambda(lam_grid) # W/m²/m spectral flux 
            
            band_power_per_area = np.trapezoid(F, lam_grid) # W/m² integrated over band
            total_band_energy   = band_power_per_area * area * dt
            print(f"Total luminosity receuved from star in band (W):{total_band_energy:.3e}, on the surface area (m²): {area:.3e}")
            print(f"Total flux received from star in band {lam_band_m[0]*1e9:.1f}nm-{lam_band_m[1]*1e9:.1f}nm: {band_power_per_area:.1f} W/m²")
            weight = total_band_energy / N
        else:
            weight = 1.0
    
        photons = []
        for i in range(N):
            p = PhotonPacket(position=initial[i],
                             luminosity=weight,
                             wavelength=lam_samples[i],
                             initial_theta= np.pi - self._direction[0],
                             initial_phi= self._direction[1] + np.pi, 
                             number_density=number_density)
            photons.append(p)
        return photons

    def direction(self):
        """
        Return the star direction as (theta, phi) in radians.
        """
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
                 spectral_efficiency=np.array([0.5, 0.9, 0.7])):
        self.atm = atmosphere
        self.star = star
        self.position = np.array(position, dtype=float)
        self.nx, self.ny = image_size
        self.fov_x = np.deg2rad(fov_deg[0])
        self.fov_y = np.deg2rad(fov_deg[1])

        # camera orientation
        self.forward = forward / np.linalg.norm(forward)
        self.up = up / np.linalg.norm(up)
        self.right = np.cross(self.up, self.forward)
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
        """Compute the angular radius of the star""" 
        return np.arcsin(np.clip(self.star.R / self.star.D, 0.0, 1.0)) # before it was R*1e6, i changed it so it is more consistent we put the actual avalue in the main func

    # -----------------------------------------------------------
    def integrate_spectral_lum(self, initial_position, depth, direction, spectral_source_function, alpha):
        """
        Computes the spectral luminosity along a ray.
        initial_position: (x, y, z) coordinates of the starting point
        depth: distance to integrate along the ray
        direction: (dx, dy, dz) direction vector
        Returns: 3D array of spectral luminosity per bin (blue, green, red)
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
            t_next = min(tMaxX, tMaxY, tMaxZ) # distance to next boundary crossing (from initial position)
            delta = min(t_next, depth) - t_curr # length traveled in this cell
            length_in_cell[Xcell, Ycell, Zcell] += alpha * delta * spectral_source_function[Xcell, Ycell, Zcell] * np.exp(-alpha * (depth - t_curr))

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
    
    def render(self, projection='fisheye', radius_ratio=1.0, include_star=True, use_saved_data=True, file='./data/simulation_output.npz'):
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
        alpha = sigma * self.atm.number_density()  # extinction coefficient per bin
        
        if os.path.exists(file) and use_saved_data:
            data = np.load(file, allow_pickle=True)

            spectral_source_function = data['spectral_source_function']
        else:
            spectral_source_function = self.atm.spectral_source_function(edges)

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

                elif projection == 'fisheye':  # fisheye hemisphere equidistant
                    dx = (i - cx) / Rpix
                    dy = (j - cy) / Rpix
                    r = np.hypot(dx, dy)
                    if r > 1.0:
                        continue
                    theta = r * (np.pi/2.0)
                    phi = np.arctan2(-dy, dx)
                    dir_vec = np.cos(theta)*self.forward + np.sin(theta)*(np.cos(phi)*self.right + np.sin(phi)*self.up)
                    dir_vec /= np.linalg.norm(dir_vec)

                depth = self.atm.distance_to_boundary(self.position, dir_vec)
                if depth <= 0:
                    continue
                pixel_spectral = self.integrate_spectral_lum(self.position, depth, dir_vec, spectral_source_function, alpha)
                
                if include_star:
                    
                    # check if star is in this pixel
                    star_theta, star_phi = self.star_direction
                    star_dir = direction_in_cartesian(star_theta, star_phi)
                    star_dir /= np.linalg.norm(star_dir)
                    cos_angle = np.dot(dir_vec, star_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    #print("Stellar angular radius (rad):", self.star_angular_radius())
                    if angle < self.star_angular_radius()*2:
                        print(f"Pixel ({i}, {j}) includes star at angle {angle*180/np.pi} deg")
                        # angular size and solid angle of the star
                        alpha_star = self.star_angular_radius()
                        omega_star = 2.0 * np.pi * (1.0 - np.cos(alpha_star))

                        # spectral fractions per bin (R,G,B order as used)
                        star_spec = np.array([0.125, 0.136, 0.103])
                        L_b = self.star.luminosity() * star_spec  # luminosity per band (W)

                        # add band-by-band: F_b = L_b / (4πD^2), attenuated by atmosphere
                        for b in range(self._n_bins):
                            transmittance_star = np.exp(-alpha[b] * depth)
                            F_b = L_b[b] / (4.0 * np.pi * (self.star.D ** 2))
                            pixel_spectral[b] += F_b * transmittance_star #F_b * frac * transmittance_star
                
                img[j, i, :] = pixel_spectral

        return img
    
    def overlay_fisheye_grid(self, ax, nx, ny, rings_deg=(15, 30, 45, 60, 75)):
                cx = (nx - 1) / 2.0
                cy = (ny - 1) / 2.0
                R  = min(nx, ny) / 2.0

                # Anneaux de theta (θ = r * 90° -> r = θ/90)
                for deg in rings_deg:
                    r = (deg / 90.0) * R
                    circle = plt.Circle((cx, cy), r, color='w', fill=False, alpha=0.3, linewidth=1)
                    ax.add_patch(circle)
                    ax.text(cx + r + 4, cy, f'{deg}°', color='w', va='center', fontsize=8)

                    # Horizon (90°)
                    horizon = plt.Circle((cx, cy), R, color='w', fill=False, alpha=0.6, linewidth=1.5)
                    ax.add_patch(horizon)
                    ax.text(cx + R + 6, cy, '90° (horizon)', color='w', va='center', fontsize=8)

                    # Cardinaux (φ)
                    ax.text(cx + R*0.95, cy, 'E (φ=0)', color='w', ha='left', va='center', fontsize=8)
                    ax.text(cx - R*0.95, cy, 'W (φ=±π)', color='w', ha='right', va='center', fontsize=8)
                    ax.text(cx, cy - R*0.95, 'S (φ=-π/2)', color='w', ha='center', va='top', fontsize=8)
                    ax.text(cx, cy + R*0.95, 'N (φ=+π/2)', color='w', ha='center', va='bottom', fontsize=8)
    def show_spectrum(self, img, pixelx=None, pixely=None):
        """Bar chart of spectral intensity at one pixel in W·m⁻²·sr⁻¹·nm⁻¹."""
        # Choix du pixel (défaut: centre de l'image)
        if pixelx is None:
            pixelx = self.nx // 2
        if pixely is None:
            pixely = self.ny // 2

        # Attention: img est indexé [j, i, c] = [y, x, channel]
        spec_band = img[pixely, pixelx, :]  # ordre [B, G, R]

        # Bords des bandes en m → centres et largeurs
        edges_m = np.asarray(self.spectral_edges, dtype=float)            # shape (4,)
        widths_m = np.diff(edges_m)                                       # (3,)
        centers_nm = 0.5 * (edges_m[:-1] + edges_m[1:]) * 1e9
        widths_nm = widths_m * 1e9
        # Tracé
        plt.figure(figsize=(6, 4))
        colors_bgr = ['#4f81bd', '#9bbb59', '#c0504d']  # B, G, R
        plt.bar(centers_nm, spec_band, width=widths_nm, color=colors_bgr, alpha=0.9)

        # Marqueurs des bords de bande
        for e_nm in edges_m * 1e9:
            plt.axvline(e_nm, color='k', alpha=0.15, lw=1)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (W·m⁻²·sr⁻¹)')
        plt.title(f'Pixel spectrum at (x={pixelx}, y={pixely})')
        plt.tight_layout()
        plt.savefig(f'./figures/observer_spectrum_{pixelx}_{pixely}.png')
        plt.show()

    def show_truecolor(self, img, projection='pinhole'):
        """Display rendered image as truecolor using spectral sensitivities."""
        # normalize per channel with efficiency
        rgb = np.zeros_like(img)
        for c in range(3):
            rgb[:, :, c] = img[:, :, 2-c] * self.spectral_efficiency[c]
        # clip robuste par canal, puis normalisation
        
        m = np.percentile(rgb, 99.5)
        if m > 0:
            rgb = rgb / m

        plt.figure(figsize=(8, 8))
        extent = [-180, 180, 180, 0]
        if projection == 'fisheye':
            plt.imshow(np.clip(rgb, 0.0, 1.0)** (1.0 / 2.2), interpolation=None)
            ax = plt.gca()
            self.overlay_fisheye_grid(ax, self.nx, self.ny)
        else:
            plt.imshow(np.clip(rgb, 0.0, 1.0)** (1.0 / 2.2), extent=extent, interpolation=None)
            plt.xlabel('Azimuth φ (degrees)')
            plt.ylabel('Elevation θ (degrees)')
            plt.title('Observer Truecolor Image')
            
        plt.savefig('./figures/observer_output.png')
        plt.show()


class Simulation:
    def __init__(self, atm=None, star=None, obs=None, N=10):

        self.observer = obs
        self.atmosphere = atm
        self.star = star
        
        """ initial = np.array([np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N),
                            np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N),
                            (self.atmosphere.shape()[2] * self.atmosphere.cell_size() - 0.001)*np.ones(N)]).T """
        initial, area = self.initial_positions(N)
        self.N = initial.shape[0]
        print("Initial photon positions shape:", initial.shape)
        nx, ny, _ = self.atmosphere.shape()
        dx = self.atmosphere.cell_size()
        #area = (nx*dx) * (ny*dx)   # top surface area
        dt   = 1.0                 # s

        self.photons = self.star.createPhotonPackets(initial, self.N,
                                                     use_physical_units=True,
                                                     area=area, dt=dt,
                                                     lam_band_m=(self.observer.spectral_edges[0], self.observer.spectral_edges[-1]), 
                                                     number_density=self.atmosphere.number_density())
        print("Number of photon packets created:", len(self.photons))
    def initial_positions(self, N):
        dir = self.star.direction()
        theta_star, phi_star = dir
        faceR = (theta_star > 0) & (theta_star < np.pi) & (((phi_star > -np.pi/2) & (phi_star < 0))| (phi_star < np.pi/2))
        faceL = (theta_star > 0) & (theta_star < np.pi) & (((phi_star > np.pi/2) & (phi_star < 3*np.pi/2)) | ((phi_star < -np.pi/2) & (phi_star > -3*np.pi/2)))
        faceFront = (theta_star > 0 ) & (theta_star < np.pi) & (phi_star > 0) & (phi_star < np.pi)
        faceBack = (theta_star > 0 ) & (theta_star < np.pi) & (phi_star < 0) & (phi_star > -np.pi)
        faceTop = (theta_star < np.pi/2)
        faceBottom = False
        faces = [faceR, faceL, faceFront, faceBack, faceTop, faceBottom]
        n_faces = sum(faces)
        print("Number of illuminated faces:", n_faces, "faces:", faces)
        areas = [self.atmosphere.shape()[1] * self.atmosphere.cell_size() * self.atmosphere.shape()[2] * self.atmosphere.cell_size(),
                 self.atmosphere.shape()[1] * self.atmosphere.cell_size() * self.atmosphere.shape()[2] * self.atmosphere.cell_size(),
                 self.atmosphere.shape()[0] * self.atmosphere.cell_size() * self.atmosphere.shape()[2] * self.atmosphere.cell_size(),
                 self.atmosphere.shape()[0] * self.atmosphere.cell_size() * self.atmosphere.shape()[2] * self.atmosphere.cell_size(),
                 self.atmosphere.shape()[0] * self.atmosphere.cell_size() * self.atmosphere.shape()[1] * self.atmosphere.cell_size(),
                 self.atmosphere.shape()[0] * self.atmosphere.cell_size() * self.atmosphere.shape()[1] * self.atmosphere.cell_size()]
        total_area = sum([areas[i] for i in range(6) if faces[i]])
        initial = np.empty((0,3))

        
        for i, face in enumerate(faces):
            if face:
                if i == 0:  # Right face
                    N_face = round(N * (areas[0] / total_area))
                    x = np.full(N_face, self.atmosphere.shape()[0] * self.atmosphere.cell_size()- 0.001)
                    y = np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N_face)
                    z = np.random.uniform(0, self.atmosphere.shape()[2] * self.atmosphere.cell_size(), N_face)
                elif i == 1:  # Left face
                    N_face = round(N * (areas[1] / total_area))
                    x = np.zeros(N_face) + 0.001
                    y = np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N_face)
                    z = np.random.uniform(0, self.atmosphere.shape()[2] * self.atmosphere.cell_size(), N_face)
                elif i == 2:  # Front face
                    N_face = round(N * (areas[2] / total_area))
                    x = np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N_face)
                    y = np.full(N_face, self.atmosphere.shape()[1] * self.atmosphere.cell_size() - 0.001)
                    z = np.random.uniform(0, self.atmosphere.shape()[2] * self.atmosphere.cell_size(), N_face)
                elif i == 3:  # Back face
                    N_face = round(N * (areas[3] / total_area))
                    x = np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N_face)
                    y = np.zeros(N_face) + 0.001
                    z = np.random.uniform(0, self.atmosphere.shape()[2] * self.atmosphere.cell_size(), N_face)
                elif i == 4:  # Top face
                    N_face = round(N * (areas[4] / total_area))
                    x = np.random.uniform(0, self.atmosphere.shape()[0] * self.atmosphere.cell_size(), N_face)
                    y = np.random.uniform(0, self.atmosphere.shape()[1] * self.atmosphere.cell_size(), N_face)
                    z = np.full(N_face, self.atmosphere.shape()[2] * self.atmosphere.cell_size() - 0.001)
                initial =  np.vstack((initial, np.column_stack((x, y, z))))
        return initial, total_area

    def run(self, save_output=None):
        count_total = 0
        events_per_band = [0, 0, 0]      # [B, G, R]
        photons_per_band = [0, 0, 0]

        E_in = 0
        E_esc = 0
        E_dep = 0

        for photon in self.photons:
            E_in += photon.luminosity()
            # Classe la bande une seule fois
            bidx = np.searchsorted(self.observer.spectral_edges, photon.wavelength()) - 1
            if bidx < 0 or bidx > 2:
                bidx = None
            else:
                photons_per_band[bidx] += 1
            while photon.luminosity() > photon.luminosity_threshold() and self.atmosphere.in_box(photon.position()):
                L0 = photon.luminosity()
                photon.random_walk()
                tau, theta, phi = photon.get_random_walk()
                length = photon.optical_length()

                # Dépôt (la fonction tronque d'elle-même à la frontière si besoin)
                self.atmosphere.deposit_luminosity(photon)
                E_dep += L0 - photon.luminosity()
                # La fin de pas est-elle à l'intérieur ?
                end_inside = self.atmosphere.in_box(photon.position() + direction_in_cartesian(theta, phi) * length)
                if end_inside:
                    photon.move()                  # un scatter a bien lieu au bout du pas
                    count_total += 1
                    if bidx is not None:
                        events_per_band[bidx] += 1
                else:
                    # Dernier segment: pas de scatter, on ne compte pas
                    L = self.atmosphere.distance_to_boundary(photon.position(), direction_in_cartesian(theta, phi))
                    photon.set_optical_length(L)
                    photon.move()
                    break
            E_esc += photon.luminosity()
        print("Total interaction events per photon packets:", count_total / len(self.photons))
        print("Number of scatter events per bands: Blue:", events_per_band[0]/photons_per_band[0] if photons_per_band[0] > 0 else 0, "Green:", events_per_band[1]/photons_per_band[1] if photons_per_band[1] > 0 else 0, "Red:", events_per_band[2]/photons_per_band[2] if photons_per_band[2] > 0 else 0)
        print(f"Energy in: {E_in:.3e} Energy escaped: {E_esc:.3e} Energy deposited: {E_dep:.3e}")
        print(f"Relative energy check: (E_in - E_esc - E_dep)/E_in = {(E_in - E_esc - E_dep)/E_in:.3e}")


        if save_output is not None:
            np.savez(save_output, source_function = self.atmosphere.source_function(),spectral_source_function=self.atmosphere.spectral_source_function(self.observer.spectral_edges))

    def plot3D(self, band = None, rays=False, use_saved_data=True):
        if band is None:
            luminosity = self.atmosphere.source_function_integrated()
        else:
            if use_saved_data and os.path.exists('./data/simulation_horizon.npz'):
                data = np.load('./data/simulation_horizon.npz')
                spectral_source_func= data['spectral_source_function']
            else:
                spectral_source_func = self.atmosphere.spectral_source_function(self.observer.spectral_edges)

            if band == 'red':
                luminosity = spectral_source_func[:,:,:,2]
            elif band == 'green':
                luminosity = spectral_source_func[:,:,:,1]
            elif band == 'blue':
                luminosity = spectral_source_func[:,:,:,0]
            else:
                raise ValueError("Band must be one of 'red', 'green', 'blue', or None for total luminosity.")

        norm = colors.Normalize(vmin=np.min(luminosity), vmax=np.max(luminosity))
        facecolors = cm.rainbow_r(norm(luminosity))
        nx, ny, nz = self.atmosphere.shape()
        cell_size = self.atmosphere.cell_size()

        x = np.arange(0, (nx + 1) * cell_size, cell_size)
        y = np.arange(0, (ny + 1) * cell_size, cell_size)
        z = np.arange(0, (nz + 1) * cell_size, cell_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.voxels(X, Y, Z, luminosity > 0, facecolors=facecolors, alpha=0.2)
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
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax, shrink=0.5, aspect=5, label='Luminosity Deposited')
        if band is None:
            band = 'total'
        plt.title(f'{band} luminosity deposited')
        plt.savefig(f'./figures/simulation_output_{band}.png')
        plt.show()

    def plot_sourceFunction(self, mode='bands', use_saved_data=True, show_checks=True, n_bins=200):
        """
        Affiche la source fonction spectrale à partir des événements par voxel.
        Modes:
        - 'events' : nuage de points des événements bruts (W·m⁻²·sr⁻¹, pas de densité spectrale)
        - 'bands'  : histogramme par bandes R/G/B en unités W·m⁻²·sr⁻¹·nm⁻¹
        - 'hist'   : histogramme fin en λ (nm) en unités W·m⁻²·sr⁻¹·nm⁻¹
        - 'fine'   : courbe lissée en λ (nm) en unités W·m⁻²·sr⁻¹·nm⁻¹
        """
        # Récupération des événements bruts (lambda, valeur)
        if os.path.exists('./data/simulation_output.npz') and use_saved_data:
            data = np.load('./data/simulation_output.npz', allow_pickle=True)
            src = data['source_function']
        else:
            src = self.atmosphere.source_function()  # (nx, ny, nz) avec listes [(lam, val)]

        lam_list, val_list = [], []
        for idx in np.ndindex(src.shape):
            ev = src[idx]
            if not ev:
                continue
            for lam, val in ev:
                lam_list.append(lam)  # m
                val_list.append(val)  # W·m⁻²·sr⁻¹ par événement

        if len(lam_list) == 0:
            print("No source-function events to plot.")
            return

        lam = np.asarray(lam_list, dtype=float)
        vals = np.asarray(val_list, dtype=float)
        m = np.isfinite(lam) & np.isfinite(vals)
        lam, vals = lam[m], vals[m]

        plt.figure(figsize=(8, 5))

        # Mode événements: pas de densité spectrale → pas de nm⁻¹
        if mode == 'events':
            plt.scatter(lam * 1e9, vals, s=6, alpha=0.25, edgecolors='none')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Deposited intensity per event (W·m⁻²·sr⁻¹)')
            plt.title('Source-function events (no binning)')
            if self.observer.spectral_edges is not None:
                for e in np.asarray(self.observer.spectral_edges) * 1e9:
                    plt.axvline(e, color='k', alpha=0.1, lw=1)
            plt.tight_layout()
            plt.savefig('./figures/source_function_events.png')
            plt.show()
            return

        # Bandes: densité spectrale en nm⁻¹
        if mode == 'bands':
            edges = np.asarray(self.observer.spectral_edges, dtype=float)  # m
            totals, _ = np.histogram(lam, bins=edges, weights=vals)        # W·m⁻²·sr⁻¹ par bande
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths_m = np.diff(edges)
            spectral_vals_nm = (totals / widths_m) * 1e-9                  # W·m⁻²·sr⁻¹·nm⁻¹

            if show_checks:
                widths_nm = widths_m * 1e9
                sum_raw = vals.sum()
                sum_rebinned = (spectral_vals_nm * widths_nm).sum()
                rel_err = 0.0 if sum_raw == 0 else abs(sum_rebinned - sum_raw) / sum_raw
                print(f"[bands] energy check: raw={sum_raw:.3e}, rebinned={sum_rebinned:.3e}, rel_err={rel_err:.3e}")

            plt.bar(centers * 1e9, spectral_vals_nm, width=widths_m * 1e9,
                    align='center', color=['#4f81bd', '#9bbb59', '#c0504d'])
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Spectral intensity (W·m⁻²·sr⁻¹·nm⁻¹)')
            plt.title('Spectral source function (bands, nm⁻¹)')
            plt.tight_layout()
            plt.savefig('./figures/source_function_bands_nm.png')
            plt.show()
            return

        # Histogramme par bandes: valeurs par voxel (nm⁻¹), agrégées (moyenne) au lieu de la somme globale
        if mode == 'hist':
            edges = np.asarray(self.observer.spectral_edges, dtype=float)   # m, len=4
            widths_m = np.diff(edges)                                       # (3,) bande widths
            spectral = self.atmosphere.spectral_source_function(edges)      # (nx, ny, nz, 3), W·m⁻²·sr⁻¹ par bande et voxel

            # Convertir en W·m⁻²·sr⁻¹·nm⁻¹ par voxel et par bande
            S_nm = spectral / widths_m                                      # broadcast on last axis (3)
            S_nm = S_nm * 1e-9                                              # m⁻¹ → nm⁻¹

            # Aplatir tous les voxels
            flat = S_nm.reshape(-1, S_nm.shape[-1])                         # (Nvox, 3)

            # Statistiques robustes par bande
            mean_vals = np.nanmean(flat, axis=0)
            p10 = np.nanpercentile(flat, 10, axis=0)
            p90 = np.nanpercentile(flat, 90, axis=0)

            centers_nm = 0.5 * (edges[:-1] + edges[1:]) * 1e9
            widths_nm = widths_m * 1e9

            plt.bar(centers_nm, mean_vals, width=widths_nm,
                    color=['#4f81bd', '#9bbb59', '#c0504d'], alpha=0.85, label='Mean per voxel')

            # Barres d'erreur (10–90 percentile) pour visualiser la dispersion
            yerr = np.vstack((mean_vals - p10, p90 - mean_vals))
            plt.errorbar(centers_nm, mean_vals, yerr=yerr, fmt='none', ecolor='k', alpha=0.5, capsize=3)

            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Spectral intensity per voxel (W·m⁻²·sr⁻¹·nm⁻¹)')
            plt.title('Spectral source function (per-voxel mean, nm⁻¹)')
            plt.tight_layout()
            plt.savefig('./figures/source_function_hist_vox_nm.png')
            plt.show()
            return

        # Courbe fine en λ: nm⁻¹ (line plot)
        if mode == 'fine':
            lam_min = max(lam.min(), 300e-9)
            lam_max = min(lam.max(), 800e-9)
            edges = np.linspace(lam_min, lam_max, n_bins + 1)              # m
            totals, _ = np.histogram(lam, bins=edges, weights=vals)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths_m = np.diff(edges)
            spectral_vals_nm = (totals / widths_m) * 1e-9                  # W·m⁻²·sr⁻¹·nm⁻¹
            widths_nm = widths_m * 1e9

            if show_checks:
                sum_raw = vals.sum()
                sum_rebinned = (spectral_vals_nm * widths_nm).sum()
                rel_err = 0.0 if sum_raw == 0 else abs(sum_rebinned - sum_raw) / sum_raw
                print(f"[fine] energy check: raw={sum_raw:.3e}, rebinned={sum_rebinned:.3e}, rel_err={rel_err:.3e}")

            plt.plot(centers * 1e9, spectral_vals_nm, lw=1.8)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Spectral intensity (W·m⁻²·sr⁻¹·nm⁻¹)')
            plt.title('Spectral source function (fine, nm⁻¹)')
            if self.observer.spectral_edges is not None:
                for e in np.asarray(self.observer.spectral_edges) * 1e9:
                    plt.axvline(e, color='k', alpha=0.1, lw=1)
            plt.tight_layout()
            plt.savefig('./figures/source_function_fine_nm.png')
            plt.show()
            return

        raise ValueError("mode must be one of: 'events', 'bands', 'hist', 'fine'")
    
    def print_sourceFunction_stats(self, use_saved_data=True, file='./data/simulation_output.npz'):
        if os.path.exists(file) and use_saved_data:
            data = np.load(file, allow_pickle=True)
            src = data['source_function']
        else:
            src = self.atmosphere.source_function()  # (nx, ny, nz) avec listes [(lam, val)]

        cell_intensities = []
        cell_intensities_per_band = [[], [], []]  # B, G, R
        for idx in np.ndindex(src.shape):
            cell_I = 0.0
            cell_blue = 0
            cell_grn = 0
            cell_red = 0
            ev = src[idx]
            if not ev:
                continue
            for lam, val in ev:
                if 380e-9 <= lam < 495e-9:
                    cell_blue += val
                    
                elif 495e-9 <= lam < 570e-9:
                    cell_grn += val
                    
                elif 570e-9 <= lam < 700e-9:
                    cell_red += val
                    
                cell_I += val

            cell_intensities_per_band[0].append(cell_blue)
            cell_intensities_per_band[1].append(cell_grn)
            cell_intensities_per_band[2].append(cell_red)
            
            cell_intensities.append(cell_I)
        intensity_tot= np.sum(cell_intensities)
        luminosity_tot= intensity_tot * (self.atmosphere.cell_size()** 2 * self.atmosphere.shape()[0] * self.atmosphere.shape()[1]) * 4.0 * np.pi
        print(f"Total luminosity deposited in atmosphere: {luminosity_tot:.3e} W")

        print("Source-function statistics:")
        print(f"  Total deposited intensity: {np.sum(cell_intensities):.3e} W·m⁻²·sr⁻¹")
        print(f"Mean intensity per cell: {np.mean(cell_intensities):.3e} W·m⁻²·sr⁻¹")
        print(f" Median intensity per cell: {np.median(cell_intensities):.3e} W·m⁻²·sr⁻¹")
        print(f"  Intensity std. dev.: {np.std(cell_intensities):.3e} W·m⁻²·sr⁻¹")
        bands = ['Blue', 'Green', 'Red']
        for b in range(3):  
            band_vals = cell_intensities_per_band[b]
            if len(band_vals) == 0:
                continue
            print(f" {bands[b]} band:")
            print(f"  Total deposited intensity: {np.sum(band_vals):.3e} W·m⁻²·sr⁻¹")
            print(f" Mean intensity per cell: {np.mean(band_vals):.3e} W·m⁻²·sr⁻¹")
            print(f" Median intensity per cell: {np.median(band_vals):.3e} W·m⁻²·sr⁻¹")
            print(f"  Intensity std. dev.: {np.std(band_vals):.3e} W·m⁻²·sr⁻¹")

    def observe(self, projection='fisheye', use_saved_data=True, file='./data/simulation_output.npz'):

        img = self.observer.render(include_star=True, projection=projection, use_saved_data=use_saved_data, file=file)

        self.observer.show_truecolor(img, projection=projection)
        self.observer.show_spectrum(img, pixelx=image_size[0]-5, pixely=image_size[1]//2)
        self.observer.show_spectrum(img, pixelx=5, pixely=image_size[1]//2)


if __name__ == "__main__":
    boxsize = (100, 100, 15) # in number of cells
    cell_size = 1e4 # in meters: 10 km

    N = 1_000_000 # number of photon packets
    star = Star(model='Sun', direction=(np.pi/2, np.pi)) #direction: theta, phi
    atm = Atmosphere(shape = boxsize, cell_size=cell_size, number_density=1.3e24)  # number density in m^-3
    
    obs_pos = [(atm.shape()[0] * atm.cell_size())/2, 
                   (atm.shape()[1] * atm.cell_size())/2,
                   0]
    image_size = (480, 480)
    spectral_edges = [380e-9, 495e-9, 570e-9, 700e-9]  # visible spectrum in meters
    spectral_efficiency = [0.7, 1.0, 0.5]  # R, G, B sensitivities
    observer = Observer( atm, star, 
                        position=obs_pos, 
                        image_size=image_size, 
                        spectral_efficiency=spectral_efficiency, 
                        spectral_edges=spectral_edges)

    sim = Simulation(atm=atm, star=star, obs=observer, N=N)
    output_file = './data/simulation_horizon.npz'
    sim.run(save_output=output_file)
    #sim.plot3D(use_saved_data=False)
    #sim.plot_sourceFunction(use_saved_data=True, mode='bands')
    #sim.plot_sourceFunction(use_saved_data=True, mode='hist')
    #sim.print_sourceFunction_stats(use_saved_data=True, file = output_file)
    """ sim.plot(band='blue')
    sim.plot(band='green')
    sim.plot(band='red') """

    sim.observe(projection='fisheye', use_saved_data=False, file = output_file)

    





    
