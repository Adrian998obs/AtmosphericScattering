import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.constants import h, c, k_B
h=h.value
c=c.value
k_B=k_B.value
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import simulation

def test_star_luminosity():
    """Test the luminosity calculation of the Star class."""
    star = simulation.Star(model='Sun')
    luminosity = star.luminosity()
    expected_luminosity = 3.828e26  # Expected luminosity of the Sun in Watts
    assert np.isclose(luminosity, expected_luminosity, rtol=0.01), f"Calculated luminosity {luminosity} does not match expected {expected_luminosity}"

def test_star_spectrum():
    """Test the spectral energy distribution of the Star class."""
    T = 10000 # Temperature of the Sun in Kelvin
    R = 6.96e8  # Radius of the Sun in meters
    D = 1.496e11  # Distance from Sun to Earth in meters
    star = simulation.Star(T=T, R=R, D=D)

    N = 10000
    lam_band_nm = (380, 700)  # Visible range in nm
    x_samples = star.sample_blackbody_x(N, x_max=20.0, y_max=1.6, lam_band_m=(lam_band_nm[0]*1e-9, lam_band_nm[1]*1e-9))
    x_samples = np.clip(x_samples, 1e-12, None)
    lam_samples = ((h*c) / k_B) / (star.T * x_samples)
    N_blue = len(lam_samples[(lam_samples < 495e-9)& (lam_samples >= 380e-9)])
    N_green = len(lam_samples[(lam_samples < 570e-9)& (lam_samples >= 495e-9)])
    N_red = len(lam_samples[(lam_samples < 700e-9)& (lam_samples >= 570e-9)])
    print(f"Created {N} photon packets: {N_blue} blue, {N_green} green, {N_red} red.")
    # grilles et spectre théorique (en m)
    l = np.linspace(lam_band_nm[0]*1e-9, lam_band_nm[1]*1e-9,  1000)  # m
    B_lambda = (2*h*c**2) / (l**5) * 1.0 / (np.exp((h*c) / (l*k_B*star.T)) - 1.0)

    # normalisations (PDF par m)
    B_energy_pdf_m = B_lambda / np.trapezoid(B_lambda, l)
    # PDF en nombre de photons par m : B_lambda / (hc/λ) = B_lambda * λ / (hc)
    B_photon = B_lambda * l / (h*c)
    B_photon_pdf_m = B_photon / np.trapezoid(B_photon, l)

    # convertir les PDFs "par m" -> "par nm" (x-axis will be in nm)
    B_energy_pdf_per_nm = B_energy_pdf_m * 1e-9
    B_photon_pdf_per_nm = B_photon_pdf_m * 1e-9

    # histogramme des échantillons en nm
    lambda_samples_nm = lam_samples * 1e9
    bins = np.linspace(lambda_samples_nm.min(), lambda_samples_nm.max(), 100)
    counts, edges = np.histogram(lambda_samples_nm, bins=bins, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # plot
    plt.figure(figsize=(7,4))
    plt.plot(l*1e9, B_energy_pdf_per_nm, label='BB energy PDF (per nm)', color='k')
    plt.plot(l*1e9, B_photon_pdf_per_nm, label='Photon-count PDF (per nm)', color='gray', linestyle='--')
    plt.plot(bin_centers, counts, drawstyle='steps-mid', label='Sampled histogram', color='C0')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('PDF (per nm)')
    plt.xlim(0, 700)
    plt.legend()
    plt.title('Compare sampled wavelengths with energy / photon PDFs')
    plt.tight_layout()
    plt.savefig('./figures/test_star_spectrum.png')
    plt.show()

def test_total_luminosity():
    star = simulation.Star(model='Sun')
    lam_grid = np.linspace(380e-9, 700e-9, 5000)
    F = star.irradiance_F_lambda(lam_grid) # W/m²/m spectral flux 
    band_power_per_area = np.trapezoid(F, lam_grid) # W/m² integrated over band

    N_photons = 1_000_000
    photons = star.createPhotonPackets([0,0,0]*N_photons,N_photons, lam_band_m=(380e-9,700e-9), use_physical_units=True, area=1.0, dt=1.0)

    total_flux_received = np.sum([photon.luminosity() for photon in photons])
    assert np.isclose(total_flux_received, band_power_per_area, rtol=1e-2), f"Total photon luminosity {total_flux_received} does not match star luminosity {band_power_per_area}"
    
if __name__ == "__main__":
    test_star_luminosity()
    test_star_spectrum()
    test_total_luminosity()
    print("Star luminosity and spectrum tests completed successfully.")