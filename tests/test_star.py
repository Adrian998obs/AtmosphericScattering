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
    star = simulation.Star(model='Sun')

    lambda_samples = star.lambda_sample(10000)  # en mètres

    # grilles et spectre théorique (en m)
    l = np.linspace(1e-9, 3e-6, 1000)  # m
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
    lambda_samples_nm = lambda_samples * 1e9
    bins = np.linspace(lambda_samples_nm.min(), lambda_samples_nm.max(), 50)
    counts, edges = np.histogram(lambda_samples_nm, bins=bins, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # plot
    plt.figure(figsize=(7,4))
    plt.plot(l*1e9, B_energy_pdf_per_nm, label='BB energy PDF (per nm)', color='k')
    plt.plot(l*1e9, B_photon_pdf_per_nm, label='Photon-count PDF (per nm)', color='gray', linestyle='--')
    plt.plot(bin_centers, counts, drawstyle='steps-mid', label='Sampled histogram', color='C0')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('PDF (per nm)')
    plt.legend()
    plt.title('Compare sampled wavelengths with energy / photon PDFs')
    plt.tight_layout()
    plt.savefig('./figures/test_star_spectrum.png')
    plt.show()

if __name__ == "__main__":
    test_star_luminosity()
    test_star_spectrum()
    print("Star luminosity and spectrum tests completed successfully.")