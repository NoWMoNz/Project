import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random

if random_source_position == True:
    source_position = np.array([random.randint(0, 9),random.randint(0, 9)])  # True source at (4 km, 5 km
else:
    source_position = np.array(source_position)

source_position = np.array(source_position)  # True source at (4 km, 5 km)
sensor_positions = np.array(sensor_positions)

# Parameters
array_size = 10  # Array area in km
grid_resolution = 200  # Finer grid resolution for better MUSIC estimation
wave_speed = 5  # Wave propagation speed in km/s
sampling_rate = 100  # Sampling rate in Hz
num_samples = 1000  # Number of time samples
noise_level = 0.05  # Reduced noise level to improve SNR

# Generate earthquake signal (synthetic signal)
earthquake_signal = np.sin(2 * np.pi * np.linspace(0, 1, 100))

# Check the actual number of sensors
num_sensors = sensor_positions.shape[0]  # Dynamically set based on available sensors

# Simulate received signals with noise and delays
received_signals = np.zeros((num_sensors, num_samples))
for i in range(num_sensors):
    distance = np.linalg.norm(sensor_positions[i] - source_position)
    delay_samples = int(distance / wave_speed * sampling_rate)
    received_signals[i, delay_samples:delay_samples + len(earthquake_signal)] = earthquake_signal
    received_signals[i] += noise_level * np.random.randn(num_samples)  # Add noise

# Perform SVD to calculate the noise subspace
U, S, Vt = np.linalg.svd(received_signals)
num_signals = 1  # Number of signals (this could vary based on your setup)
noise_subspace = U[:, num_signals:]  # Retain all columns after the first signal

# Check for empty noise subspace
if noise_subspace.shape[1] == 0:
    raise ValueError("The noise subspace is empty. Adjust the number of signals or input data.")

# Define grid for MUSIC spectrum calculation
x_grid = np.linspace(0, array_size, grid_resolution)
y_grid = np.linspace(0, array_size, grid_resolution)
grid_x, grid_y = np.meshgrid(x_grid, y_grid)

# MUSIC Spectrum Calculation
spectrum_2d = np.zeros_like(grid_x)
for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        test_position = np.array([grid_x[i, j], grid_y[i, j]])
        distances = np.linalg.norm(sensor_positions - test_position, axis=1)
        steering_vector = np.exp(-1j * 2 * np.pi * distances / wave_speed)

        # Calculate MUSIC spectrum
        if noise_subspace.shape[1] > 0:  # Ensure there's noise subspace to work with
            spectrum_2d[i, j] = 1 / np.abs(steering_vector.conj().T @ noise_subspace @ noise_subspace.conj().T @ steering_vector)

# Convert spectrum to dB scale
spectrum_dB = 10 * np.log10(spectrum_2d)

# Find peaks in the MUSIC spectrum to estimate source location
spectrum_flat = spectrum_dB.flatten()
peaks, _ = find_peaks(spectrum_flat, height=-5)  # Adjusted threshold for more sensitivity
peak_coords = np.unravel_index(peaks, spectrum_2d.shape)

# Debugging: Print peaks found
peak_coords = np.unravel_index(peaks, spectrum_2d.shape)

# Print estimated coordinates for all found peaks
print("Peaks found at the following coordinates:")
for index in range(len(peaks)):
    x_coord = grid_x[peak_coords[0][index], peak_coords[1][index]]
    y_coord = grid_y[peak_coords[0][index], peak_coords[1][index]]
    print(f"Peak {index + 1}: Estimated Source Location at X = {x_coord:.2f} km, Y = {y_coord:.2f} km")

# Extract the highest peak (best guess for source location)
if peaks.size > 0:
    max_peak_index = np.argmax(spectrum_flat[peaks])
    # Get the coordinates for the maximum peak
    max_peak_coords = (grid_x[peak_coords[0][max_peak_index], peak_coords[1][max_peak_index]],
                       grid_y[peak_coords[0][max_peak_index], peak_coords[1][max_peak_index]])
    print(f"Highest Peak: Estimated Source Location: X = {max_peak_coords[0]:.2f} km, Y = {max_peak_coords[1]:.2f} km")
else:
    max_peak_coords = (None, None)  # Handle case where no peaks are found

# Plot MUSIC spectrum and highlight the estimated and true source positions
plt.imshow(spectrum_dB, extent=[0, array_size, 0, array_size], origin='lower')
plt.colorbar(label='MUSIC Spectrum (dB)')

# Plot the estimated source if found
if max_peak_coords[0] is not None:
    plt.scatter(max_peak_coords[0], max_peak_coords[1], c='blue', label='Estimated Source', marker='x')

# Plot the true source
plt.scatter(source_position[0], source_position[1], c='red', label='True Source')

# Plot all sensors
plt.scatter(sensor_positions[:, 0], sensor_positions[:, 1], c='white', label='Sensors', marker='o')

plt.xlabel('X (km)')
plt.ylabel('Y (km)')
plt.title('MUSIC Spectrum with Estimated Source and Sensor Locations')
plt.legend()
plt.show()

