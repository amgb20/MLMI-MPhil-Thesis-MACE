import numpy as np
import matplotlib.pyplot as plt

def plot_water_density(data_path, output_path):
    # Load the thermo data
    data = np.loadtxt(data_path)
    time_fs = data[:, 0]
    temp = data[:, 1]
    density = data[:, 2]
    pressure = data[:, 4]
    energy = data[:, 5]

    # Plot temperature evolution
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(time_fs, temp, 'b-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Time')
    plt.grid(True)

    # Plot density evolution
    plt.subplot(2, 2, 2)
    plt.plot(time_fs, density, 'r-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Density (g/cm³)')
    plt.title('Density vs Time')
    plt.grid(True)

    # Plot pressure evolution
    plt.subplot(2, 2, 3)
    plt.plot(time_fs, pressure, 'g-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Pressure (bar)')
    plt.title('Pressure vs Time')
    plt.grid(True)

    # Plot energy evolution
    plt.subplot(2, 2, 4)
    plt.plot(time_fs, energy, 'm-', linewidth=2)
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.title('Energy vs Time')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)


def main():

    data_path = "Experiments/numerical_stability/src/water-density/run_test_cueq_fp16_300.0.thermo"
    output_path = "Experiments/numerical_stability/src/water-density/water_density.png"
    plot_water_density(data_path, output_path)

if __name__ == "__main__":
    main()
