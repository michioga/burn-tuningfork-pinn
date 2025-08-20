
import pandas as pd
import numpy as np

# --- Constants from src/constants.rs ---
YOUNGS_MODULUS = 206e9  # Pa
DENSITY = 7850.0  # kg/m^3
POISSON_RATIO = 0.3
K_FACTOR = 3.5160
PI = np.pi

# --- Select the theory to diagnose ---
# Change this to 'Timoshenko' to test the other theory
BEAM_THEORY_CHOICE = 'Euler' 

def calculate_theoretical_frequency(prong_length, prong_diameter):
    """
    Calculates the theoretical frequency based on the selected beam theory,
    mimicking the logic in src/physics.rs.
    """
    prong_length_m = prong_length
    prong_diameter_m = prong_diameter

    # --- Calculations from src/physics.rs ---
    area = (prong_diameter_m**2) * (PI / 4.0)
    moment_of_inertia = (prong_diameter_m**4) * (PI / 64.0)

    # Euler-Bernoulli frequency
    stiffness_e = moment_of_inertia * YOUNGS_MODULUS
    density_mass = area * DENSITY
    sqrt_term_e = np.sqrt(stiffness_e / density_mass)
    length_term_e = prong_length_m**2
    euler_freq = (sqrt_term_e * K_FACTOR / (2.0 * PI)) / length_term_e

    if BEAM_THEORY_CHOICE == 'Euler':
        return euler_freq

    # Timoshenko correction
    shear_modulus = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    shear_coeff = (6.0 * (1.0 + POISSON_RATIO)) / (7.0 + 6.0 * POISSON_RATIO)
    
    numerator = moment_of_inertia * YOUNGS_MODULUS
    denominator = area * (prong_length_m**2) * (shear_coeff * shear_modulus)
    correction_factor = numerator / denominator
    
    # This is the suspicious formula from the original code
    timo_freq = euler_freq / np.sqrt(3.0 * correction_factor + 1.0)
    
    return timo_freq

def main():
    # Load the dataset
    try:
        df = pd.read_csv("data/summary_parameters.csv")
    except FileNotFoundError:
        print("Error: data/summary_parameters.csv not found.")
        return

    # Check for required columns
    required_cols = ['prong_L', 'prong_D', 'eigenfrequency_1']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain the columns: {required_cols}")
        return

    # Calculate theoretical frequencies and errors
    errors = []
    for index, row in df.iterrows():
        # The dimension values in the CSV seem to be in mm, let's convert to meters
        prong_l_m = row['prong_L'] / 1000.0
        prong_d_m = row['prong_D'] / 1000.0
        
        fem_freq = row['eigenfrequency_1']
        
        if fem_freq == 0:
            continue

        theoretical_freq = calculate_theoretical_frequency(prong_l_m, prong_d_m)
        
        error = np.abs(theoretical_freq - fem_freq) / fem_freq
        errors.append(error)

    # --- Report Results ---
    if not errors:
        print("No valid data found to calculate errors.")
        return
        
    average_error_percent = np.mean(errors) * 100
    max_error_percent = np.max(errors) * 100
    
    print("\n--- Physics Model Discrepancy Report ---")
    print(f"Average Error between Timoshenko model and FEM data: {average_error_percent:.2f}%")
    print(f"Maximum Error: {max_error_percent:.2f}%")
    print("------------------------------------------\n")

    if average_error_percent > 5:
        print("Diagnosis:")
        print(f"The physical model has a significant average deviation of >5% from the FEM data.")
        print("This level of error explains why setting ALPHA to 0.5 hurts accuracy.")
        print("The model is being pulled towards a physical law that is a poor fit for the high-fidelity data.\n")
    else:
        print("Diagnosis:")
        print("The average error of the physical model is low (<5%).")
        print("The issue might be related to the penalty terms in the loss function rather than the frequency calculation itself.\n")


if __name__ == "__main__":
    main()
