import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
import os
from preisach import PreisachModel, analyticalPreisachFunction2, preisachIntegration

# -------------------------------
# Main Execution
# -------------------------------
# Ensure output directory exists
os.makedirs('Plots', exist_ok=True)

if __name__ == "__main__":
    model = PreisachModel(n=200, alpha0=1.0)
    gridX, gridY = model.gridX, model.gridY
    width = model.width

    # Use analytical Preisach function 2
    A = 1.0
    Hc = 0.01
    sigma = 0.03
    preisach = analyticalPreisachFunction2(A, Hc, sigma, gridX, gridY)

    # Compute Everett function
    everett = preisachIntegration(width, preisach)
    everett /= np.max(everett)  # Normalize

    # Interpolate
    points = np.column_stack((gridX.ravel(), gridY.ravel()))
    values = everett.ravel()
    everettInterp = LinearNDInterpolator(points, values, fill_value=0.0)
    model.setEverettFunction(everettInterp)

    # Show Everett function
    fig = plt.figure(figsize=(8, 6))
    model.showEverettFunction(fig)

    # Invert model
    invModel = model.invert()
    fig = plt.figure(figsize=(8, 6))
    invModel.showEverettFunction(fig)

    # Generate input signal
    nSamps = 2500
    phi = np.linspace(0, 2 * np.pi + np.pi / 2, nSamps)

    sawtooth = np.zeros(nSamps)
    sawtooth[phi < np.pi / 2] = 0.7 * 2 / np.pi * phi[phi < np.pi / 2]
    sawtooth[(phi >= np.pi / 2) & (phi < 3 * np.pi / 2)] = -0.7 * 2 / np.pi * (phi[(phi >= np.pi / 2) & (phi < 3 * np.pi / 2)] - np.pi)
    sawtooth[phi >= 3 * np.pi / 2] = 0.7 * 2 / np.pi * (phi[phi >= 3 * np.pi / 2] - 2 * np.pi)

    input_signal = 0.15 * np.sin(30 * phi) + sawtooth

    # Initialize models in demagnetized state
    model.setDemagState(80)
    invModel.setDemagState(80)

    # Simulate cascade: input → model → invModel
    middle = np.zeros_like(input_signal)
    output = np.zeros_like(input_signal)
    for i, u in enumerate(input_signal):
        middle[i] = model(u)
        output[i] = invModel(middle[i])

    # Plot signals
    plt.figure(figsize=(10, 6))
    plt.plot(input_signal, label='Input')
    plt.plot(middle, label='Middle (Forward Model Output)')
    plt.plot(output, label='Output (Inverse Model Output)')
    plt.legend()
    plt.title('Forward + Inverse Preisach Model')
    plt.xlabel('Sample')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Animate both models
    sim1 = model.animateHysteresis()
    sim2 = invModel.animateHysteresis()

    # Optional: Save animations
    sim1.save('Plots/preisach_simulation.mp4', fps=30, dpi=150)
    sim2.save('Plots/hysteresis_inverted_simulation.mp4', fps=30, dpi=150)

