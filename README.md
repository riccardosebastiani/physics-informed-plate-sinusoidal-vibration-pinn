# Physics-Informed Neural Network (PINN) for Vibrating Plate Modeling

This project implements a Physics-Informed Neural Network (PINN) to model the displacement field of an orthotropic square plate of dimensions 10x10 square meters with simply supported boundaries and sinusoidal loading.

## 📌 Project Overview

- Geometry: orthotropic square plate (10x10 m)
- Physics: Navier equations for plate vibration
- Boundary conditions: simply supported
- Method: PINN with SIREN activation (sinusoidal representation networks) and ReLoBRaLo (Re-Initialization and Loss Balancing of Residuals)
- Input: space-time coordinates (x, y, t)
- Output: displacement u(x, y, t)

## 🛠 Technologies

- Python
- PyTorch
- NumPy
- SciPy
- Matplotlib

## 📁 Files

- `main.py` – Main execution script
- `training.py` – Training loop and optimization
- `modules.py` – Neural network models
- `loss_functions.py` – PINN loss functions
- `Data_Set.py` – Dataset generator
- `WAlg.py` – Analytical solution module
- `dataio.py` – Utility functions for saving and loading

**Author**: Riccardo Sebastiani Croce
