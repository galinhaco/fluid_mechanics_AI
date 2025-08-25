# 💨 Hybrid Optimization System for Duct Layouts

This project implements an **intelligent hybrid optimizer** for duct design in an industrial warehouse, taking into account physical and engineering constraints such as obstacles, curve angles, and pressure losses.

The solution combines:
- **Genetic Algorithm (GA)**
- **Particle Swarm Optimization (PSO)**
- **Refined Local Search**

---

## 🚀 Features

- Automatic generation of multiple duct layouts.
- Compliance with **standard bend angles of 30°, 45°, and 90°** (with configurable tolerance).
- Obstacle avoidance (walls and boundaries).
- Calculation of:
  - Total duct length
  - Structural mass
  - Pressure losses (friction + bends)
  - Required fan power
  - Flow rate and dynamic pressure
- Selection of **diverse and optimized layouts**.
- **3D visualization with matplotlib**.
- Export of **Excel-style detailed tables**.

---

## 📂 Project Structure

- `fluid_mechanics.py` → Main code containing:
  - Optimization classes (`LayoutGenetico`, `Particula`, `OtimizadorHibridoIA`)
  - Layout evaluation functions
  - Plotting and table generation
  - Main execution entry point `main_otimizacao()`

---

## 🛠️ Technologies Used

- [Python 3.x](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://scipy.org/)

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/galinhaco/fluid_mechanics_AI.git
cd your-repo
pip install -r requirements.txt
