[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![v1.0.0-alpha](https://img.shields.io/badge/version-1.0.0--alpha-yellow.svg)](https://github.com/ilidio-costa/L-PBF-Processing-Maps-Predictive-Analytical-Modelling/releases)

# Predictive Modeling Of L-PBF Printability Maps


<div align="center">
  <img src="images/Logo_LPBF.png" alt="L-PBF Processing Maps Logo" width="200">
</div align="left">


## Scientific Overview

The optimization of Laser Powder Bed Fusion (L-PBF) processing parameters is frequently constrained by the extensive experimental overhead required to identify stable melting regimes. This framework provides a predictive analytical solution to estimate melt pool morphology and evaluate manufacturing stability. By coupling classical thermal models with established defect criteria, the tool delineates "Safe Zones" from regions prone to balling, lack of fusion (LoF), and keyhole-induced porosity.

## Quick Start

### 1. Installation
Clone the repository and install the core dependencies:
```bash
git clone https://github.com/ilidio-costa/L-PBF-Processing-Maps-Predictive-Analytical-Modelling.git
pip install numpy scipy matplotlib
```
## Theoretical Framework

### Thermal Modeling Strategy

To capture the complex heat transfer regimes in L-PBF, the engine employs a hybrid modeling approach:
- **Conduction Mode (Eagar-Tsai Model):** Utilizes a traveling Gaussian-distributed heat source on a semi-infinite substrate to predict surface dimensions ($W$ and $L$) .
- **Deep Penetration Mode (Gladush-Smurov Model):** Accounts for keyhole physics by treating the laser as a cylindrical heat source, providing accurate depth ($D$) predictions in high-intensity regimes .
- **Dimensionless Scaling (Rubenchik Model):** Normalizes spatial coordinates and thermal variables to determine melt pool dimensions via universal functions ($g$) and normalized enthalpy ($B$) .


### Defect Evaluation Criteria

The printability of a specific parameter set is defined by the simultaneous avoidance of multiple competing defect regimes. Each analytical criterion is encapsulated within an isolated Python module in the `src/defects/` directory and adheres to a standardized API. This modularity allows the framework to evaluate each physical instability independently while maintaining a unified hierarchical logic for final visualization.

The table below details the current defect modules, their underlying analytical models, and the mathematical conditions they evaluate:

### Algorithmic Implementation of Defect Criteria
| Defect Type | Analytical Model | Python Module | Defect Condition (Returns `True`) |
| :--- | :--- | :--- | :--- |
| **Balling** | Plateau-Rayleigh | `ball01.py` | [cite_start]$L/W \ge 2.3$ [cite: 873] |
| **Balling** | Yadroitsev Stability | `ball02.py` | [cite_start]$\frac{\pi W}{L} \le \sqrt{\frac{2}{3}}$ [cite: 873] |
| **Lack of Fusion** | Depth-to-Layer Ratio | `lof01.py` | $D \le t$  |
| **Lack of Fusion** | Geometric Overlap | `lof02.py` | $(\frac{h}{W})^2 + \frac{t}{t+D} \ge 1.0$  |
| **Lack of Fusion** | Depth-to-Layer Ratio | `lof03.py` | $D/t > 1.5$  |
| **Keyholing** | King Keyhole Criteria | `key01.py` | $A \cdot P > \pi \rho C_p T_m \sqrt{\alpha v a^3} \cdot \frac{\pi T_b}{T_m}$  |
| **Keyholing** | Geometric Ratio | `key02.py` | $W/D < 2.0$  |
| **Keyholing** | Geometric Ratio | `key03.py` | $W/D < 1.5$  |
| **Keyholing** | Geometric Ratio | `key04.py` | $W/D < 2.75$  |
| **Keyholing** | Gan Keyhole Criteria | `key05.py` | $Ke = \frac{A \cdot P}{(T_m - T_0) \pi \rho C_p \sqrt{\alpha v a^3}} > 6.0$ |

### Modular API and Priority Mapping
Each defect module uses a standardized function signature: `def check(dimensions, process_parameters, material) -> bool`. This enables the core physics engine to:
* **Perform Isolated Evaluations**: Individual boundaries can be verified independently to confirm specific mathematical thresholds without interference from overlapping regimes.
* **Route Defect Priority**: In regions where multiple conditions are met, a priority-routing algorithm determines the dominant mechanism for visualization in the printability map.
* **Integrate Uncertainty**: Results can be normalized into model probabilities using a softmax function, identifying regions where predictive uncertainty is high (e.g., when models yield conflicting results).


## Repository Architecture & API Reference

[cite_start]The project follows a modular architecture ensuring a strict separation of concerns between physics, data management, and visualization.

### Physics Engine (`src/physics.py`)
* **`compute_printability_map`**: Master function that evaluates a $P-v$ grid and applies defect priority mapping.
* **`calculate_melt_pool_dimensions`**: Hybrid wrapper resolving surface dimensions via Rubenchik/Eagar-Tsai and depth via the maximum envelope of Gladush-Smurov and Eagar-Tsai.
* **`get_rubenchik_dimensions`**: Performs bounded numerical optimization to extract exact $L$, $W$, and $D$ from the dimensionless temperature field.

### Visualization Suite (`src/plots.py`)
* **`plot_deterministic_map`**: Renders pre-computed defect matrices using Gaussian-smoothed contours.

<div align="center">
  <img src="images/printability_map.png" alt="printability_map" width="400">
</div align="left">

* **`plot_safe_zone_evolution`**: A multidimensional engine for "Third Axis" evaluation (e.g., sensitivity to $T_{ambient}$, spot size, or layer thickness).

<div align="center">
  <img src="images/printability_map_z_axis.png" alt="printability_map_z_axis" width="400">
</div align="left">

* **`plot_all_zones_evolution`**: A multidimensional engine for "Third Axis" evaluation (e.g., sensitivity to $T_{ambient}$, spot size, or layer thickness).

<div align="center">
  <img src="images/printability_map_z_axis_all.png" alt="printability_map_z_axis" width="400">
</div align="left">

* **`top_view_eagar_tsai` / `side_view_rubenchik`**: Generates high-fidelity cross-sectional thermal profiles of the melt pool.

<div align="center">
  <img src="images/eagar_tsai.png" alt="melt pool" width="400">
</div align="left">

### Material Management (`src/data_loader.py`)
* **`load_material`**: Parses JSON files to calculate derived properties like thermal diffusivity ($\alpha = k / (\rho C_p)$).
* **`calculate_dynamic_absorptivity`**: Estimates laser coupling using the Hagen-Rubens approximation.

### Material JSON Structure
To ensure physical accuracy, thermophysical properties (Density, Specific Heat, and Thermal Conductivity) should be extracted at or near the material's melting point. Material files in the `materials/` directory follow this structure:

```json
{
    "name": "NiTi",
    "rho": 6100,
    "C_p": 510,
    "k": 4.4,
    "T_b": 3033,
    "T_m": 1583,
    "A": 0.32,
    "alpha": 8e-6,
    "electrical_resistivity": 8.2e-7
}
```

| JSON Key | Description | Symbol | Units |
| :--- | :--- | :---: | :--- |
| `rho` | Density | $\rho$ | $kg/m^{3}$ |
| `C_p` | Specific Heat Capacity | $C_{p}$ | $J/(kg \cdot K)$  |
| `k` | Thermal Conductivity | $k$ | $W/(m \cdot K)$  |
| `T_b` | Boiling Temperature | $T_{b}$ | $K$  |
| `T_m` | Melting Temperature | $T_{m}$ | $K$  |
| `A` | Laser Absorptivity | $A$ | Dimensionless  |
| `alpha` | Thermal Diffusivity | $\alpha$ | $m^{2}/s$ |
| `electrical_resistivity` | Electrical Resistivity | $\rho_{e}$ | $\Omega \cdot m$  |

## References

- B. Zhang et al., "An efficient framework for printability assessment in laser powder bed fusion metal additive manufacturing," *Additive Manufacturing*, vol. 46, p. 102018, 2021.

- S. Sheikh et al., "High-throughput alloy and process design for metal additive manufacturing," *npj Computational Materials*, vol. 11, no. 1, 2025.

- T. W. Eagar and N.-s. Tsai, "Temperature fields produced by traveling distributed heat sources," *Welding Journal*, vol. 62, no. 12, pp. 346s-355s, 1983.

- J.-N. Zhu et al., "Predictive analytical modelling and experimental validation of processing maps in additive manufacturing of nitinol alloys," *Additive Manufacturing*, vol. 38, p. 101802, 2021.

- S. Sheikh et al., "An automated computational framework to construct printability maps for additively manufactured metal alloys," *npj Computational Materials*, vol. 10, no. 1, 2024.

- Thermo-Calc Software AB, "Thermo-Calc Software Additive Manufacturing Module: Methodology for calculation of phase diagrams (CALPHAD) and steady-state melt pool simulations," Solna, Sweden, 2026.

- L. Johnson et al., "Assessing printability maps in additive manufacturing of metal alloys," *SSRN Electronic Journal*, 2019.

- N. Wu, B. Whalen, J. Ma, and P. V. Balachandran, "Probabilistic printability maps for laser powder bed fusion via functional calibration and uncertainty propagation," *Journal of Computing and Information Science in Engineering*, vol. 24, no. 11, 2024.

- X. Huang et al., "Hybrid microstructure-defect printability map in laser powder bed fusion additive manufacturing," *Computational Materials Science*, vol. 209, p. 111401, 2022.

- I. Yadroitsev, A. Gusarov, I. Yadroitsava, and I. Smurov, "Single track formation in selective laser melting of metal powders," *Journal of Materials Processing Technology*, vol. 210, no. 12, pp. 1624-1631, 2010.

- R. Seede et al., "An ultra-high strength martensitic steel fabricated using selective laser melting additive manufacturing: Densification, microstructure, and mechanical properties," *Acta Materialia*, vol. 186, pp. 199-214, 2020.

- W. E. King et al., "Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing," *Journal of Materials Processing Technology*, vol. 214, no. 12, pp. 2915-2925, 2014.

- Z. Gan et al., "Universal scaling laws of keyhole stability and porosity in 3D printing of metals," *Nature Communications*, vol. 12, no. 1, 2021.

- S. Sinha and T. Mukherjee, "Mitigation of gas porosity in additive manufacturing using experimental data analysis and mechanistic modeling," *Materials*, vol. 17, no. 7, p. 1569, 2024.

- A. M. Rubenchik, W. E. King, and S. S. Wu, "Scaling laws for the additive manufacturing," *Journal of Materials Processing Technology*, vol. 257, pp. 234-243, 2018.

- D. Schuöcker, *Handbook of the Eurolaser Academy*, Springer International Publishing.


## License
Distributed under the **GPLv3 License**. See `LICENSE` for details.

## Author
**Ilídio Costa**
*Personal Project Report: PP-LPBF-PMOLPM-001*