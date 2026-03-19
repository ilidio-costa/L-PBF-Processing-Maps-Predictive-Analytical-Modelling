# Contributing to L-PBF Processing Maps

First off, thank you for considering contributing to this project! It is research-oriented tools like this that help move additive manufacturing forward.

## How Can I Contribute?

### Expanding the Material Database
We aim to support as many alloys as possible. To add a new material:
1. Navigate to the `materials/` directory.
2. Create a new `.json` file (e.g., `Ti64.json`).
3. Ensure you include all required fields: `rho`, `C_p`, `k`, `T_b`, `T_m`, `A`, `alpha`, and `electrical_resistivity`.
4. Please provide references for your data.

### Adding New Defect Criteria
If you have a new analytical model for defects (e.g., a specific porosity threshold):
1. Create a new python file in `src/defects/`.
2. Follow the standardized API: `def check(dimensions, process_parameters, material) -> bool`.
3. Provide references for your model as it will be added to the documentation.

### Reporting Bugs
If you find a bug in the Eagar-Tsai or Rubenchik implementations, please open an Issue. Include:
* The material JSON used.
* The specific $P-v$ parameters that caused the error.
* A description of the expected vs. actual result.

## Pull Request Process
1. Fork the repo and create your branch from `main`.
2. If you've added code, ensure your changes don't break the existing tests in the `test/` directory.
3. Update the README.md or documentation if you changed core logic.
4. Submit the PR with a clear description of your changes and the scientific references supporting them.

## Community & Conduct
By participating in this project, you agree to abide by standard academic integrity and open-source cooperation.

Your name will be added to the contributors list in the README.md, and your contributions will be acknowledged in any publications resulting from this project.