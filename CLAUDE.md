# JAX-PLUTO

**Description:** A high-performance rewrite of the [gPLUTO](https://gitlab.com/PLUTO-code/gPLUTO) MHD code using JAX. This project aims to replicate the shock-capturing and finite-volume capabilities of the reference gPLUTO implementation while leveraging JAX for hardware acceleration (GPU/TPU) and automatic differentiation.

## Tech Stack
*   **Core:** Python 3.10+, JAX, NumPy
*   **Physics/Numerics:** Finite Volume Methods, Riemann Solvers (HLL, HLLC, Roe), MHD/Hydrodynamics
*   **Utilities:** `chex` (testing/assertions), `jaxtyping` (array shape hints)
*   **Viz:** Matplotlib 

## Development Commands

### Environment
*   **Install Dependencies:** `pip install -r requirements.txt`
*   **Install Editable:** `pip install -e .`
*   **Update Deps:** `pip install --upgrade jax jaxlib numpy`

### Running
*   **Run Simulation:** `python scripts/run_simulation.py --config configs/shock_tube.yaml`
*   **Run Single Step (Debug):** `python src/main.py --debug --steps 1`
*   **Profile:** `python -m cProfile -o profile.stats src/main.py`

### Testing
*   **Run All Tests:** `pytest`
*   **Run Fast Tests:** `pytest -m "not slow"`
*   **Run Specific Test:** `pytest tests/test_riemann.py::test_hllc_solver`
*   **Watch Tests:** `pytest-watch`

### Linting & Formatting
*   **Format Code:** `black src tests`
*   **Lint:** `ruff check src tests`
*   **Type Check:** `mypy src` (focus on array shape consistency)

## Code Guidelines

### JAX Specifics & Functional Purity
*   **Pure Functions:** All numerical kernels MUST be pure. No side effects. No global state mutation.
*   **Array Immutability:** Never use in-place assignment (e.g., `x[i] = val`). Use `x = x.at[i].set(val)`.
*   **JIT Compilation:** Design core update loops to be JIT-compatible.
    *   Avoid Python control flow (`if`/`for`) depending on data values; use `jax.lax.cond`, `jax.lax.scan`, or `jax.lax.select`.
    *   Ensure static argument shapes where possible.
*   **PRNG Handling:** Pass `jax.random.PRNGKey` explicitly. Never rely on global random seeds.

### Numerical Implementation
*   **Precision:** Default to `float64` for validation, allow toggling to `float32` for performance via config.
*   **Vectorization:** Prefer `jax.vmap` over explicit loops for computing fluxes across grid faces.
*   **Consistency:** Coordinate systems (Cartesian, Cylindrical) should be handled via metric terms, not separate codebases if possible.

### Naming & Style
*   **Variables:** `snake_case` for variables and functions.
*   **Physics Variables:** Use standard notation: `rho` (density), `vel` (velocity vector), `prs` (pressure), `bx`, `by`, `bz` (magnetic fields).
*   **Classes:** `PascalCase`. Use `equinox.Module` or `flax.struct.dataclass` if state management is required.
*   **Docstrings:** Google Style. **Crucial:** Document expected array shapes in inputs/outputs.
    *   *Example:* `Args: u (Array): State vector of shape [nx, ny, n_vars].`

### Type Hinting
Use `jaxtyping` to strictly define array dimensions.
```python
from jaxtyping import Float, Array
def compute_flux(u: Float[Array, "nx ny 8"]) -> Float[Array, "nx ny 8"]:
    ...
 
