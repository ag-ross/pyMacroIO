# Dynamic Disequilibrium Input–Output Model

A Python implementation of a simple plain-vanilla single-region Dynamic Disequilibrium Input–Output (IO) model with Leontief- and CES-style production, inventory dynamics, labour hiring and firing, and two shocks: the consumption shock (simple example) and an input-availability shock.

## Overview

In this setup of the Dynamic Disequilibrium[^d] IO model it is strictly single-region: one set of sectors, one technical-coefficient matrix, and one final-demand vector per period. It is discrete-time and sector-level. Production can be Leontief (traditional or adapted), linear, or CES. Inventories are managed with target levels and adjustment speeds; labour adjusts via hiring and firing subject to capacity bounds. Consumption demand follows a Muellbauer-style specification with persistence, labour income, and expectation terms. Simple examples use the consumption shock (intensity, duration, start period) and optionally the input-availability shock (reduction in effective availability of a key supplier sector over a duration). The baseline implementation uses IO data loaded from a Python pickle file; row and value-added identities are checked at initialisation. Monte Carlo uncertainty analysis over parameter distributions is supported, and results can be plotted as total output or percentage change from baseline, with optional uncertainty bands. Theoretical underpinnings and data sources are described in references [1–5] below; reference 4 gives the data source and licence for the example inputs (EXIOBASE).

[^d]: Persistent possibility of excess demand or supply, with quantity adjustment and rationing rather than instantaneous price-mediated market clearing.

This project provides:

- **Core model**: `SingleRegionInputOutputModel` with configurable production (Leontief / leontief.adapted / linear / CES), inventories, labour adjustment, consumption shock, and input-availability shock
- **Scenarios**: `ScenarioManager`, `Scenario`, and `ScenarioRunResult` for baseline and shocked runs; comparison to baseline (GDP and realised consumption)
- **Uncertainty**: `MonteCarloUncertaintyAnalysis` for parameter sampling, run ensembles, and metrics (mean, quantiles) for GDP, consumption, and gross output
- **Configuration**: `ModelConfig` with validation (`n_periods` > 0, `time_frequency` in `"daily"` or `"quarterly"`)

Further equations and implementation choices for the simple plain-vanilla single-region Dynamic Disequilibrium Input–Output (IO) model are documented in `docs/Mathematical_summary.pdf`.


## Requirements

- Python 3.8+
- NumPy
- Matplotlib

No formal installation is required. The repository may be cloned or copied and the application run from the project root so that the data file is found (default: `data/example_data.pkl`; or `ModelConfig.data_path` may be set to the correct path). The example data are derived from a subset of EXIOBASE data (see References, item 4).

## Quick Start

### Baseline run and plotting

```python
from pathlib import Path
from pyMacroIO import (
    ModelConfig,
    ScenarioManager,
    MonteCarloUncertaintyAnalysis,
    ENABLE_PLOTTING,
)

config = ModelConfig(
    n_periods=30,
    time_frequency="daily",
    prod_function="leontief",
    diagnostics=False,
)
manager = ScenarioManager(config)
baseline_run = manager.run_baseline(force=True)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=50)
mc.run_uncertainty_analysis(shock_scenario="baseline", seed=42)
uncertainty = mc.get_uncertainty_data_for_plotting()

if ENABLE_PLOTTING:
    baseline_run.model.plot_results(
        baseline_run.results,
        baseline_results=None,
        title_suffix="(Daily Baseline)",
        save_path=str(figures_dir / "baseline.png"),
        uncertainty_data=uncertainty,
    )
```

### Consumption-shock scenario

```python
from pyMacroIO import run_consumption_shock_scenario

scenario_run, baseline_run = run_consumption_shock_scenario(
    intensity=0.2,
    duration=3,
    start=2,
)

# Percentage deviation from baseline
from pyMacroIO import ScenarioManager
comparison = ScenarioManager.compare_to_baseline(scenario_run, baseline_run)
# comparison["gdp_pct"], comparison["consumption_pct"]
```

### Input-availability shock scenario

```python
from pyMacroIO import run_input_availability_shock_scenario

# Key supplier sector is used when input_sector_label is None
scenario_run, baseline_run = run_input_availability_shock_scenario(
    input_sector_label=None,
    reduction_pct=0.6,
    duration=3,
    start=2,
)
# run_input_availability_shock_all_prod_functions(...) plots all production functions in one figure
```

## Documentation

- **Mathematical summary**: `docs/Mathematical_summary.pdf` describes the simple plain-vanilla single-region Dynamic Disequilibrium Input–Output (IO) model: equations, data calibration, essential-input identification, production, inventories, labour, consumption, the consumption shock, and the input-availability shock.

## Key Features

### Core model

- **SingleRegionInputOutputModel**: simple plain-vanilla single-region Dynamic Disequilibrium IO model with Leontief / adapted Leontief / linear / CES production
- **ModelConfig**: Scenario-overridable parameters with validation (`n_periods`, `time_frequency`, `prod_function`, etc.)
- **Inventories**: Target levels and adjustment speed `tau`
- **Labour**: Hiring and firing with sector-specific speeds and capacity bounds

### Scenarios and shocks

- **ScenarioManager**: Baseline run (cached), scenario run with shock callables, comparison to baseline
- **Consumption shock (simple example)**: Scenarios apply it by setting `model.epsilon_[t]` over a start period and duration (e.g. via `run_consumption_shock_scenario`). When demand is the binding constraint, all production functions yield identical results.
- **Input-availability shock**: Effective availability of a chosen input sector is reduced by a fraction over a duration (e.g. via `run_input_availability_shock_scenario`). When `input_sector_label` is not specified, the key supplier sector (largest forward supply) is used. Production functions differ when input availability binds; linear can remain near baseline because it uses aggregate input (perfect substitution).
- **Overrides**: Expectations (\(\xi\)) and labour disruption (\(\delta\)) may be set via overrides on `model.xi_` and `model.delta_`; for a baseline or unshocked run they remain at defaults.

### Uncertainty

- **MonteCarloUncertaintyAnalysis**: Parameter distributions, sampling, run ensemble, and metrics (mean, std, quantiles) for GDP, consumption, and gross output; integration with `plot_results` for uncertainty bands. Uncertainty bands reflect parameter uncertainty only (shock uncertainty is not included). The bands may remain wide or widen after the shock has ended, since the sampled parameters govern dynamics in every period.

## Data

The default data file is a Python pickle (`data/example_data.pkl`) containing base-year IO and final-demand data. The following entries must be included: `sector_labels` (list of sector names), `Z0` (inter-industry flows, square matrix \(N\times N\)), `cons_vec`, `gov_vec`, `inv_vec`, `invnt_vec`, `exp_vec` (final-demand vectors of length \(N\)), `l0`, `cap0`, `tax0`, `imp0` (value-added components, length \(N\)), and `consumer_taxes_total`, `fd_imports_totals`. ROW and value-added identities are assumed and checked at load. Gross output is derived from the row identity. Definitions are given in `docs/Mathematical_summary.pdf` (Data and Parameter Calibration). The example data are derived from a subset of EXIOBASE data (see References, item 4).

## Outputs

- **figures/baseline.png**: Total output (or absolute values) for the baseline run; optional uncertainty bands (parameter uncertainty) when `uncertainty_data` is passed to `plot_results`.
- **figures/consumption_shock_all_prod_functions.png**: Percentage change from baseline for the consumption-shock scenario, all production functions in one plot; optional uncertainty bands (parameter uncertainty).
- **figures/input_availability_shock_all_prod_functions.png**: Percentage change from baseline for the input-availability shock (key supplier reduced), all production functions in one plot; optional uncertainty bands (parameter uncertainty). Produced when `ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT` is `True`.

These figures are produced when the main script is run with `ENABLE_PLOTTING` set to `True`:

```bash
python3 pyMacroIO.py
```

## Licence

The licence for this software is described in the LICENSE file. The licence for the example data is different; the full terms are specified by the data source. See [4] and the [EXIOBASE licence file](https://zenodo.org/records/15689391/preview/LICENSE.txt) for the authoritative conditions.

### How to cite

This implementation should be cited as: [0], building on [1–3]; [4] should also be cited when the included example data (a small subset of EXIOBASE) are used. The licence thereof must also be adhered to (see Licence section above).

## References

0. Ross, A. G. (2025). A Python implementation of a single-region Dynamic Disequilibrium Input–Output model. 

1. Pichler, A., Pangallo, M., del Rio-Chanona, R. M., Lafond, F., & Farmer, J. D. (2022). Forecasting the propagation of pandemic shocks with a dynamic input–output model. *Journal of Economic Dynamics and Control*, 144, 104527. <https://doi.org/10.1016/j.jedc.2022.104527>

2. Ross, A. G., McGregor, P. G., & Swales, J. K. (2024). Labour market dynamics in the era of technological advancements: The system-wide impacts of labour augmenting technological change. *Technology in Society*, 77, 102539. <https://doi.org/10.1016/j.techsoc.2024.102539>

3. Raseta, M., Ross, A. G., & Voegele, S. (2025). Macro-level implications of the energy system transition to net-zero carbon emissions: Identifying quick wins amid short-term constraints. *Economic Analysis and Policy*, 85, 1065–1078. <https://doi.org/10.1016/j.eap.2025.01.011>

4. Stadler, K., Wood, R., Bulavskaya, T., Södersten, C.-J., Simas, M., Schmidt, S., Usubiaga, A., Acosta-Fernández, J., Kuenen, J., Bruckner, M., Giljum, S., Lutter, S., Merciai, S., Schmidt, J. H., Theurl, M. C., Plutzar, C., Kastner, T., Eisenmenger, N., Erb, K.-H., … Tukker, A. (2025). EXIOBASE 3 (3.9.6) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.15689391>

5. Miller, R. E., \& Blair, P. D. (2009). Input-output analysis: foundations and extensions. Cambridge university press.

