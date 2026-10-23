# St. Louis GHG Knowledge-Based Decision Intelligence

Reproducible code for the short communication **“Data-Driven AI and
Knowledge-Based Decision Intelligence for St. Louis GHG Projections and
Emissions Reduction Planning.”**

The study aggregates public NREL SLOPE county projections for the 15-county
St. Louis MO–IL metropolitan statistical area, tests a knowledge-guided
XGBoost model against three baselines on an untouched 2043–2050 interval,
and ranks regional intervention portfolios through 2050. The pipeline creates
exactly two manuscript figures and two result tables.

## Reproduce in Git Bash

```bash
git clone https://github.com/narimallae-wq/stl-ghg-knowledge-decision-intelligence.git
cd stl-ghg-knowledge-decision-intelligence
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
python -m pip install -r requirements.txt
python analysis.py
```

Use `python analysis.py --refresh-data` to download every county-scenario
series again. Without that flag, the committed data snapshot is used for a
fast, deterministic rerun.

## Google Colab

Open `stl_ghg_decision_intelligence.ipynb` in Colab and run both cells. The
notebook already points to this repository, installs the pinned environment,
executes the same analysis, and previews the two figures and tables.

## Compact repository map

- `analysis.py` — download, validation, models, decision rules, and outputs
- `data/stl_slope_scenarios.csv` — versioned public-data snapshot (31,320 rows)
- `outputs/` — exactly two PNG figures and two CSV tables
- `stl_ghg_decision_intelligence.ipynb` — minimal Colab runner
- `requirements.txt` — tested package versions

## Evidence and interpretation limits

NREL SLOPE supplies annual **energy-related CO2 scenario projections**, not
observed future emissions and not a complete community GHG inventory. The
East-West Gateway 2022 inventory (44.5 MMTCO2e) is used only as an external
boundary check because its gases, sectors, and accounting methods differ.
The urban-planning row is a transparent 5–15% vehicle-miles-traveled
sensitivity rule; it is not an NREL scenario. Results support comparative
planning under stated assumptions and should not be read as certain 2050
predictions.

## Public sources

- [NREL SLOPE Scenario Planner](https://maps.nrel.gov/slope/)
- [NREL SLOPE methodology](https://www.nrel.gov/gis/slope)
- [St. Louis Regional Greenhouse Gas Reduction Plan](https://www.ewgateway.org/what-we-do/air-quality-planning/climate-pollution-reduction-grant-program/)
- [U.S. EPA St. Louis Priority Climate Action Plan](https://www.epa.gov/system/files/documents/2024-03/st-louis-mo-il-msa-pcap.pdf)

## Citation and license

Please cite the accompanying manuscript and the source datasets. Code is
provided for academic reproducibility. Source data remain subject to their
publishers' terms and attribution requirements.
