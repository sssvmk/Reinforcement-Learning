# DATA NOTES (Stage 0)

- Data folder: `C:/Users/username/OneDrive - XXYY/spain enegry/` (read-only; paths in `config.yaml`).
- `energy_dataset.csv`: shape (35064, 29) - matches expectation.
- `weather_features.csv`: shape (178396, 17) - matches expectation.
- City names contain stray whitespace (e.g. " Barcelona"); strip them. Counts per city exceed 35,064 for Madrid (36,267), Bilbao (35,951), Seville (35,557), Barcelona (35,476); Valencia is below (35,145), so duplicates and gaps both exist (to analyze in Stage 1).
- Temperatures are in Kelvin (270.475 K ~ -2.7 C); convert to Celsius.
- Environment: torch 2.14.1 (CPU), sklearn 1.6.1, scipy 1.18.1, pandas 3.0.6, numpy 2.2.6, statsmodels, lightgbm, cvxpy, pyyaml all available. No installs needed.
- Prompt file confirmed to end with `=== END OF PROMPT ===`.
- Project code lives in `energy_procurement/`; outputs go to `energy_procurement/outputs/`.
