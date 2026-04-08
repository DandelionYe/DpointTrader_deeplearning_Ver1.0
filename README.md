# DpointTrader Deep Learning Basket Framework

This repository is a basket-level research and backtesting framework for A-share style panel data. It loads per-ticker CSV files, builds panel-safe features, trains cross-sectional models, evaluates them with time-aware validation, converts scores into portfolios, and runs backtests with reproducible experiment outputs.

The current implementation supports:

- Walk-forward validation: `wf`
- Walk-forward with embargo gap: `wf_embargo`
- Nested walk-forward validation: `nested_wf`
- Optional final holdout evaluation
- Config search with multiple candidates and multiple seeds
- Tabular models: `xgb`, `mlp`
- Sequence models with ticker-safe sequence construction: `lstm`, `gru`, `cnn`, `transformer`
- Single-run and rolling retrain workflows
- Experiment manifests, Excel reports, HTML reports, saved models, scores, and backtest artifacts

## Repository Scope

The main entry point is [main_basket.py](/J:/DpointTrader_deeplearning_Ver1.0/main_basket.py). It orchestrates:

1. Basket loading
2. Panel validation
3. Feature and label construction
4. Split planning
5. Search and model selection
6. Final evaluation on OOF or holdout
7. Portfolio construction and backtesting
8. Report and manifest export

Core modules:

- [basket_loader.py](/J:/DpointTrader_deeplearning_Ver1.0/basket_loader.py): loads basket folders and standardizes CSV inputs
- [feature_dpoint.py](/J:/DpointTrader_deeplearning_Ver1.0/feature_dpoint.py): panel feature and label engineering
- [splitters.py](/J:/DpointTrader_deeplearning_Ver1.0/splitters.py): walk-forward, embargo, nested WF, and final holdout split logic
- [search_space.py](/J:/DpointTrader_deeplearning_Ver1.0/search_space.py): candidate config sampling
- [search_engine.py](/J:/DpointTrader_deeplearning_Ver1.0/search_engine.py): formal search loop
- [sequence_builder.py](/J:/DpointTrader_deeplearning_Ver1.0/sequence_builder.py): ticker-safe sequence window generation
- [models.py](/J:/DpointTrader_deeplearning_Ver1.0/models.py): model registry and train/predict functions
- [panel_trainer.py](/J:/DpointTrader_deeplearning_Ver1.0/panel_trainer.py): panel-safe training, prediction, OOF evaluation, nested evaluation
- [portfolio_builder.py](/J:/DpointTrader_deeplearning_Ver1.0/portfolio_builder.py): score-to-portfolio conversion
- [backtester_engine.py](/J:/DpointTrader_deeplearning_Ver1.0/backtester_engine.py): backtest engine
- [rolling_retrainer.py](/J:/DpointTrader_deeplearning_Ver1.0/rolling_retrainer.py): rolling retraining and snapshot export
- [utils.py](/J:/DpointTrader_deeplearning_Ver1.0/utils.py): experiment directories, manifests, seeds, and utility helpers

## Data Contract

Input data is expected as one CSV file per ticker inside a basket folder such as `data/basket_1/`.

Supported source columns:

- `Date`
- `Open (CNY, qfq)`
- `High (CNY, qfq)`
- `Low (CNY, qfq)`
- `Close (CNY, qfq)`
- `Volume (shares)`

These are standardized internally to:

- `date`
- `open_qfq`
- `high_qfq`
- `low_qfq`
- `close_qfq`
- `volume`

Minimal example:

```csv
"Date","Open (CNY, qfq)","High (CNY, qfq)","Low (CNY, qfq)","Close (CNY, qfq)","Volume (shares)"
2024-01-01,10.0,10.5,9.8,10.2,1000000
2024-01-02,10.2,10.8,10.0,10.5,1200000
```

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not available, tabular workflows still run. Torch-dependent tests are skipped automatically.

## Main Workflows

### 1. Single Experiment

Basic XGBoost walk-forward run:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --model_type xgb ^
  --split_mode wf ^
  --use_holdout 0 ^
  --runs 10 ^
  --output_dir ./output_basket
```

Embargo plus holdout with MLP:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --model_type mlp ^
  --split_mode wf_embargo ^
  --embargo_days 5 ^
  --use_holdout 1 ^
  --holdout_ratio 0.15 ^
  --runs 10 ^
  --output_dir ./output_basket
```

Nested walk-forward search:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --model_type xgb ^
  --split_mode nested_wf ^
  --n_outer_folds 3 ^
  --n_inner_folds 2 ^
  --use_holdout 0 ^
  --runs 10 ^
  --output_dir ./output_basket
```

Sequence model run:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --model_type lstm ^
  --seq_len 20 ^
  --num_layers 1 ^
  --split_mode wf ^
  --use_holdout 0 ^
  --runs 5 ^
  --device cpu ^
  --output_dir ./output_basket
```

### 2. Continue Training / Continue Search

Continue from the latest experiment:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --continue_from latest ^
  --additional_runs 20
```

Reuse an existing model for evaluation-only flow:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --continue_from ./output_basket/exp_001 ^
  --runs 0
```

### 3. Rolling Retrain

Expanding window rolling retrain:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --run_mode rolling ^
  --rolling_mode expanding ^
  --retrain_frequency monthly ^
  --min_history_days 120 ^
  --model_type xgb ^
  --runs 3 ^
  --output_dir ./output_basket
```

Fixed-length rolling window:

```bash
python main_basket.py ^
  --basket basket_1 ^
  --data_root ./data ^
  --run_mode rolling ^
  --rolling_mode rolling ^
  --rolling_window_length 252 ^
  --retrain_frequency monthly ^
  --min_history_days 120 ^
  --model_type xgb ^
  --runs 3 ^
  --output_dir ./output_basket
```

## Validation and Evaluation Model

The framework separates:

- Search-time evaluation on `search_X/search_y`
- Optional final holdout evaluation on `holdout_X/holdout_y`
- Final reporting on `final_eval_scores_df`

Evaluation modes:

- `evaluation_split = "oof"` when there is no holdout
- `evaluation_split = "holdout"` when a final holdout is used
- rolling snapshots use forward evaluation windows after each retrain date

Split metadata is exported into experiment manifests under `split_info`.

## Changelog

### 2026-03-22

Historical root-level debug logs were consolidated into this note and removed from the repository.

Those temporary logs captured one-off PowerShell runs of `main_basket.py` related to:

- GPU runtime detection
- backtest window override checks
- MLP/GPU smoke execution
- search flow verification

They had no runtime role in the framework, were not read by any code path, and only served as ad hoc debugging artifacts during development.

## Search Engine

Search is no longer just repeated runs of one fixed config. The framework now:

- samples candidate configs from [search_space.py](/J:/DpointTrader_deeplearning_Ver1.0/search_space.py)
- assigns seeds per candidate
- evaluates each candidate on WF or nested WF
- ranks candidates with `selection_metric`
- retrains the best config on the search training set before final evaluation

Supported selection metrics:

- `rank_ic_mean`
- `topk_return_mean`

Search summary is exported into the manifest under `search_summary`.

## Supported Models

Tabular:

- `xgb`
- `mlp`

Sequence:

- `lstm`
- `gru`
- `cnn`
- `transformer`

Sequence models are panel-safe:

- windows are built per ticker
- rows are sorted by date within each ticker
- no sequence window crosses ticker boundaries
- predictions are returned only for valid post-warmup rows

## Output Structure

Single-run experiment directory:

```text
output_basket/
  exp_001/
    manifest.json
    results.xlsx
    report.html
    models/
      model_YYYYMMDD_HHMMSS.joblib
    artifacts/
      scores.csv
      equity_curve.csv
      trades.csv
```

Rolling retrain snapshot directory:

```text
output_basket/
  exp_001/
    manifest.json
    snapshots/
      snapshot_001/
        manifest.json
        model.joblib
        scores.csv
        equity_curve.csv
```

Manifest contents include:

- `best_config`
- `metrics`
- `evaluation_split`
- `search_runs_completed`
- `split_info`
- `search_summary`

## Testing

Run the full test suite:

```bash
pytest -q
```

Current status in this repository:

- full test suite passes
- validation, leakage, reproducibility, sequence boundary, rolling retrain, and end-to-end coverage are included

Important test files:

- [tests/test_split_plan.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_split_plan.py)
- [tests/test_holdout_isolation.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_holdout_isolation.py)
- [tests/test_embargo_integrity.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_embargo_integrity.py)
- [tests/test_no_leakage_panel.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_no_leakage_panel.py)
- [tests/test_reproducibility_panel.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_reproducibility_panel.py)
- [tests/test_sequence_builder.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_sequence_builder.py)
- [tests/test_sequence_boundary_by_ticker.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_sequence_boundary_by_ticker.py)
- [tests/test_sequence_prediction_shape.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_sequence_prediction_shape.py)
- [tests/test_search_engine.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_search_engine.py)
- [tests/test_torch_models_smoke.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_torch_models_smoke.py)
- [tests/test_end_to_end.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_end_to_end.py)
- [tests/test_rolling_retrainer.py](/J:/DpointTrader_deeplearning_Ver1.0/tests/test_rolling_retrainer.py)

## Notes

- The codebase is designed around time-aware evaluation. Do not interpret full-sample fitted predictions as valid historical out-of-sample performance.
- Sequence models intentionally produce fewer scored rows because of warmup length.
- Sparse panel warnings can be normal when tickers have different listing histories.

## Help

Show the CLI reference:

```bash
python main_basket.py --help
```
