# Freight rate prediction

Predicts `posted_rate` for the 12,000 loads in `validation.csv` and fills the fixed December chart.

## Data

The assessment files are included in the `spotter/` folder:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`
- `score.py`

The code looks for them next to itself first, then in `spotter/`, then inside a zip (set `ZIP_PATH` in the notebook, or pass `--zip your.zip` to the script).

## Run

Option 1, the notebook:

```bash
python -m pip install -r requirements.txt jupyter
jupyter notebook freight_rate.ipynb
```

Then choose Run All. Set `RUN_EVAL = False` in the first code cell to skip the validation tables.

Option 2, the script (same pipeline):

```bash
python -m pip install -r requirements.txt
python freight_rate.py --out outputs --eval
```

`--eval` also prints the forward-in-time validation table.

Both give identical predictions.

Outputs in `outputs/`:
- `validation_predictions.csv` (load_id, predicted_rate)
- `december_chart_inputs.csv`, the December file with predicted_rate filled
- `scorer_results/candidate_december.png`, made by the provided score.py (the notebook and script both run it)
- `validation_results.csv`, the validation table
- `daily_level.png`, the daily market level used in the report

## Repo contents

- `freight_rate.ipynb`: the main walkthrough, with data checks, findings, model, validation and outputs saved
- `freight_rate.py`: the same pipeline as a single script
- `requirements.txt`: dependencies
- `validation_predictions.csv`: the predictions (load_id, predicted_rate)
- `spotter/`: the assessment files

## Approach

1. Target is log(rate per mile). Weights with a flipped sign are made positive.
2. Daily market level: a GBM on load features only gives a load effect. Its residual, averaged per day, is the market level of that day.
3. That level is explained by a small linear model on the daily mean of market_index, quote_signal and a time trend, so it can be extrapolated to Nov and Dec.
4. A second GBM (3 seeds, Huber loss) learns the row-level structure with the daily level removed.
5. Final rate = distance x exp(row model + daily level).

Validation is forward in time (train on earlier months, test on later months) because the validation set is Nov and Dec while the labeled data is Jan to Oct. A shuffled KFold would leak the daily market level and look better than it is. The validation table reports MAE, MAPE and RMSE on all rows, and also on "clean" rows (rate per mile between 1 and 5). About 1.5% of labels are random spikes that no model can predict, and they dominate RMSE on all rows.

## Assumptions

- The model carries an upward time trend into Nov and Dec. Forward tests favour including it, but if the trend does not continue, predictions would run about 4% high.
- The December chart rows have no market_index or quote_signal, so the daily averages from the feature columns of `validation.csv` are used. No labels are used for validation rows.
- Negative weights (a sign error in 292 train and 145 validation rows) are flipped to positive on an in-memory copy. The original files are never modified.
- Validation numbers come from holding out later months of `train-test.csv`. The real score is calculated by Spotter.
