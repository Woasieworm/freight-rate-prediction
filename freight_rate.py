"""
Freight rate prediction (Spotter assessment).

Run:
    python freight_rate.py --out outputs
    python freight_rate.py --out outputs --eval     (also prints validation tables)

Put the assessment files next to this script (or in a folder named spotter). If they are in a zip, add --zip spotter.zip.

Approach in one paragraph
-------------------------
posted_rate is almost deterministic once two things are separated:
  1. the load itself (distance, equipment, weight, lane geometry)  -> row model
  2. the market level of that day                                  -> daily model
The target is log(rate per mile). Stage 1 learns the load effect, the daily mean of its residual
is the market level of each day, and a small linear model explains that level from the daily
market_index, quote_signal and a time trend (a linear model can extrapolate to Nov and Dec,
a tree model cannot). Stage 2 learns the row-level structure with the daily level taken out.
About 1.5% of labels are random unpredictable spikes (x0.2-0.6 or x2-5), so every GBM uses a
Huber loss instead of squared error.
"""
import argparse
import os
import subprocess
import sys
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

T0 = pd.Timestamp("2025-01-01")
SEEDS = (1, 2, 3)
STATIC = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "logd", "weight", "eq"]
ROW = STATIC + ["market_index", "quote_signal", "mi_dev", "qs_dev", "dow"]
GBM = dict(
    objective="huber", alpha=0.15, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4,
)


# data
REQUIRED = ["train-test.csv", "validation.csv", "validation-predictions-template.csv", "december-chart-inputs.csv", "score.py"]


def resolve(folder, name):
    """Path of a data file, accepting hyphens or underscores in the name (train-test.csv or train_test.csv)."""
    for n in (name, name.replace("-", "_")):
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    return None


def find_data_dir(zip_path, out_dir):
    """Find the assessment files. Accepts loose files next to the code, a folder named spotter or data,
    or the files inside a zip. The original files are only read, never changed."""
    for folder in (".", "spotter", "data"):
        if all(resolve(folder, f) for f in REQUIRED):
            return folder
    if os.path.exists(zip_path):
        extract = os.path.join(out_dir, "extracted_spotter")
        if not os.path.exists(extract):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract)
        for root, _, _ in os.walk(extract):
            if all(resolve(root, f) for f in REQUIRED):
                return root
    raise FileNotFoundError(
        "Could not find the assessment files. Put train-test.csv, validation.csv, "
        "validation-predictions-template.csv, december-chart-inputs.csv and score.py next to this script "
        "(or in a folder named spotter), or pass --zip with the path to a zip containing them."
    )


def load(data_dir):
    train = pd.read_csv(resolve(data_dir, "train-test.csv"), parse_dates=["date"])
    valid = pd.read_csv(resolve(data_dir, "validation.csv"), parse_dates=["date"])
    template = pd.read_csv(resolve(data_dir, "validation-predictions-template.csv"))
    dec = pd.read_csv(resolve(data_dir, "december-chart-inputs.csv"), parse_dates=["date"])
    return train, valid, template, dec


def add_basic_features(df):
    df = df.copy()
    df["weight"] = df["weight"].abs()  # 292 train and 145 validation weights have a flipped sign
    df["eq"] = df["equipment"].map({"Dry Van": 0, "Flatbed": 1, "Reefer": 2})
    df["logd"] = np.log(df["distance"])
    df["dow"] = df["date"].dt.dayofweek
    return df


def build_daily_table(train, valid):
    """One row per calendar day, built from market_index / quote_signal of train AND validation
    rows (features only, no labels). Those columns are noisy per row, so the daily mean is a
    much cleaner market signal."""
    both = pd.concat([train[["date", "market_index", "quote_signal"]], valid[["date", "market_index", "quote_signal"]]])
    d = both.groupby("date").agg(d_mi=("market_index", "mean"), d_qs=("quote_signal", "mean")).sort_index()
    d["tday"] = (d.index - T0).days / 30.0
    d["mi_l1"] = d["d_mi"].shift(1).bfill()
    d["mi_l2"] = d["d_mi"].shift(2).bfill()
    return d


def attach_daily(df, daily):
    df = df.join(daily[["d_mi", "d_qs", "tday"]], on="date")
    df["mi_dev"] = df["market_index"].fillna(df["d_mi"]) - df["d_mi"]
    df["qs_dev"] = df["quote_signal"] - df["d_qs"]
    return df


# model
def _gbm(seed, n):
    return lgb.LGBMRegressor(**GBM, n_estimators=n, random_state=seed)


def _daily_design(d, kind):
    if kind == "A":
        cols = [d["d_mi"], d["d_qs"], d["tday"]]
    else:
        cols = [d["d_mi"], d["mi_l1"], d["mi_l2"], d["d_qs"], d["tday"]]
    return np.c_[(np.ones(len(d)),) + tuple(c.values for c in cols)]


class FreightModel:
    def fit(self, tr, daily):
        y = np.log(tr["posted_rate"].values / tr["distance"].values)

        # stage 1: load effect only, out-of-fold so the daily level is not memorised
        oof = np.zeros(len(tr))
        for a, b in KFold(4, shuffle=True, random_state=0).split(tr):
            m = _gbm(1, 400).fit(tr.iloc[a][STATIC], y[a])
            oof[b] = m.predict(tr.iloc[b][STATIC])
        r0 = y - oof
        sd = 1.4826 * np.median(np.abs(r0 - np.median(r0)))
        keep = np.abs(r0) < 6 * sd  # drop the random spikes before averaging per day
        level = pd.Series(r0[keep]).groupby(tr["date"].values[keep]).mean()
        level.index = pd.to_datetime(level.index)
        self.level = level

        # daily model: market level of a day from market_index, quote_signal, time trend
        dd = daily.loc[level.index]
        self.beta = {k: np.linalg.lstsq(_daily_design(dd, k), level.values, rcond=None)[0] for k in "AB"}

        # stage 2: row model with the actual daily level taken out
        off = tr["date"].map(level).fillna(0).values
        self.row_models = [_gbm(s, 500).fit(tr[ROW], y - off) for s in SEEDS]
        return self

    def daily_level(self, dates, daily):
        d = daily.loc[dates]
        pred = np.mean([_daily_design(d, k) @ self.beta[k] for k in "AB"], axis=0)
        return pd.Series(pred, index=d.index)

    def predict(self, df, daily):
        lvl = self.daily_level(sorted(df["date"].unique()), daily)
        row = np.mean([m.predict(df[ROW]) for m in self.row_models], axis=0)
        return df["distance"].values * np.exp(row + df["date"].map(lvl).values)


# evaluation
def metrics(y, p):
    e = p - y
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "mape": float(np.mean(np.abs(e) / y))}


def baseline_fit_predict(tr, te):
    """Plain LightGBM on the raw columns, squared error, no daily structure. Reference point."""
    cols = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight", "market_index", "quote_signal", "eq", "dow"]
    m = lgb.LGBMRegressor(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_estimators=600, random_state=1, verbose=-1, n_jobs=4)
    return m.fit(tr[cols], tr["posted_rate"]).predict(te[cols])


def evaluate(train, daily):
    rows = []
    mon = train["date"].dt.month
    clean_mask = (train["posted_rate"] / train["distance"]).between(1, 5)
    cities = sorted(set(train["pickup"]) | set(train["delivery"]))
    held = set(cities[::8])  # 1 city in 8, used for the unseen-city test
    tests = {
        "Jul-Aug (train Jan-Jun)": (mon < 7, mon.isin([7, 8])),
        "Sep-Oct (train Jan-Aug)": (mon < 9, mon.isin([9, 10])),
    }
    tests_city = {"Sep-Oct, unseen cities": (
        (mon < 9) & ~(train["pickup"].isin(held) | train["delivery"].isin(held)),
        mon.isin([9, 10]) & (train["pickup"].isin(held) | train["delivery"].isin(held)))}
    for name, (itr, ite) in {**tests, **tests_city}.items():
        tr, te = train[itr], train[ite]
        y = te["posted_rate"].values
        cm = clean_mask[ite].values
        fm = FreightModel().fit(tr, daily)
        for label, p in [("baseline LightGBM", baseline_fit_predict(tr, te)), ("final model", fm.predict(te, daily))]:
            a, c = metrics(y, p), metrics(y[cm], p[cm])
            rows.append({"test": name, "model": label, "rmse_all": a["rmse"], "mae_all": a["mae"], "mape_all": a["mape"],
                         "rmse_clean": c["rmse"], "mae_clean": c["mae"], "mape_clean": c["mape"]})
    res = pd.DataFrame(rows)
    return res


# main
def december_frame(train, valid, dec, daily):
    d = dec.copy()
    pk = train.loc[train["pickup"] == "Lexington", ["pickup_lat", "pickup_lon"]].iloc[0]
    dl = train.loc[train["delivery"] == "Fort Wayne", ["delivery_lat", "delivery_lon"]].iloc[0]
    d["pickup_lat"], d["pickup_lon"] = pk["pickup_lat"], pk["pickup_lon"]
    d["delivery_lat"], d["delivery_lon"] = dl["delivery_lat"], dl["delivery_lon"]
    d["market_index"] = d["date"].map(daily["d_mi"])
    d["quote_signal"] = d["date"].map(daily["d_qs"])
    d = add_basic_features(d)
    return attach_daily(d, daily)


def daily_plot(model, daily, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dates = daily.index
    fit = model.daily_level(dates, daily)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(model.level.index, model.level.values, lw=1.2, label="observed daily level (train)")
    ax.plot(fit.index, fit.values, lw=1.2, label="fitted from market_index, quote_signal, trend")
    ax.axvline(pd.Timestamp("2025-11-01"), color="grey", ls="--", lw=1)
    ax.text(pd.Timestamp("2025-11-03"), ax.get_ylim()[0], "validation period", va="bottom", fontsize=8, color="grey")
    ax.set_ylabel("log rate-per-mile level vs load effect")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="spotter.zip", help="only needed if the files are zipped")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = find_data_dir(args.zip, args.out)

    train, valid, template, dec = load(data_dir)
    daily = build_daily_table(train, valid)
    train, valid = add_basic_features(train), add_basic_features(valid)
    train, valid = attach_daily(train, daily), attach_daily(valid, daily)

    if args.eval:
        res = evaluate(train, daily)
        res.to_csv(os.path.join(args.out, "validation_results.csv"), index=False)
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(res.to_string(index=False))

    model = FreightModel().fit(train, daily)

    sub = template[["load_id"]].merge(valid.assign(predicted_rate=model.predict(valid, daily))[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert sub["predicted_rate"].notna().all() and len(sub) == 12000
    pred_path = os.path.join(args.out, "validation_predictions.csv")
    sub.to_csv(pred_path, index=False)

    d = december_frame(train, valid, dec, daily)
    out_dec = dec.copy()
    out_dec["date"] = out_dec["date"].dt.strftime("%Y-%m-%d")
    out_dec["predicted_rate"] = model.predict(d, daily)
    dec_path = os.path.join(args.out, "december_chart_inputs.csv")
    out_dec.to_csv(dec_path, index=False)

    daily_plot(model, daily, os.path.join(args.out, "daily_level.png"))

    score = resolve(data_dir, "score.py")
    r = subprocess.run([sys.executable, score, "--predictions", pred_path, "--december-predictions", dec_path,
                        "--output-dir", os.path.join(args.out, "scorer_results")], capture_output=True, text=True)
    print(r.stdout, r.stderr)


if __name__ == "__main__":
    main()
