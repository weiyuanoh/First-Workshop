"""Portfolio construction demo with forecasted expected returns (Ridge/LASSO) + forecast evaluation + alpha.

This script:
1) Downloads (or simulates) asset prices -> monthly returns
2) Downloads (or simulates) Fama-French factors (Mkt-RF, SMB, HML, RF)
3) Builds rolling forecasts of next-month EXCESS returns using lagged factors with RidgeCV/LassoCV
4) Constructs MSR portfolios using predicted ER and trailing covariance
5) Evaluates forecast quality (IC, MSE, OOS R^2, hit-rate) and strategy alpha (FF3 regression)

Examples:
    # Online mode (internet required)
    python run_portfolio_backtest_demo_v2.py --tickers AAPL MSFT AMZN NVDA XOM JPM --start 2010-01-01 --end 2024-12-31 --window 60

    # Offline mode (synthetic data; validates pipeline)
    python run_portfolio_backtest_demo_v2.py --offline-demo
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV, LassoCV
import yfinance as yf
from datetime import date 
import utils as erk


def load_ff_factors_local(csv_path: str) -> pd.DataFrame:
    # Find header row (line starting with ",Mkt-RF")
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    header_i = next(i for i, line in enumerate(lines) if line.strip().startswith(",Mkt-RF"))

    ff = pd.read_csv(csv_path, skiprows=header_i)
    ff = ff.rename(columns={ff.columns[0]: "Date"})
    ff["Date"] = ff["Date"].astype(str).str.strip()
    ff = ff[ff["Date"].str.match(r"^\d{6}$")].copy()   # keep only YYYYMM
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m")
    ff = ff.set_index(ff["Date"].dt.to_period("M")).drop(columns=["Date"])
    ff = ff[["Mkt-RF", "SMB", "HML", "RF"]].astype(float) / 100.0
    return ff

def to_monthly_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    prices_m = adj_close.resample("M").last()
    rets_m = prices_m.pct_change().dropna()
    rets_m.index = rets_m.index.to_period("M")
    return rets_m


def _ols_alpha_tstat(y: pd.Series, X: pd.DataFrame):
    """
    Returns alpha (intercept), its t-stat, and full params for y ~ alpha + X.
    Uses statsmodels if available, else numpy OLS with classical SE.
    """
    y = y.dropna()
    X = X.loc[y.index].copy().dropna()
    y = y.loc[X.index]

    try:
        import statsmodels.api as sm
        res = sm.OLS(y, sm.add_constant(X)).fit()
        alpha = float(res.params["const"])
        tstat = float(res.tvalues["const"])
        return alpha, tstat, res.params
    except Exception:
        X_ = np.column_stack([np.ones(len(X)), X.values])
        y_ = y.values.reshape(-1, 1)
        beta = np.linalg.lstsq(X_, y_, rcond=None)[0]
        yhat = X_ @ beta
        resid = y_ - yhat
        n, k1 = X_.shape
        s2 = float((resid.T @ resid) / (n - k1))
        covb = s2 * np.linalg.inv(X_.T @ X_)
        se_alpha = float(np.sqrt(covb[0, 0]))
        alpha = float(beta[0, 0])
        tstat = alpha / se_alpha if se_alpha > 0 else np.nan
        params = pd.Series([alpha] + list(beta[1:, 0]), index=["const"] + list(X.columns))
        return alpha, tstat, params


def msr_forecasted_er_backtest(
    mret: pd.DataFrame,
    ff: pd.DataFrame,
    window: int = 60,
    model: str = "ridge",
    use_excess: bool = True,
    cv_folds: int = 5,
):
    """
    Predict next-month (excess) returns using lagged FF3 factors, then compute MSR weights.

    Features at time t are factors_{t-1} predicting return_t.
    At rebalance for month t, train on [t-window, t-1] and predict at t.

    Returns:
        pret_total: pd.Series (total portfolio returns)
        wts: pd.DataFrame (weights time series)
        yhat: pd.DataFrame (predicted excess returns per asset)
        yreal_excess: pd.DataFrame (realized excess returns per asset)
        ic: pd.Series (cross-sectional IC each month)
    """
    mret = mret.copy()
    ff = ff.copy()

    if not isinstance(mret.index, pd.PeriodIndex):
        mret.index = pd.to_datetime(mret.index).to_period("M")
    if not isinstance(ff.index, pd.PeriodIndex):
        ff.index = pd.to_datetime(ff.index).to_period("M")

    common_idx = mret.index.intersection(ff.index)
    mret = mret.loc[common_idx].dropna(how="any")
    ff = ff.loc[mret.index].dropna()

    X = ff[["Mkt-RF", "SMB", "HML"]].copy()
    X_lag = X.shift(1).dropna()

    if use_excess:
        y_excess = mret.sub(ff["RF"], axis=0)
    else:
        y_excess = mret.copy()

    y_excess = y_excess.loc[X_lag.index].dropna(how="any")
    X_lag = X_lag.loc[y_excess.index]
    dates = y_excess.index

    ridge_alphas = np.logspace(-4, 4, 50)
    lasso_alphas = np.logspace(-6, 1, 60)

    port_rets_total, weights_list, yhat_rows, ic_list = [], [], [], []

    for t in range(window, len(dates)):
        train_idx = dates[t - window:t]
        pred_idx = dates[t]

        X_train = X_lag.loc[train_idx].values
        X_pred = X_lag.loc[[pred_idx]].values

        cov = mret.loc[train_idx].cov()

        er_hat = []
        for col in y_excess.columns:
            y_train = y_excess.loc[train_idx, col].values

            if model.lower() == "ridge":
                reg = RidgeCV(alphas=ridge_alphas, fit_intercept=True).fit(X_train, y_train)
            elif model.lower() == "lasso":
                reg = LassoCV(alphas=lasso_alphas, cv=cv_folds, fit_intercept=True, max_iter=20000).fit(X_train, y_train)
            else:
                raise ValueError("model must be 'ridge' or 'lasso'")

            er_hat.append(float(reg.predict(X_pred)[0]))

        er_hat = pd.Series(er_hat, index=y_excess.columns)

        rf = 0.0 if use_excess else float(ff.loc[pred_idx, "RF"])
        w = pd.Series(erk.msr(rf, er_hat, cov), index=y_excess.columns)

        r_realized_total = float((w * mret.loc[pred_idx]).sum())

        y_real_ex = y_excess.loc[pred_idx]
        if y_real_ex.std(ddof=0) > 0 and er_hat.std(ddof=0) > 0:
            ic_val = float(np.corrcoef(er_hat.values, y_real_ex.values)[0, 1])
        else:
            ic_val = np.nan

        weights_list.append(w)
        port_rets_total.append(r_realized_total)
        yhat_rows.append(er_hat.rename(pred_idx))
        ic_list.append((pred_idx, ic_val))

    wts = pd.DataFrame(weights_list, index=dates[window:], columns=y_excess.columns)
    pret_total = pd.Series(port_rets_total, index=dates[window:], name=f"MSR_{model.upper()}_Forecast")
    yhat = pd.DataFrame(yhat_rows)
    yreal_excess = y_excess.loc[pret_total.index]
    ic = pd.Series({d: v for d, v in ic_list}, name=f"IC_{model.upper()}")
    return pret_total, wts, yhat, yreal_excess, ic


def evaluate_forecasts(yhat: pd.DataFrame, yreal: pd.DataFrame, ic: pd.Series, label: str):
    idx = yhat.index.intersection(yreal.index)
    yhat = yhat.loc[idx]
    yreal = yreal.loc[idx]

    err = (yreal - yhat)
    mse = float((err**2).stack().mean())
    mae = float(err.abs().stack().mean())

    sse = float((err**2).stack().sum())
    ybar = float(yreal.stack().mean())
    sst = float(((yreal - ybar)**2).stack().sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan

    hit = float(((np.sign(yhat) == np.sign(yreal)).stack().mean()))

    ic_clean = ic.dropna()
    ic_mean = float(ic_clean.mean()) if len(ic_clean) else np.nan
    ic_std = float(ic_clean.std(ddof=1)) if len(ic_clean) > 1 else np.nan
    ic_t = ic_mean / (ic_std / np.sqrt(len(ic_clean))) if (ic_std and ic_std > 0) else np.nan

    print(f"\n=== Forecast Evaluation: {label} ===")
    print(f"Pooled MSE:  {mse:.6f}")
    print(f"Pooled MAE:  {mae:.6f}")
    print(f"OOS R^2:     {r2:.4f}")
    print(f"Hit-rate:    {hit:.3f}")
    print(f"Mean IC:     {ic_mean:.4f}")
    print(f"IC t-stat:   {ic_t:.2f}   (n={len(ic_clean)})")

    return {"mse": mse, "mae": mae, "r2": r2, "hit": hit, "ic_mean": ic_mean, "ic_t": ic_t}


def evaluate_alpha(strategy_rets: pd.Series, ff: pd.DataFrame, label: str):
    if not isinstance(strategy_rets.index, pd.PeriodIndex):
        strategy_rets.index = pd.to_datetime(strategy_rets.index).to_period("M")
    ff = ff.copy()
    if not isinstance(ff.index, pd.PeriodIndex):
        ff.index = pd.to_datetime(ff.index).to_period("M")

    idx = strategy_rets.index.intersection(ff.index)
    r = strategy_rets.loc[idx]
    rf = ff.loc[idx, "RF"]
    y = (r - rf).dropna()
    X = ff.loc[y.index, ["Mkt-RF", "SMB", "HML"]]

    alpha_m, alpha_t, params = _ols_alpha_tstat(y, X)
    alpha_ann = (1 + alpha_m) ** 12 - 1

    print(f"\n=== Alpha (FF3) for {label} ===")
    print(f"Monthly alpha:    {alpha_m:.5f}")
    print(f"Annualized alpha: {alpha_ann:.2%}")
    print(f"Alpha t-stat:     {alpha_t:.2f}")

    return {"alpha_monthly": alpha_m, "alpha_annualized": alpha_ann, "alpha_t": alpha_t, "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN", "NVDA", "XOM", "JPM"])
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--offline-demo", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    
    adj = yf.download(args.tickers, start=args.start, end=args.end, auto_adjust=True, progress=False)["Close"]
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(args.tickers[0])
    mret = to_monthly_returns(adj)
    ff = load_ff_factors_local("F-F_Research_Data_Factors.csv")  # relative path beside your script


    ew = erk.backtest_ws(mret, estimation_window=args.window, weighting=erk.weight_ew)
    gmv_s = erk.backtest_ws(mret, estimation_window=args.window, weighting=erk.weight_gmv, cov_estimator=erk.sample_cov)
    gmv_cc = erk.backtest_ws(mret, estimation_window=args.window, weighting=erk.weight_gmv, cov_estimator=erk.cc_cov)
    gmv_sh = erk.backtest_ws(mret, estimation_window=args.window, weighting=erk.weight_gmv, cov_estimator=erk.shrinkage_cov, delta=0.5)

    msr_ridge_ret, msr_ridge_wts, ridge_yhat, ridge_yreal, ridge_ic = msr_forecasted_er_backtest(
        mret, ff, window=args.window, model="ridge", use_excess=True
    )
    msr_lasso_ret, msr_lasso_wts, lasso_yhat, lasso_yreal, lasso_ic = msr_forecasted_er_backtest(
        mret, ff, window=args.window, model="lasso", use_excess=True
    )

    bt = pd.DataFrame({
        "EW": ew,
        "GMV-Sample": gmv_s,
        "GMV-CC": gmv_cc,
        "GMV-Shrink0.5": gmv_sh,
        "MSR-Ridge-Fcst": msr_ridge_ret,
        "MSR-Lasso-Fcst": msr_lasso_ret,
    }).dropna()

    print("\nSummary stats (monthly -> annualized):")
    print(erk.summary_stats(bt))

    evaluate_forecasts(ridge_yhat, ridge_yreal, ridge_ic, "Ridge")
    evaluate_forecasts(lasso_yhat, lasso_yreal, lasso_ic, "LASSO")

    # Alpha: highlight the key talking point
    evaluate_alpha(bt["EW"], ff, "EW (benchmark)")
    evaluate_alpha(bt["MSR-Ridge-Fcst"], ff, "MSR Ridge Forecast")
    evaluate_alpha(bt["MSR-Lasso-Fcst"], ff, "MSR LASSO Forecast")

    if not args.no_plot:
        wealth = (1 + bt).cumprod()
        wealth.plot(title="Strategy Wealth (normalized)")
        plt.show()

        pd.DataFrame({"IC_Ridge": ridge_ic, "IC_Lasso": lasso_ic}).dropna().rolling(12).mean().plot(
            title="12M Rolling Information Coefficient (IC)"
        )
        plt.show()

    print("\nLast MSR-Ridge weights:")
    print(msr_ridge_wts.iloc[-1].sort_values(ascending=False))
    print("\nLast MSR-LASSO weights:")
    print(msr_lasso_wts.iloc[-1].sort_values(ascending=False))


if __name__ == "__main__":
    main()
