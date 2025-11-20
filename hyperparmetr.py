# timeseries_prophet_sarimax_bayes.py
# Requires: numpy, pandas, matplotlib, statsmodels, scikit-learn, scipy
# Run in a notebook or Python environment.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import pickle

np.random.seed(42)
random.seed(42)
os.makedirs('/mnt/data', exist_ok=True)

# ---------- 1. Create dataset: 3 years hourly ----------
years = 3
periods = years * 365 * 24  # ~26280 hours
idx = pd.date_range(start="2017-01-01", periods=periods, freq='H')
t = np.arange(periods)

# Components:
trend = 0.0005 * t + 0.0000008 * (t**2)  # slight upward curvature
year_hours = 365.25 * 24
yearly = 12 * np.sin(2*np.pi*t/year_hours) + 2.5 * np.sin(4*np.pi*t/year_hours + 0.4)
week_hours = 7 * 24
weekly = 4.0 * np.sin(2*np.pi*t/week_hours - 0.15) + 1.0 * np.cos(4*np.pi*t/week_hours + 0.8)
daily = 2.2 * np.sin(2*np.pi*t/24 - 0.6) + 0.6 * np.cos(2*np.pi*t/12 + 0.3)

noise = np.random.normal(scale=1.1, size=periods) * (1 + 0.25*np.sin(2*np.pi*t/year_hours))
series = trend + yearly + weekly + daily + noise

# Inject two large irregular events (spikes/drops)
def inject_spike(arr, start_idx, magnitude, width=12):
    end = min(len(arr), start_idx + width)
    arr[start_idx:end] += magnitude * np.linspace(1.0, 0.2, end-start_idx)

inject_spike(series, int(0.4*periods), magnitude=35.0, width=24)
inject_spike(series, int(0.8*periods), magnitude=-28.0, width=36)
# some random medium spikes
for _ in range(5):
    s = np.random.randint(0, periods-12)
    inject_spike(series, s, magnitude=np.random.uniform(6, 14), width=np.random.randint(4,12))

df = pd.DataFrame({'ds': idx, 'y': series})
df.to_csv('/mnt/data/simulated_hourly_timeseries_3yr.csv', index=False)
print("Saved dataset to /mnt/data/simulated_hourly_timeseries_3yr.csv")

# ---------- 2. Rolling-origin CV helper ----------
def rolling_origin_cv(y, initial_train_hours, horizon_hours, step_hours):
    n = len(y)
    origin = initial_train_hours
    while origin + horizon_hours <= n:
        train_idx = np.arange(0, origin)
        test_idx = np.arange(origin, origin + horizon_hours)
        yield train_idx, test_idx
        origin += step_hours

# ---------- 3. Metrics ----------
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE': float(mape(y_true, y_pred))
    }

# ---------- 4. Forecast helpers (fast, with safe timeouts) ----------
def sarimax_forecast_safe(train_y, order, seasonal_order, trend='c', steps=168, maxiter=50):
    """
    Fit SARIMAX but set maxiter low so fits are faster. Return NaNs on failure.
    """
    try:
        model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order,
                        trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=maxiter)
        preds = res.forecast(steps=steps)
        return np.asarray(preds), res
    except Exception as e:
        # return NaNs to indicate bad candidate
        return np.full(steps, np.nan), None

# Fallback baseline: ExponentialSmoothing (Holt-Winters) for speed
def holtwinters_forecast(train_y, steps=168):
    try:
        model = ExponentialSmoothing(train_y, seasonal_periods=24, trend='add', seasonal='add', damped_trend=False)
        res = model.fit(optimized=True, use_brute=True)
        return res.forecast(steps), res
    except Exception:
        return np.full(steps, np.nan), None

# ---------- 5. Baseline evaluation ----------
y = df['y'].values
initial_train = 1 * 365 * 24  # 1 year initial training
horizon = 7 * 24  # 1-week horizon
step = 7 * 24  # move by a week for folds

# Baseline: try SARIMAX (1,1,1)(1,1,1,168) else fall back to HW
baseline_order = (1,1,1)
seasonal_period = 24 * 7
baseline_seasonal = (1,1,1, seasonal_period)

baseline_fold_metrics = []
fold_i = 0
for train_idx, test_idx in rolling_origin_cv(y, initial_train, horizon, step):
    fold_i += 1
    train_y = y[train_idx]
    true = y[test_idx]
    preds, _ = sarimax_forecast_safe(train_y, baseline_order, baseline_seasonal, trend='c', steps=horizon, maxiter=30)
    if np.isnan(preds).any():
        preds, _ = holtwinters_forecast(train_y, steps=horizon)
    if np.isnan(preds).any():
        # skip fold if both failed
        continue
    baseline_fold_metrics.append({**compute_metrics(true, preds), 'fold':fold_i})

baseline_cv_df = pd.DataFrame(baseline_fold_metrics)
baseline_summary = baseline_cv_df[['MAE','RMSE','MAPE']].mean().to_dict()
print("Baseline CV summary (averaged):", baseline_summary)

# ---------- 6. Bayesian-style hyperparameter search (GP surrogate) ----------
# Conservative budget to finish in reasonable time
n_init = 6
n_iter = 12  # total iterations
seasonal_period = 24 * 7

search_space = {
    'p': (0,3),
    'd': (0,1),
    'q': (0,3),
    'P': (0,2),
    'D': (0,1),
    'Q': (0,2),
    'trend': (0,1)  # 0 -> 'n', 1 -> 'c'
}

def sample_random_candidate():
    return {
        'p': int(np.random.randint(search_space['p'][0], search_space['p'][1]+1)),
        'd': int(np.random.randint(search_space['d'][0], search_space['d'][1]+1)),
        'q': int(np.random.randint(search_space['q'][0], search_space['q'][1]+1)),
        'P': int(np.random.randint(search_space['P'][0], search_space['P'][1]+1)),
        'D': int(np.random.randint(search_space['D'][0], search_space['D'][1]+1)),
        'Q': int(np.random.randint(search_space['Q'][0], search_space['Q'][1]+1)),
        'trend': int(np.random.choice([0,1]))
    }

def candidate_to_vector(c):
    return np.array([c['p'], c['d'], c['q'], c['P'], c['D'], c['Q'], c['trend']], dtype=float)

# Objective: average RMSE across a small number of folds
def evaluate_candidate(candidate, max_folds=4):
    order = (candidate['p'], candidate['d'], candidate['q'])
    seasonal = (candidate['P'], candidate['D'], candidate['Q'], seasonal_period)
    trend = 'c' if candidate['trend']==1 else 'n'
    rmses = []
    fold_count = 0
    for train_idx, test_idx in rolling_origin_cv(y, initial_train, horizon, step):
        fold_count += 1
        if fold_count > max_folds: break
        train_y = y[train_idx]
        true = y[test_idx]
        preds, _ = sarimax_forecast_safe(train_y, order, seasonal, trend=trend, steps=horizon, maxiter=25)
        if np.isnan(preds).any():
            # penalize failures heavily
            return np.inf
        rmses.append(np.sqrt(mean_squared_error(true, preds)))
    return float(np.mean(rmses)) if len(rmses)>0 else np.inf

# initial random evaluations
evaluated = []
X_list = []
y_list = []
for i in range(n_init):
    c = sample_random_candidate()
    val = evaluate_candidate(c)
    evaluated.append((c,val))
    X_list.append(candidate_to_vector(c))
    y_list.append(val)
    print(f"Init {i+1}/{n_init}: candidate={c}, cv_rmse={val}")

# Fit GP and run small acquisition loop
X = np.vstack(X_list)
y_vals = np.array(y_list).reshape(-1,1)
scaler = StandardScaler().fit(X)

kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2, random_state=42)
gp.fit(scaler.transform(X), y_vals.ravel())

def propose_next(gp, scaler, n_candidates=200, xi=0.01):
    # random candidate pool, choose by Expected Improvement (minimization)
    pool = [sample_random_candidate() for _ in range(n_candidates)]
    Xp = np.vstack([candidate_to_vector(p) for p in pool])
    Xps = scaler.transform(Xp)
    mu, std = gp.predict(Xps, return_std=True)
    mu = mu.ravel()
    std = std.ravel()
    best = gp.y_train_.min()
    imp = best - mu - xi
    Z = imp / (std + 1e-9)
    ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std==0.0] = 0.0
    idx = np.argmax(ei)
    return pool[idx], float(ei[idx])

for it in range(n_iter - n_init):
    cand, score = propose_next(gp, scaler, n_candidates=300)
    val = evaluate_candidate(cand)
    evaluated.append((cand,val))
    X_list.append(candidate_to_vector(cand))
    y_list.append(val)
    # refit
    X = np.vstack(X_list)
    y_vals = np.array(y_list).reshape(-1,1)
    scaler = StandardScaler().fit(X)
    gp.fit(scaler.transform(X), y_vals.ravel())
    print(f"Iter {it+1}/{n_iter-n_init}: cand={cand}, cv_rmse={val}, acq={score}")

# best candidate
best_idx = int(np.argmin(y_list))
best_candidate = evaluated[best_idx][0]
best_cv_rmse = evaluated[best_idx][1]
print("Best CV candidate:", best_candidate, "cv_rmse:", best_cv_rmse)

# ---------- 7. Final training & evaluation on holdout final year ----------
test_horizon = 365 * 24  # final 1 year test
train_full = y[:-test_horizon]
test_full = y[-test_horizon:]

# baseline trained on full train_full
base_preds, _ = sarimax_forecast_safe(train_full, baseline_order, baseline_seasonal, trend='c', steps=test_horizon, maxiter=80)
if np.isnan(base_preds).any():
    base_preds, _ = holtwinters_forecast(train_full, steps=test_horizon)

opt_order = (best_candidate['p'], best_candidate['d'], best_candidate['q'])
opt_seasonal = (best_candidate['P'], best_candidate['D'], best_candidate['Q'], seasonal_period)
opt_trend = 'c' if best_candidate['trend']==1 else 'n'
opt_preds, _ = sarimax_forecast_safe(train_full, opt_order, opt_seasonal, trend=opt_trend, steps=test_horizon, maxiter=80)

baseline_test_metrics = compute_metrics(test_full, base_preds)
opt_test_metrics = compute_metrics(test_full, opt_preds)

summary_df = pd.DataFrame([
    {'model':'baseline_sarimax', **baseline_test_metrics},
    {'model':'optimized_sarimax', **opt_test_metrics}
])
summary_df.to_csv('/mnt/data/model_comparison_metrics.csv', index=False)
with open('/mnt/data/best_candidate.pkl','wb') as f:
    pickle.dump({'best_candidate':best_candidate,'best_cv_rmse':best_cv_rmse}, f)

print("\nFinal test metrics saved to /mnt/data/model_comparison_metrics.csv")
print(summary_df)

# Save search history
history_df = pd.DataFrame([{**c, 'cv_rmse':v} for c,v in evaluated])
history_df.to_csv('/mnt/data/bayesian_search_history.csv', index=False)
print("Bayesian search history saved to /mnt/data/bayesian_search_history.csv")

# Quick plots (optional)
plt.figure(figsize=(10,3))
plt.plot(df['ds'].iloc[-2000:], df['y'].iloc[-2000:], label='series tail')
plt.title('Last 2000 hours sample of series')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,3))
test_index = df['ds'].iloc[-test_horizon:]
plt.plot(test_index, test_full, label='y_true', alpha=0.6)
plt.plot(test_index, base_preds, label='baseline_pred', alpha=0.8)
plt.plot(test_index, opt_preds, label='opt_pred', alpha=0.8)
plt.legend()
plt.title('Holdout year: true vs baseline vs optimized')
plt.tight_layout()
plt.show()
