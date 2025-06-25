import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from pathlib import Path

from src.envs.dp import DPTable

table = DPTable.load(Path("C:\\Users\\rober\\CSE3000\\data\\dp_cache\\dp_table_a15_e15_tc0p0001_dataf8695d276f0e966c.npz"))

V = table.value_table
Q_min = table.q_min_table
importance = (V - Q_min).flatten()
importance = importance[importance > 1e-6]

n = importance.size
lambda_param = (n - 1) / np.sum(importance)

x_vals = np.linspace(importance.min(), importance.max(), 400)
y_vals = lambda_param * np.exp(-lambda_param * x_vals)

# --- 2. P-P Plot for the Initial Exponential Fit ---
print("Generating P-P plot for the initial exponential fit...")
sorted_importance = np.sort(importance)
empirical_cdf = np.arange(1, n + 1) / n
theoretical_cdf_exp = 1 - np.exp(-lambda_param * sorted_importance)

plt.figure(figsize=(8, 8))
plt.scatter(empirical_cdf, theoretical_cdf_exp, alpha=0.3, s=5, label='Data')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit (y=x)')

plt.title('P-P Plot: Exponential Fit vs. Empirical Data')
plt.xlabel('Empirical Cumulative Probability')
plt.ylabel('Theoretical Cumulative Probability (Exponential)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("pp_plot_exponential.png", format="png")
# plt.show()


# --- 3. Fit More Models and Compare on Histogram ---
print("Generating comparison histogram with all fits...")

# METHOD A: Robust lambda estimate using the median
median_importance = np.median(importance)
lambda_robust = np.log(2) / median_importance

# METHOD B: Fit a more flexible Weibull distribution
shape_w, loc_w, scale_w = stats.weibull_min.fit(importance, floc=0)

# Plot the comparison
plt.figure(figsize=(12, 8))
plt.hist(importance, bins=200, density=True, label='Importance Histogram', alpha=0.6)
hist_max_y = plt.gca().get_ylim()[1] # For readable plot limits

# Plot MLE exponential fit
lambda_mle = 1 / np.mean(importance)
y_mle = lambda_mle * np.exp(-lambda_mle * x_vals)
plt.plot(x_vals, y_mle, 'r-', lw=2, label=f'Exponential Fit (MLE, $\\lambda$={lambda_mle:.2f})')

# Plot Robust exponential fit
y_robust = lambda_robust * np.exp(-lambda_robust * x_vals)
plt.plot(x_vals, y_robust, 'g--', lw=2, label=f'Exponential Fit (Median, $\\lambda$={lambda_robust:.2f})')

# Plot Weibull distribution fit
y_weibull = stats.weibull_min.pdf(x_vals, shape_w, loc=loc_w, scale=scale_w)
plt.plot(x_vals, y_weibull, 'b:', lw=3, label=f'Weibull Fit (shape={shape_w:.2f})')

plt.xlabel('Importance')
plt.ylabel('Density')
plt.title('Comparison of Distribution Fits to Importance Data')
plt.legend()
plt.grid(True)
plt.ylim(top=hist_max_y * 1.1)
plt.savefig("histogram_comparison_fit.svg", format="svg")
# plt.show()


# --- 4. P-P Plot for the Robust (Median) Exponential Fit ---
print("Generating P-P plot for the robust exponential fit...")
theoretical_cdf_robust = 1 - np.exp(-lambda_robust * sorted_importance)

plt.figure(figsize=(8, 8))
plt.scatter(empirical_cdf, theoretical_cdf_robust, alpha=0.3, s=5, label='Data (Median Fit)')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit (y=x)')
plt.title('P-P Plot: Robust Exponential Fit vs. Empirical Data')
plt.xlabel('Empirical Cumulative Probability')
plt.ylabel('Theoretical Cumulative Probability (Robust Exp.)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("pp_plot_robust.png", format="png")


# --- 5. P-P Plot for the Weibull Fit ---
print("Generating P-P plot for the Weibull fit...")
theoretical_cdf_weibull = stats.weibull_min.cdf(sorted_importance, shape_w, loc=loc_w, scale=scale_w)

plt.figure(figsize=(8, 8))
plt.scatter(empirical_cdf, theoretical_cdf_weibull, alpha=0.3, s=5, label='Data (Weibull Fit)')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Fit (y=x)')
plt.title('P-P Plot: Weibull Fit vs. Empirical Data')
plt.xlabel('Empirical Cumulative Probability')
plt.ylabel('Theoretical Cumulative Probability (Weibull)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.savefig("pp_plot_weibull.png", format="png")

# --- 6. Quantitative Comparison of Fits ---

print("\n--- Model Fit Error Comparison ---")

# Calculate Theoretical CDFs if not already done
# Note: sorted_importance and empirical_cdf should be defined from earlier
lambda_mle = 1 / np.mean(importance)
theoretical_cdf_mle = 1 - np.exp(-lambda_mle * sorted_importance)

lambda_robust = np.log(2) / np.median(importance)
theoretical_cdf_robust = 1 - np.exp(-lambda_robust * sorted_importance)

shape_w, loc_w, scale_w = stats.weibull_min.fit(importance, floc=0)
theoretical_cdf_weibull = stats.weibull_min.cdf(sorted_importance, shape_w, loc=loc_w, scale=scale_w)

# Calculate Mean Absolute Error for each fit
mae_mle = np.mean(np.abs(empirical_cdf - theoretical_cdf_mle))
mae_robust = np.mean(np.abs(empirical_cdf - theoretical_cdf_robust))
mae_weibull = np.mean(np.abs(empirical_cdf - theoretical_cdf_weibull))

print(f"Standard Exponential (MLE) Fit MAE: {mae_mle:.6f}")
print(f"Robust Exponential (Median) Fit MAE: {mae_robust:.6f}")
print(f"Weibull Fit MAE:                     {mae_weibull:.6f}")

# Find and announce the best model numerically
errors = {'Standard Exponential': mae_mle, 'Robust Exponential': mae_robust, 'Weibull': mae_weibull}
best_model = min(errors, key=errors.get)

print(f"\nConclusion: The model with the lowest error is the '{best_model}' fit.")