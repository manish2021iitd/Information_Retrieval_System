import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced statistical plots
from scipy.stats import ttest_rel, wilcoxon, shapiro, probplot  # For statistical tests and Q-Q plot

# Average nDCG@k values for two models (replace with your actual evaluation results)
ndcg_tfidf = [0.689, 0.742, 0.755, 0.761, 0.767, 0.767, 0.765, 0.758, 0.755, 0.753]
ndcg_hybrid = [0.702, 0.745, 0.764, 0.763, 0.768, 0.767, 0.764, 0.763, 0.760, 0.755]

# Step 1: Compute paired differences (Hybrid - TF-IDF for each k)
differences = np.array(ndcg_hybrid) - np.array(ndcg_tfidf)

# Step 2: Plot histogram and Q-Q plot of differences, save the plot to file
plt.figure(figsize=(12, 5))  # Create a figure with specified size

plt.subplot(1, 2, 1)  # First subplot: Histogram
sns.histplot(differences, kde=True, bins=7, color='skyblue')  # Plot histogram with KDE
plt.axvline(0, color='red', linestyle='--')  # Add vertical line at 0 for reference
plt.title("Histogram of Differences (Hybrid - TF-IDF)")  # Title of histogram
plt.xlabel("Difference in nDCG@k")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)  # Second subplot: Q-Q plot
probplot(differences, dist="norm", plot=plt)  # Create Q-Q plot to assess normality
plt.title("Q-Q Plot of Differences")

plt.tight_layout()  # Prevent subplot overlap
plt.savefig("output/ndcg_diff_analysis.png", dpi=300)  # Save plot to file
plt.close()  # Close the figure to free memory in non-interactive environments

print("Plot saved as 'ndcg_diff_analysis.png' with 300 DPI.")  # Notify user

# Step 3: Perform Shapiro-Wilk test to check for normality of differences
shapiro_stat, shapiro_p = shapiro(differences)
print("Shapiro-Wilk Normality Test:")
print(f"  W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")  # Print test result

# Step 4: Choose statistical test based on normality result
if shapiro_p > 0.05:
    print("Differences appear normally distributed (p > 0.05)")
    # Use paired t-test (one-tailed) if normal
    t_stat, p_val_two_tailed = ttest_rel(ndcg_hybrid, ndcg_tfidf)  # Paired t-test
    p_val_one_tailed = p_val_two_tailed / 2  # Convert to one-tailed
    print("\n Paired t-test:")
    print(f"  t-statistic = {t_stat:.4f}, one-tailed p-value = {p_val_one_tailed:.4f}")
    if p_val_one_tailed < 0.05:
        print("Result: HYBRID significantly outperforms TF-IDF.")  # Significant result
    else:
        print("Result: No significant difference.")  # Not significant
else:
    print("Differences not normally distributed (p ≤ 0.05)")
    # Use Wilcoxon signed-rank test (non-parametric alternative)
    w_stat, w_p = wilcoxon(differences, alternative='greater')  # One-tailed test
    print("\n Wilcoxon signed-rank test:")
    print(f"  W-statistic = {w_stat:.4f}, one-tailed p-value = {w_p:.4f}")
    if w_p < 0.05:
        print(" Result: HYBRID significantly outperforms TF-IDF.")  # Significant result
    else:
        print(" Result: No significant difference.")  # Not significant
