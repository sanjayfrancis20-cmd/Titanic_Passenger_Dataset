import os
import json
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

os.makedirs("images", exist_ok=True)

# 1. Load Data
df = pd.read_csv("titanic.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Summary stats
info_str = []
df.info(buf=open("data_info.txt", "w"))
with open("data_info.txt", "r") as f:
    data_info = f.read()

describe_num = df.describe()
describe_cat = df.describe(include=['object', 'category'])
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df)) * 100
missing_df = pd.DataFrame({"Missing Count": missing_summary, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])

# Visual 1: Missing Values Heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Visual 1: Missing Value Matrix (Yellow = Missing)", fontsize=14, weight='bold', pad=12)
plt.xlabel("Features")
plt.ylabel("Observations (Passengers)")
plt.tight_layout()
plt.savefig("images/01_missing_values_heatmap.png", dpi=300)
plt.close()

# Visual 2: Univariate Distributions (Age & Fare)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['age'].dropna(), kde=True, ax=axes[0], color='#2b5c8f', bins=30)
axes[0].set_title(f"Age Distribution (Skew: {df['age'].skew():.2f})", fontsize=13, weight='bold')
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Passenger Count")
axes[0].axvline(df['age'].median(), color='red', linestyle='--', label=f"Median: {df['age'].median():.1f}")
axes[0].axvline(df['age'].mean(), color='orange', linestyle='-', label=f"Mean: {df['age'].mean():.1f}")
axes[0].legend()

sns.histplot(df['fare'], kde=True, ax=axes[1], color='#d95f02', bins=40)
axes[1].set_title(f"Fare Distribution (Skew: {df['fare'].skew():.2f})", fontsize=13, weight='bold')
axes[1].set_xlabel("Fare (British Pounds)")
axes[1].set_ylabel("Passenger Count")
axes[1].axvline(df['fare'].median(), color='red', linestyle='--', label=f"Median: {df['fare'].median():.1f}")
axes[1].axvline(df['fare'].mean(), color='orange', linestyle='-', label=f"Mean: {df['fare'].mean():.1f}")
axes[1].legend()

plt.suptitle("Visual 2: Univariate Numerical Analysis - Age & Fare Distributions", fontsize=15, weight='bold', y=1.03)
plt.tight_layout()
plt.savefig("images/02_univariate_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 3: Categorical Distributions
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
sns.countplot(data=df, x='sex', hue='sex', palette='Set2', legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Passenger Gender Distribution", weight='bold')
axes[0, 0].set_xlabel("Gender")
axes[0, 0].set_ylabel("Count")
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)

sns.countplot(data=df, x='pclass', hue='pclass', palette='Blues_r', legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Passenger Ticket Class Distribution", weight='bold')
axes[0, 1].set_xlabel("Ticket Class (1st, 2nd, 3rd)")
axes[0, 1].set_ylabel("Count")
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)

sns.countplot(data=df, x='embark_town', hue='embark_town', palette='Purples_r', legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Embarkation Town Distribution", weight='bold')
axes[1, 0].set_xlabel("Port of Embarkation")
axes[1, 0].set_ylabel("Count")
for p in axes[1, 0].patches:
    if not np.isnan(p.get_height()):
        axes[1, 0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)

sns.countplot(data=df, x='alone', hue='alone', palette='Pastel1', legend=False, ax=axes[1, 1])
axes[1, 1].set_title("Solo Traveler vs Family Distribution", weight='bold')
axes[1, 1].set_xlabel("Traveling Alone (True / False)")
axes[1, 1].set_ylabel("Count")
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=10)

plt.suptitle("Visual 3: Univariate Categorical Analysis", fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig("images/03_categorical_counts.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 4: Bivariate Survival Rates
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.barplot(data=df, x='sex', y='survived', hue='sex', palette='Set2', legend=False, ax=axes[0], errorbar=None)
axes[0].set_title("Survival Rate by Gender", weight='bold')
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height()*100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

sns.barplot(data=df, x='pclass', y='survived', hue='pclass', palette='Blues_r', legend=False, ax=axes[1], errorbar=None)
axes[1].set_title("Survival Rate by Passenger Class", weight='bold')
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height()*100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

sns.barplot(data=df, x='embark_town', y='survived', hue='embark_town', palette='Purples_r', legend=False, ax=axes[2], errorbar=None)
axes[2].set_title("Survival Rate by Embarkation Town", weight='bold')
axes[2].set_ylabel("Survival Rate")
axes[2].set_ylim(0, 1)
for p in axes[2].patches:
    if not np.isnan(p.get_height()):
        axes[2].annotate(f"{p.get_height()*100:.1f}%",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

plt.suptitle("Visual 4: Bivariate Analysis - Survival Rates Across Demographics", fontsize=15, weight='bold', y=1.03)
plt.tight_layout()
plt.savefig("images/04_survival_by_demographics.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 5: Boxplots for Outlier Detection (Age & Fare by Class and Survival)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(data=df, x='pclass', y='age', hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'}, ax=axes[0])
axes[0].set_title("Age Distribution by Class and Survival Status", weight='bold')
axes[0].set_xlabel("Ticket Class (1st, 2nd, 3rd)")
axes[0].set_ylabel("Age (years)")
axes[0].legend(title='Survived', labels=['Did not survive', 'Survived'])

sns.boxplot(data=df, x='pclass', y='fare', hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'}, ax=axes[1])
axes[1].set_title("Fare Distribution by Class and Survival (Outliers Visible)", weight='bold')
axes[1].set_xlabel("Ticket Class (1st, 2nd, 3rd)")
axes[1].set_ylabel("Fare")
axes[1].set_yscale('log')
axes[1].set_ylabel("Fare (Log Scale - GBP)")
axes[1].legend(title='Survived', labels=['Did not survive', 'Survived'])

plt.suptitle("Visual 5: Outlier Detection and Variance Analysis via Boxplots", fontsize=15, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig("images/05_boxplots_age_fare_by_class.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 6: Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": .8}, mask=mask)
plt.title("Visual 6: Correlation Matrix Heatmap of Numerical Features", fontsize=14, weight='bold', pad=12)
plt.tight_layout()
plt.savefig("images/06_correlation_heatmap.png", dpi=300)
plt.close()

# Visual 7: Pairplot
pairplot_features = ['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch']
pairplot_df = df[pairplot_features].dropna()
g = sns.pairplot(pairplot_df, hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'},
                 markers=["o", "s"], diag_kind="kde", corner=True, height=2.2)
g.fig.suptitle("Visual 7: Seaborn Pairplot Across Key Numerical Features (Hue = Survived)", y=1.02, fontsize=14, weight='bold')
plt.savefig("images/07_pairplot_titanic.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 8: Scatterplot (Age vs Fare colored by Survived and shaped by Sex)
plt.figure(figsize=(11, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='survived', style='sex',
                palette={0: '#d95f02', 1: '#2b5c8f'}, alpha=0.75, s=60)
plt.title("Visual 8: Multivariate Scatterplot - Age vs Fare by Survival and Gender", fontsize=14, weight='bold', pad=12)
plt.xlabel("Age (years)")
plt.ylabel("Fare (British Pounds)")
plt.ylim(-5, 300) # trim extreme 500+ ticket outlier for visual clarity
plt.legend(title='Status & Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("images/08_scatterplot_age_vs_fare.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 9: Handling Skewed Data (Raw vs Log Transformation of Fare)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
raw_fare_skew = df['fare'].skew()
log_fare = np.log1p(df['fare'])
log_fare_skew = log_fare.skew()

sns.histplot(df['fare'], kde=True, ax=axes[0], color='#d95f02', bins=35)
axes[0].set_title(f"Original Fare (Positive Skew = {raw_fare_skew:.2f})", weight='bold')
axes[0].set_xlabel("Fare")
axes[0].set_ylabel("Frequency")

sns.histplot(log_fare, kde=True, ax=axes[1], color='#2ca02c', bins=30)
axes[1].set_title(f"Log-Transformed Fare: log1p(Fare) (Skew = {log_fare_skew:.2f})", weight='bold')
axes[1].set_xlabel("Log(Fare + 1)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Visual 9: Demonstration of Skewness Handling via Log Transformation", fontsize=15, weight='bold', y=1.03)
plt.tight_layout()
plt.savefig("images/09_skewness_transformation.png", dpi=300, bbox_inches='tight')
plt.close()

# Visual 10: Multivariate Catplot (Survival by Class, Sex, and Embarkation Town)
cat_g = sns.catplot(
    data=df, x='pclass', y='survived', hue='sex', col='embark_town',
    kind='bar', palette='Set2', errorbar=None, height=4.5, aspect=0.9
)
cat_g.fig.subplots_adjust(top=0.82)
cat_g.fig.suptitle("Visual 10: Multi-panel Catplot: Survival Rate by Class, Gender & Port of Embarkation", fontsize=13, weight='bold')
cat_g.set_axis_labels("Passenger Class", "Survival Rate")
cat_g.set_titles("Embark Town: {col_name}")
for ax in cat_g.axes.flat:
    for p in ax.patches:
        if not np.isnan(p.get_height()) and p.get_height() > 0:
            ax.annotate(f"{p.get_height()*100:.0f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontsize=9, weight='bold')
plt.savefig("images/10_multivariate_catplot.png", dpi=300, bbox_inches='tight')
plt.close()

print("All 10 figures generated successfully!")
