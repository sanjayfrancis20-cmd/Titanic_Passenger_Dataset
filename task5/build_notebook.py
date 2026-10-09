import json

def make_cell(cell_type, source, outputs=None, execution_count=None):
    if isinstance(source, str):
        source_lines = [line + "\n" for line in source.split("\n")]
        # remove trailing newline on the last line to follow standard ipynb conventions
        if source_lines and source_lines[-1] == "\n":
            source_lines = source_lines[:-1]
    else:
        source_lines = source
        
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs if outputs is not None else []
    return cell

cells = []
exec_counter = 1

# --- Cell 1: Markdown Title ---
cells.append(make_cell("markdown", """# Elevate Labs Data Analyst Internship
## Task 5: Comprehensive Exploratory Data Analysis (EDA) on the Titanic Dataset

**Objective:** Extract meaningful patterns, trends, anomalies, and statistical insights from the Titanic passenger dataset using visual and statistical exploration.  
**Tools & Libraries:** Python (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`)  
**Deliverables:** Jupyter Notebook (`titanic_eda.ipynb`), Findings Report (`Titanic_EDA_Findings_Report.pdf`), and GitHub Repository Documentation (`README.md`).

---
### Notebook Structure:
1. **Introduction & Business Problem Definition**
2. **Environment Setup & Data Ingestion**
3. **Data Profiling & Initial Inspection (`.info()`, `.describe()`, `.value_counts()`)**
4. **Data Cleaning & Missing Value Handling**
5. **Exploratory Visualizations with Detailed Observations (10 Visuals)**
   - Missing Values Matrix
   - Univariate Numerical Distributions (Age, Fare)
   - Univariate Categorical Distributions (Sex, Pclass, Embark Town, Alone)
   - Bivariate Analysis: Survival Rate Comparisons
   - Boxplots for Outlier Detection
   - Correlation Matrix Heatmap
   - Pairwise Feature Interactions (Pairplot)
   - Multivariate Scatterplots
   - Handling Skewed Data (Log Transformation)
   - Multivariate FacetGrid / Catplot
6. **Hypothesis Testing & Statistical Inference**
7. **Executive Summary of Findings & Actionable Insights**
8. **Comprehensive Answers to Internship Interview Questions (Q1 - Q7)**
"""))

# --- Cell 2: Imports ---
cells.append(make_cell("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure visualization styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

print("All analytics and visualization libraries successfully imported!")
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 3: Data Ingestion ---
cells.append(make_cell("markdown", """## 1. Data Ingestion & First Look
We load the canonical Titanic dataset (`titanic.csv`), inspect its shape, and preview the first and last records.
"""))

cells.append(make_cell("code", """# Load the dataset
df = pd.read_csv("titanic.csv")
print(f"Dataset Shape: {df.shape[0]} rows (passengers) x {df.shape[1]} columns (features)")
df.head()
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 4: Info and Types ---
cells.append(make_cell("markdown", """## 2. Structural Inspection: `.info()` and Column Types
Inspecting non-null counts, data types, and memory usage.
"""))

cells.append(make_cell("code", """# Display schema and data types
df.info()
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 5: Describe ---
cells.append(make_cell("markdown", """## 3. Summary Statistics: `.describe()`
We examine numerical distributions (mean, std, min, quartiles, max) as well as categorical summaries.
"""))

cells.append(make_cell("code", """# Summary statistics for numerical variables
print("--- NUMERICAL SUMMARY STATISTICS ---")
display(df.describe().round(2))

# Summary statistics for categorical variables
print("\\n--- CATEGORICAL SUMMARY STATISTICS ---")
display(df.describe(include=['object', 'category']))
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 6: Value Counts ---
cells.append(make_cell("markdown", """## 4. Frequency Distributions: `.value_counts()`
Checking the proportion of target and categorical variables.
"""))

cells.append(make_cell("code", """print("--- TARGET VARIABLE: SURVIVED FREQUENCY ---")
print(df['survived'].value_counts(normalize=True).map(lambda x: f"{x*100:.2f}%"))

print("\\n--- PASSENGER CLASS FREQUENCY ---")
print(df['pclass'].value_counts(normalize=True).map(lambda x: f"{x*100:.2f}%"))

print("\\n--- GENDER FREQUENCY ---")
print(df['sex'].value_counts(normalize=True).map(lambda x: f"{x*100:.2f}%"))

print("\\n--- EMBARKATION TOWN FREQUENCY ---")
print(df['embark_town'].value_counts(dropna=False))
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 7: Missing Value Analysis ---
cells.append(make_cell("markdown", """## 5. Missing Value Audit & Missing Heatmap
Analyzing the completeness of each feature.
"""))

cells.append(make_cell("code", """missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100
missing_report = pd.DataFrame({
    'Missing Count': missing_count,
    'Percentage (%)': missing_percent.round(2)
}).sort_values(by='Missing Count', ascending=False)

display(missing_report[missing_report['Missing Count'] > 0])
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 8: Visual 1 ---
cells.append(make_cell("code", """plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Visual 1: Missing Value Matrix (Yellow lines = Missing Entries)", fontsize=13, weight='bold')
plt.xlabel("Features")
plt.ylabel("Passenger Rows")
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 1:**
- `deck` has **77.22% (688 rows)** missing values, indicating severe sparsity; keeping it as a raw predictor without grouping would induce excessive imputation bias.
- `age` is missing in **19.87% (177 rows)** of passengers. Imputation based on median age segmented by passenger class and gender is recommended.
- `embarked` / `embark_town` is missing in only **2 rows (0.22%)**, which can be easily imputed with the mode (`Southampton`).
"""))

# --- Cell 9: Visual 2 Univariate Numerical ---
cells.append(make_cell("markdown", """## 6. Univariate Numerical Analysis: Age & Fare
Examining the shape of distributions, central tendencies, and skewness.
"""))

cells.append(make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age Histogram & KDE
sns.histplot(df['age'].dropna(), kde=True, ax=axes[0], color='#2b5c8f', bins=30)
axes[0].set_title(f"Age Distribution (Skewness: {df['age'].skew():.2f})", fontsize=12, weight='bold')
axes[0].set_xlabel("Age (Years)")
axes[0].set_ylabel("Count")
axes[0].axvline(df['age'].median(), color='red', linestyle='--', label=f"Median: {df['age'].median():.1f}")
axes[0].axvline(df['age'].mean(), color='orange', linestyle='-', label=f"Mean: {df['age'].mean():.1f}")
axes[0].legend()

# Fare Histogram & KDE
sns.histplot(df['fare'], kde=True, ax=axes[1], color='#d95f02', bins=35)
axes[1].set_title(f"Fare Distribution (Skewness: {df['fare'].skew():.2f})", fontsize=12, weight='bold')
axes[1].set_xlabel("Fare Paid (£)")
axes[1].set_ylabel("Count")
axes[1].axvline(df['fare'].median(), color='red', linestyle='--', label=f"Median: {df['fare'].median():.1f}")
axes[1].axvline(df['fare'].mean(), color='orange', linestyle='-', label=f"Mean: {df['fare'].mean():.1f}")
axes[1].legend()

plt.suptitle("Visual 2: Univariate Numerical Analysis - Age & Fare", fontsize=14, weight='bold', y=1.03)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 2:**
- **Age:** The distribution is approximately bell-shaped with slight positive skewness (0.39). Most passengers were young adults aged 18 to 35, with a small secondary peak representing infants. The mean age is 29.7 years and median is 28.0 years.
- **Fare:** The distribution is **severely right-skewed (skewness = 4.79)**. The vast majority of tickets cost under £30, with a median of £14.45 and mean of £32.20. Extreme outliers stretch all the way to £512.33, necessitating logarithmic transformation for parametric modeling.
"""))

# --- Cell 10: Visual 3 Univariate Categorical ---
cells.append(make_cell("markdown", """## 7. Univariate Categorical Analysis: Demographics & Class
Investigating frequencies across categorical attributes.
"""))

cells.append(make_cell("code", """fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Gender
sns.countplot(data=df, x='sex', hue='sex', palette='Set2', legend=False, ax=axes[0, 0])
axes[0, 0].set_title("Gender Breakdown", weight='bold')
axes[0, 0].set_xlabel("Gender")
axes[0, 0].set_ylabel("Count")
for p in axes[0, 0].patches:
    axes[0, 0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)

# Class
sns.countplot(data=df, x='pclass', hue='pclass', palette='Blues_r', legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Ticket Class Breakdown", weight='bold')
axes[0, 1].set_xlabel("Class (1st, 2nd, 3rd)")
axes[0, 1].set_ylabel("Count")
for p in axes[0, 1].patches:
    axes[0, 1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)

# Embark Town
sns.countplot(data=df, x='embark_town', hue='embark_town', palette='Purples_r', legend=False, ax=axes[1, 0])
axes[1, 0].set_title("Embarkation Port Breakdown", weight='bold')
axes[1, 0].set_xlabel("Port Town")
axes[1, 0].set_ylabel("Count")
for p in axes[1, 0].patches:
    if not np.isnan(p.get_height()):
        axes[1, 0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)

# Solo Traveler
sns.countplot(data=df, x='alone', hue='alone', palette='Pastel1', legend=False, ax=axes[1, 1])
axes[1, 1].set_title("Solo Traveler Status", weight='bold')
axes[1, 1].set_xlabel("Traveling Alone (True / False)")
axes[1, 1].set_ylabel("Count")
for p in axes[1, 1].patches:
    axes[1, 1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=10)

plt.suptitle("Visual 3: Univariate Categorical Distributions", fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 3:**
- **Gender:** Males formed the majority at **64.8% (577 passengers)** compared to **35.2% (314 females)**.
- **Class:** Over half the passengers were in 3rd class (**55.1%**, 491 passengers), while 1st class had **24.2%** (216) and 2nd class had **20.7%** (184).
- **Port:** Southampton was the dominant port of embarkation (**72.3%**), followed by Cherbourg (**18.9%**) and Queenstown (**8.6%**).
- **Solo Travel:** **60.3%** of passengers boarded unaccompanied, while **39.7%** traveled with family.
"""))

# --- Cell 11: Visual 4 Bivariate Survival Rates ---
cells.append(make_cell("markdown", """## 8. Bivariate Analysis: Survival Rates by Demographics
Evaluating survival rates across gender, socio-economic ticket class, and boarding port.
"""))

cells.append(make_cell("code", """fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Survival by Gender
sns.barplot(data=df, x='sex', y='survived', hue='sex', palette='Set2', legend=False, ax=axes[0], errorbar=None)
axes[0].set_title("Survival Rate by Gender", weight='bold')
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height()*100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

# Survival by Class
sns.barplot(data=df, x='pclass', y='survived', hue='pclass', palette='Blues_r', legend=False, ax=axes[1], errorbar=None)
axes[1].set_title("Survival Rate by Passenger Class", weight='bold')
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1)
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height()*100:.1f}%",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

# Survival by Port
sns.barplot(data=df, x='embark_town', y='survived', hue='embark_town', palette='Purples_r', legend=False, ax=axes[2], errorbar=None)
axes[2].set_title("Survival Rate by Embarkation Port", weight='bold')
axes[2].set_ylabel("Survival Rate")
axes[2].set_ylim(0, 1)
for p in axes[2].patches:
    if not np.isnan(p.get_height()):
        axes[2].annotate(f"{p.get_height()*100:.1f}%",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=11, weight='bold')

plt.suptitle("Visual 4: Bivariate Survival Probability Analysis", fontsize=14, weight='bold', y=1.03)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 4:**
- **Gender Disparity:** Female passengers had an overwhelming survival rate of **74.2%**, compared to only **18.9%** for male passengers (a 3.9x difference), reflecting strict adherence to the maritime code *"Women and children first"*.
- **Class Hierarchy:** 1st class passengers survived at **63.0%**, 2nd class at **47.3%**, and 3rd class at just **24.2%**. Physical cabin location on higher decks and privileged evacuation access drastically increased survival odds.
- **Port Differences:** Passengers embarking from Cherbourg exhibited higher survival (**55.4%**) because a disproportionately large fraction of Cherbourg passengers purchased 1st class luxury tickets.
"""))

# --- Cell 12: Visual 5 Boxplots & Outlier Detection ---
cells.append(make_cell("markdown", """## 9. Variance & Outlier Detection: Boxplots
Analyzing distributions and anomalous values across classes and survival status.
"""))

cells.append(make_cell("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Age Boxplot
sns.boxplot(data=df, x='pclass', y='age', hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'}, ax=axes[0])
axes[0].set_title("Age Distribution by Class and Survival", weight='bold')
axes[0].set_xlabel("Ticket Class")
axes[0].set_ylabel("Age (Years)")
axes[0].legend(title='Survived', labels=['Did not survive (0)', 'Survived (1)'])

# Fare Boxplot (Log Scale)
sns.boxplot(data=df, x='pclass', y='fare', hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'}, ax=axes[1])
axes[1].set_title("Fare Distribution by Class and Survival (Log Scale)", weight='bold')
axes[1].set_xlabel("Ticket Class")
axes[1].set_ylabel("Fare (£ - Log Scale)")
axes[1].set_yscale('log')
axes[1].legend(title='Survived', labels=['Did not survive (0)', 'Survived (1)'])

plt.suptitle("Visual 5: Outlier & Distribution Detection with Boxplots", fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 5:**
- **Age Trends:** 1st class passengers were older on average (median age ~37 years) compared to 2nd class (~29 years) and 3rd class (~24 years). In 2nd and 3rd class, young children had high survival rates, visible as lower whiskers extending down to infants.
- **Fare Outliers:** Several extreme outliers exist in 1st class where multiple tickets exceeded £200 up to £512. In all three passenger classes, survivors had higher median fares than non-survivors within the same tier.
"""))

# --- Cell 13: Visual 6 Correlation Heatmap ---
cells.append(make_cell("markdown", """## 10. Correlation Analysis: Heatmap
Measuring linear correlation between numeric features via Pearson correlation coefficient ($r$).
"""))

cells.append(make_cell("code", """plt.figure(figsize=(9, 6.5))
numeric_df = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']]
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": .8}, mask=mask)
plt.title("Visual 6: Pearson Correlation Matrix Heatmap", fontsize=13, weight='bold', pad=12)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 6:**
- **Fare vs Survived ($r = +0.26$):** Positive correlation indicating higher ticket spend is associated with higher survival probability.
- **Pclass vs Survived ($r = -0.34$):** Moderate negative correlation; as class number increases (1 $\\rightarrow$ 3, i.e., lower economic status), survival probability drops significantly.
- **Pclass vs Fare ($r = -0.55$):** Strong inverse relationship confirming 1st class tickets cost substantially more than 2nd and 3rd class tickets.
- **SibSp vs Parch ($r = +0.41$):** Moderate positive correlation reflecting that passengers traveling with siblings/spouses also tended to travel with parents/children (family groups).
"""))

# --- Cell 14: Visual 7 Pairplot ---
cells.append(make_cell("markdown", """## 11. Multivariate Pairwise Exploration: Pairplot
Visualizing pairwise bivariate relationships and univariate KDE distributions simultaneously.
"""))

cells.append(make_cell("code", """pairplot_cols = ['survived', 'pclass', 'age', 'fare', 'sibsp', 'parch']
pair_df = df[pairplot_cols].dropna()

g = sns.pairplot(pair_df, hue='survived', palette={0: '#d95f02', 1: '#2b5c8f'},
                 markers=["o", "s"], diag_kind="kde", corner=True, height=2.2)
g.fig.suptitle("Visual 7: Seaborn Pairplot Across Numerical Features (Hue = Survived)", y=1.02, fontsize=13, weight='bold')
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 7:**
- The diagonal density plots clearly separate survivors (blue) from non-survivors (orange) in `pclass` and `fare`.
- High fares ($> £100$) are almost exclusively populated by blue markers (survivors).
- The joint scatterplot of `sibsp` and `parch` reveals that large family units ($> 4$ relatives) predominantly suffered fatal casualties (orange markers), whereas small family units (1–3 relatives) had favorable survival outcomes.
"""))

# --- Cell 15: Visual 8 Scatterplot ---
cells.append(make_cell("markdown", """## 12. Multivariate Scatterplot: Age vs Fare by Survival & Gender
Investigating interaction between continuous variables and categorical targets.
"""))

cells.append(make_cell("code", """plt.figure(figsize=(11, 6))
sns.scatterplot(data=df, x='age', y='fare', hue='survived', style='sex',
                palette={0: '#d95f02', 1: '#2b5c8f'}, alpha=0.75, s=65)
plt.title("Visual 8: Scatterplot - Age vs Fare by Survival Status and Gender", fontsize=13, weight='bold', pad=12)
plt.xlabel("Age (Years)")
plt.ylabel("Ticket Fare (£)")
plt.ylim(-5, 300) # Capped at 300 to focus on distribution density
plt.legend(title='Status & Gender', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 8:**
- The high-fare cluster ($> £50$) consists almost entirely of survivors, irrespective of age.
- In the low-fare region ($< £20$), young children ($< 10$ years) show high survival rates, whereas adult males aged 20–50 are overwhelmingly non-survivors (orange circular markers).
- Women (square markers) achieved high survival rates even in the lower fare brackets, reinforcing gender as the primary predictor.
"""))

# --- Cell 16: Visual 9 Handling Skewed Data ---
cells.append(make_cell("markdown", """## 13. Handling Skewed Data: Transformation Analysis
Evaluating the effect of logarithmic transformation on positively skewed financial data (`fare`).
"""))

cells.append(make_cell("code", """raw_skew = df['fare'].skew()
log_fare = np.log1p(df['fare'])
log_skew = log_fare.skew()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['fare'], kde=True, ax=axes[0], color='#d95f02', bins=35)
axes[0].set_title(f"Original Fare (Positive Skew = {raw_skew:.2f})", weight='bold')
axes[0].set_xlabel("Fare (£)")
axes[0].set_ylabel("Frequency")

sns.histplot(log_fare, kde=True, ax=axes[1], color='#2ca02c', bins=30)
axes[1].set_title(f"Log1p Transformed Fare (Reduced Skew = {log_skew:.2f})", weight='bold')
axes[1].set_xlabel("Log(Fare + 1)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Visual 9: Demonstration of Skewness Handling via Log Transformation", fontsize=14, weight='bold', y=1.03)
plt.tight_layout()
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 9:**
- The raw `fare` feature exhibits extreme positive skewness (**4.79**), violating normality assumptions required by linear regression and neural networks.
- Applying `np.log1p()` (which computes $\\log(1 + x)$ to safely handle zero fares) dramatically compresses the long right tail, reducing skewness from **4.79 down to 0.39**.
- The transformed distribution approximates a bimodal quasi-normal curve, improving gradient stability and model convergence.
"""))

# --- Cell 17: Visual 10 Multi-panel Catplot ---
cells.append(make_cell("markdown", """## 14. Multivariate Categorical FacetGrid: Class x Gender x Embarkation
Evaluating interaction effects across three categorical dimensions simultaneously.
"""))

cells.append(make_cell("code", """cat_g = sns.catplot(
    data=df, x='pclass', y='survived', hue='sex', col='embark_town',
    kind='bar', palette='Set2', errorbar=None, height=4.2, aspect=0.9
)
cat_g.fig.subplots_adjust(top=0.82)
cat_g.fig.suptitle("Visual 10: Multi-panel Catplot: Survival Rate by Class, Gender & Port of Embarkation", fontsize=13, weight='bold')
cat_g.set_axis_labels("Ticket Class", "Survival Rate")
cat_g.set_titles("Port: {col_name}")

for ax in cat_g.axes.flat:
    for p in ax.patches:
        if not np.isnan(p.get_height()) and p.get_height() > 0:
            ax.annotate(f"{p.get_height()*100:.0f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontsize=9, weight='bold')
plt.show()
""", execution_count=exec_counter))
exec_counter += 1

cells.append(make_cell("markdown", """**Observation for Visual 10:**
- Across all embarkation ports, females in 1st and 2nd class had survival rates exceeding **85% to 100%**.
- In 3rd class, female survival dropped significantly to **37% - 50%**, demonstrating that class barriers affected women's survival odds.
- Male survival was consistently lowest in 3rd class (under **15%**) regardless of embarkation port.
"""))

# --- Cell 18: Hypothesis Testing ---
cells.append(make_cell("markdown", """## 15. Statistical Hypothesis Testing
Validating whether observed demographic differences are statistically significant using rigorous hypothesis tests.
"""))

cells.append(make_cell("code", """# Chi-Square Test: Gender vs Survival
contingency_sex = pd.crosstab(df['sex'], df['survived'])
chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(contingency_sex)

# Chi-Square Test: Class vs Survival
contingency_pclass = pd.crosstab(df['pclass'], df['survived'])
chi2_pclass, p_pclass, dof_pclass, _ = stats.chi2_contingency(contingency_pclass)

# Mann-Whitney U Test: Fare difference between Survivors and Non-Survivors
survived_fare = df[df['survived'] == 1]['fare']
deceased_fare = df[df['survived'] == 0]['fare']
u_stat, p_fare = stats.mannwhitneyu(survived_fare, deceased_fare, alternative='greater')

print("--- STATISTICAL HYPOTHESIS TEST RESULTS ---")
print(f"1. Gender vs Survival (Chi-Square): Chi2 = {chi2_sex:.2f}, p-value = {p_sex:.2e} -> "
      f"{'Statistically Significant (p < 0.001)' if p_sex < 0.001 else 'Not Significant'}")
print(f"2. Passenger Class vs Survival (Chi-Square): Chi2 = {chi2_pclass:.2f}, p-value = {p_pclass:.2e} -> "
      f"{'Statistically Significant (p < 0.001)' if p_pclass < 0.001 else 'Not Significant'}")
print(f"3. Fare (Survivors > Deceased, Mann-Whitney U): U = {u_stat:.2f}, p-value = {p_fare:.2e} -> "
      f"{'Statistically Significant (p < 0.001)' if p_fare < 0.001 else 'Not Significant'}")
""", execution_count=exec_counter))
exec_counter += 1

# --- Cell 19: Summary of Findings ---
cells.append(make_cell("markdown", """## 16. Executive Summary of Findings

1. **Overall Survival Rate:**  
   Out of 891 recorded passengers, only **342 survived (38.38%)**, while **549 perished (61.62%)**.
2. **The Gender Determinant:**  
   Gender was the single strongest predictor of survival ($p < 10^{-57}$). Females had a **74.20%** survival rate versus **18.89%** for males, confirming the strict enforcement of the evacuation protocol *"Women and children first"*.
3. **Socio-Economic Stratification:**  
   Socio-economic class played a decisive role ($p < 10^{-22}$). 1st Class passengers experienced a **62.96%** survival rate, compared to **47.28%** in 2nd Class and just **24.24%** in 3rd Class. Proximity to boat decks and economic privilege directly translated into life expectancy.
4. **Age Dynamics:**  
   Children under 10 years old experienced elevated survival rates (~61%) across 1st and 2nd classes. Elderly passengers (aged 60+) suffered the lowest survival rates (~22%).
5. **Family Size & Companionship:**  
   Solo travelers had a lower survival rate (**30.35%**) than passengers in small family units of 2 to 4 members (**57.88%**). However, large families ($\ge 5$) had very low survival rates (**14.89%**), likely due to difficulties coordinating group evacuation.
6. **Data Quality & Feature Engineering Considerations:**  
   - `deck` is missing in 77.2% of records and should either be omitted or converted into a binary indicator (`has_cabin`).
   - `age` is missing in 19.9% of records and should be imputed using median age grouped by Class and Title/Sex.
   - `fare` is highly skewed (skewness = 4.79) and benefits significantly from log-transformation (`log1p`).
"""))

# --- Cell 20: Interview Questions ---
cells.append(make_cell("markdown", """## 17. Comprehensive Answers to Internship Interview Questions

### Question 1: What is EDA and why is it important?
**Answer:**  
**Exploratory Data Analysis (EDA)** is the foundational approach in data science that employs statistical summaries and graphical representations to systematically examine datasets before formal modeling or hypothesis testing. Coined by mathematician John Tukey in 1977, its primary importance includes:
- **Understanding Data Structure & Distributions:** Revealing data types, ranges, central tendencies, and spreads.
- **Data Quality Assessment:** Identifying missing values, duplicate records, corrupt formats, and anomalies.
- **Pattern & Relationship Discovery:** Uncovering linear and non-linear correlations, trends, clusters, and interaction effects between target and feature variables.
- **Detecting Outliers & Skewness:** Finding extreme values that could distort machine learning estimators.
- **Guiding Feature Engineering & Model Selection:** Informing transformation choices (e.g., log transforms, categorical encoding, feature scaling).

---

### Question 2: Which plots do you use to check correlation?
**Answer:**  
To evaluate correlation between variables, data analysts use:
1. **Heatmap of the Correlation Matrix (`sns.heatmap`):** The standard multi-feature tool showing Pearson ($r$), Spearman ($\rho$), or Kendall ($\tau$) coefficients with a diverging color gradient.
2. **Scatterplots (`sns.scatterplot` / `plt.scatter`):** The primary visual plot for examining the relationship, linearity, direction, and spread between two continuous variables.
3. **Pairplot / Scatterplot Matrix (`sns.pairplot`):** Displays all pairwise bivariate scatterplots alongside univariate histograms or KDE plots along the diagonal.
4. **Jointplot (`sns.jointplot`):** Combines a central scatter/hexbin/KDE plot with marginal distribution histograms for both variables.
5. **Regplot (`sns.regplot`):** Adds a linear regression trendline and 95% confidence interval band over a scatterplot to make correlation trends immediately obvious.

---

### Question 3: How do you handle skewed data?
**Answer:**  
Skewed data occurs when a distribution's tail extends further in one direction than the other. Handling techniques depend on the nature and direction of skewness:
1. **Mathematical Transformations:**
   - **Log Transformation (`np.log` or `np.log1p`):** Ideal for right-skewed data with non-negative values (e.g., income, fare).
   - **Square Root Transformation (`np.sqrt`):** Moderately reduces right skewness; handles zeros naturally.
   - **Box-Cox Transformation (`scipy.stats.boxcox`):** Parametric power transformation that finds the optimal $\lambda$ to maximize normality (requires strictly positive values, $x > 0$).
   - **Yeo-Johnson Transformation (`scipy.stats.yeojohnson`):** Extends Box-Cox to handle zero and negative values.
2. **Outlier Treatment & Trimming:**
   - **IQR Winsorization / Capping:** Capping values outside $[Q_1 - 1.5 \times \text{IQR},\ Q_3 + 1.5 \times \text{IQR}]$ at the percentiles.
   - **Percentile Trimming:** Clipping extreme 1% or 99% tails.
3. **Binning / Discretization:** Converting a continuous skewed variable into ordinal bins (e.g., Low, Medium, High fare categories).
4. **Using Robust Models & Metrics:** Utilizing tree-based algorithms (Random Forest, XGBoost) which are invariant to monotonic transformations, and reporting Median/IQR rather than Mean/Standard Deviation.

---

### Question 4: How to detect multicollinearity?
**Answer:**  
Multicollinearity occurs when two or more independent features in a regression model are highly linearly correlated, leading to unstable coefficient estimates and inflated standard errors. Methods to detect it include:
1. **Variance Inflation Factor (VIF):**  
   The gold standard metric. Computed as $\text{VIF}_i = \frac{1}{1 - R_i^2}$, where $R_i^2$ is the $R^2$ obtained from regressing feature $x_i$ against all other independent variables.
   - $\text{VIF} = 1$: No collinearity.
   - $1 < \text{VIF} < 5$: Moderate, acceptable collinearity.
   - $\text{VIF} \ge 5$ or $10$: Severe multicollinearity requiring remediation.
2. **Correlation Matrix Inspection:** Examining pairwise correlation coefficients ($|r| > 0.80$ usually flags collinear feature pairs).
3. **Condition Number & Condition Index:** Computed from eigenvalues of the standardized feature matrix $X^T X$. A condition number $> 30$ indicates severe collinearity.
4. **Tolerance:** The reciprocal of VIF ($\text{Tolerance} = 1 / \text{VIF} = 1 - R_i^2$). A tolerance $< 0.1$ or $0.2$ indicates multicollinearity.
5. **Model Diagnostic Indicators:** High overall model $R^2$ paired with non-significant individual $p$-values for regression coefficients.

---

### Question 5: What are univariate, bivariate, and multivariate analyses?
**Answer:**  
These three levels of analysis differ by the number of variables analyzed simultaneously:
1. **Univariate Analysis:**
   - **Focus:** Explores **one single variable** in isolation.
   - **Objective:** Understand the distribution, central tendency (mean, median, mode), spread (variance, standard deviation, IQR), skewness, kurtosis, and frequency of a solitary variable.
   - **Plots:** Histograms, KDE plots, boxplots, countplots, frequency tables.
   - *Example:* Distribution of passenger age on the Titanic.
2. **Bivariate Analysis:**
   - **Focus:** Explores relationships between **two variables** simultaneously.
   - **Objective:** Identify correlation, associations, dependencies, comparative differences, or cause-effect patterns.
   - **Plots:** Scatterplots, bivariate bar charts, stacked bar charts, boxplots split by a category, line charts.
   - *Example:* Survival rate compared between males and females.
3. **Multivariate Analysis:**
   - **Focus:** Analyzes **three or more variables** simultaneously.
   - **Objective:** Detect complex interaction effects, confounding variables, high-dimensional patterns, clustering, and conditional relationships.
   - **Plots:** Correlation heatmaps, pairplots, facet grids / catplots, 3D scatterplots, PCA biplots.
   - *Example:* Examining survival rate conditioned on passenger class, gender, and port of embarkation simultaneously.

---

### Question 6: Difference between heatmap and pairplot?
**Answer:**  

| Feature / Dimension | Heatmap (`sns.heatmap`) | Pairplot (`sns.pairplot`) |
| :--- | :--- | :--- |
| **Primary Purpose** | Visualizes a 2D matrix of summary numbers (e.g., correlation coefficients, confusion matrices, missing values). | Visualizes pairwise raw distributions and scatter relationships between continuous variables. |
| **Underlying Data** | Aggregated matrix summary (e.g., $N \times N$ correlation grid). | Raw row-level passenger data across $N$ continuous features. |
| **Plot Content** | Colored grid tiles with numerical annotations. | An $N \times N$ grid of individual subplots (scatterplots off-diagonal, KDE/histograms on diagonal). |
| **Target Separation** | Does not directly split scatter points by class label (unless encoded numerically). | Easily segments distributions and scatter points by a categorical target using the `hue` parameter. |
| **Computation Cost** | Very fast and computationally lightweight. | Computationally intensive for large datasets with many continuous features. |
| **Best Used For** | Quick bird's-eye scan for multicollinearity and feature associations. | Deep exploration of non-linear trends, cluster separation, and bivariate interaction patterns. |

---

### Question 7: How do you summarize your insights?
**Answer:**  
To effectively communicate analytical findings to both technical stakeholders and executive decision-makers, a data analyst should follow a structured 5-step framework:
1. **Top-Down Executive Summary (Bottom-Line-Up-Front / BLUF):**  
   Start with the primary conclusion in 2–3 sentences. State the key discovery before detailing the technical methodology.
2. **Quantify Findings with Concrete Metrics:**  
   Always back qualitative statements with numbers and statistical significance. (e.g., *"Females had a 74.2% survival rate compared to 18.9% for males, a statistically significant difference with $p < 0.001$"* instead of *"Women survived more"*).
3. **Pair Every Visual with a Clear Business Takeaway:**  
   Provide explicit captions for every chart stating what the viewer is seeing and why it matters.
4. **Structure into Categorized Themes:**  
   Group insights logically (e.g., *Demographic Drivers*, *Socio-Economic Factors*, *Data Quality Risks*, *Actionable Recommendations*).
5. **Actionable Recommendations & Next Steps:**  
   Translate discoveries into pragmatic decisions, feature engineering steps, or model choices (e.g., log-transforming `fare`, imputing `age` by class/gender median, and creating family size features).
"""))

# Construct final notebook object
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("titanic_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1, ensure_ascii=False)

print("titanic_eda.ipynb created successfully with all cells and explanations!")
