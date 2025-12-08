# Titanic Dataset – Exploratory Data Analysis 🚢

A complete exploratory data analysis (EDA) of the classic Titanic passenger dataset using Python, Pandas, Matplotlib, and Seaborn. The goal is to understand which factors (gender, class, age, fare, family size, etc.) were most strongly related to passenger survival. The Titanic dataset is one of the most commonly used beginner datasets for data analysis and machine learning practice on platforms like Kaggle. [web:30]

---

## 📁 Project Structure

titanic-eda-analysis/
├── notebooks/
│ └── titanic_eda.ipynb # Main EDA notebook
├── data/
│ └── titanic.csv # Titanic dataset
├── images/
│ ├── 01_missing_values.png
│ ├── 02_numerical_distribution.png
│ ├── 03_categorical_distribution.png
│ ├── 04_survival_analysis.png
│ ├── 05_correlation_heatmap.png
│ └── 06_family_analysis.png
├── README.md # Project documentation



---

## 📊 Dataset Description

The analysis uses the standard Titanic training dataset with passenger-level information:

- **Rows**: 891 passengers
- **Target**: `Survived` (0 = No, 1 = Yes)
- **Key Features**:
  - `Pclass` – Passenger class (1, 2, 3)
  - `Sex` – Gender
  - `Age` – Age in years
  - `SibSp` – Number of siblings/spouses aboard
  - `Parch` – Number of parents/children aboard
  - `Fare` – Ticket fare
  - `Embarked` – Port of embarkation (C/Q/S)

The dataset can be loaded either from a local CSV in `data/titanic.csv` or directly from a public URL inside the notebook.

---

## 🧪 Analysis Workflow (What the Notebook Does)

All analysis is implemented in `notebooks/titanic_eda.ipynb`. The notebook is organized into clear, sequential sections:

### 1. Setup & Data Loading
- Imports core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, and `warnings`.
- Configures display options and visual style.
- Loads the Titanic dataset either from:
  - Local file: `data/titanic.csv`, or
  - Remote URL (e.g., a GitHub-hosted CSV).

### 2. Initial Exploration
- Views the first few rows (`head()`).
- Checks dataset dimensions (`shape`).
- Displays data types and non-null counts (`info()`).
- Computes descriptive statistics for numerical features (`describe()`).
- Checks for duplicate rows.

### 3. Missing Values & Data Cleaning
- Calculates count and percentage of missing values per column.
- Visualizes missingness with a **missing-values heatmap**.
- Handles missing values:
  - `Age`: filled with the median.
  - `Embarked`: filled with the most frequent value (mode).
  - `Cabin`: dropped due to excessive missingness.
- Re-checks that no critical missing values remain.

### 4. Univariate Analysis
- **Numerical features** (`Age`, `Fare`, `SibSp`, `Parch`):
  - Histograms with KDE curves.
  - Mean and median markers on each distribution.
- **Categorical features** (`Sex`, `Pclass`, `Embarked`, `Survived`):
  - Count plots with labels.
  - Value counts printed in text.

### 5. Bivariate Analysis & Survival Patterns
- Explores how survival relates to:
  - **Gender** – Survival by `Sex`.
  - **Class** – Survival by `Pclass`.
  - **Age** – Boxplots of `Age` vs `Survived`.
  - **Fare** – Boxplots of `Fare` vs `Survived`.
- Computes overall survival rate and group-wise survival percentages (by gender and class).

### 6. Correlation Analysis
- Encodes categorical variables:
  - `Sex`: male → 1, female → 0.
  - `Embarked`: encoded with `factorize()`.
- Computes correlation of all numeric features with `Survived`.
- Builds a **correlation heatmap** for:
  - `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`.
- Interprets which features are most strongly associated with survival.

### 7. Feature Engineering: Family Size
- Creates a new feature:
  - `FamilySize = SibSp + Parch + 1`.
- Groups by `FamilySize` to compute:
  - Passenger count per family size.
  - Survival rate per family size.
- Visualizes:
  - Bar plot of passenger count by family size.
  - Line plot of survival rate vs family size.

### 8. Summary of Key Findings
- Prints a structured text summary including:
  - Overall survival rate.
  - Survival by gender.
  - Survival by passenger class.
  - Average ages for survivors vs non-survivors.
  - Observations about fare and family size.

---

## 🔍 Key Insights

From the EDA, several clear patterns emerge:

- **Overall survival**: Around 38% of passengers survived.
- **Gender**:
  - Female passengers had a much higher survival rate than males.
  - Gender is the strongest single predictor of survival in the dataset.
- **Passenger Class**:
  - 1st class passengers had significantly higher survival rates than 2nd and 3rd class.
  - Lower classes were much more vulnerable.
- **Age**:
  - Children tended to have better survival prospects than older adults.
  - The average age of survivors is slightly lower than that of non-survivors.
- **Fare**:
  - Higher fares are moderately associated with higher survival, reflecting better cabins and lifeboat access.
- **Family Size**:
  - Very small families and solo travelers show lower survival.
  - Moderate family sizes (e.g., 2–3 members) appear to have better survival rates.

These insights are valuable not only for understanding the disaster but also for selecting features for future predictive modeling.

---

## 📸 Visualizations

The following plots are generated and saved to the `images/` directory:

1. `01_missing_values.png` – Heatmap of missing data across all columns.
2. `02_numerical_distribution.png` – 2×2 grid of distributions for `Age`, `Fare`, `SibSp`, and `Parch`.
3. `03_categorical_distribution.png` – 2×2 grid of bar plots for `Sex`, `Pclass`, `Embarked`, and `Survived`.
4. `04_survival_analysis.png` – 2×2 grid:
   - Survival by gender.
   - Survival by passenger class.
   - Age vs survival (boxplot).
   - Fare vs survival (boxplot).
5. `05_correlation_heatmap.png` – Correlation matrix for key numeric features.
6. `06_family_analysis.png` – Passenger counts and survival rate by family size.

These figures can be reused in reports, slide decks, or LinkedIn posts.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Environment**: Jupyter Notebook (often run from VS Code)  
- **Libraries**:
  - `pandas` – data manipulation
  - `numpy` – numerical operations
  - `matplotlib` – plotting
  - `seaborn` – statistical visualization
  - `warnings` – suppressing non-critical warnings

---

## ⚙️ Installation & Setup

1. **Clone the repository**:

