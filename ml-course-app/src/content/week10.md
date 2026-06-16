# Week 11: Pandas Mastery & Data Wrangling

> **Goal:** Turn messy, real-world CSV tables into clean, analysis-ready tensors—fluently. Data is never clean in the real world. If you can't wrangle data, you can't do Machine Learning.

---

## Part 1: The Pandas DataFrame

Pandas is built on top of NumPy. While NumPy handles homogeneous arrays of numbers, Pandas handles heterogeneous tabular data (like an Excel spreadsheet or SQL table) where each column can have a different type (int, float, string, boolean).

The core object is the **DataFrame**.

### 1.1 Indexing (`loc` vs `iloc`)
The most common Pandas mistake is confusing `loc` and `iloc`.
- `.iloc[]` uses **integer positions** (like a NumPy array). `df.iloc[0]` gets the first row.
- `.loc[]` uses **labels** (the actual names of the indices/columns). `df.loc['Alice']` gets the row labeled "Alice".

```python
import pandas as pd

data = {'Age': [25, 30, 35], 'Income': [50k, 70k, 90k]}
df = pd.DataFrame(data, index=['Alice', 'Bob', 'Charlie'])

# Using iloc (positional)
first_person = df.iloc[0] 

# Using loc (label-based)
alice_data = df.loc['Alice'] 

# Selecting rows AND columns
bobs_income = df.loc['Bob', 'Income']
```

---

## Part 2: Data Cleaning

Machine Learning models require numbers. They cannot process `NaN` (missing values), `"Unknown"` strings, or duplicate rows.

### 2.1 Missing Data
You must decide how to handle missing data (`NaN`).
1. **Drop it:** If you have plenty of data, just drop rows with missing values using `df.dropna()`.
2. **Impute it:** Fill the missing values with the column's mean, median, or a constant using `df.fillna(df['Age'].mean())`.

### 2.2 Fixing Data Types
Sometimes a column of numbers is accidentally read as strings because one row has a typo (e.g., `"100"` vs `"10O"`).
Use `pd.to_numeric(df['Column'], errors='coerce')`. The `coerce` flag forces unparseable strings into `NaN`, allowing you to clean them.

---

## Part 3: Reshaping and Grouping

### 3.1 Split-Apply-Combine (`groupby`)
The most powerful operation in Pandas is `groupby`. It follows a three-step process:
1. **Split** the DataFrame into smaller pieces based on a key (e.g., group by 'City').
2. **Apply** a function to each piece independently (e.g., calculate the mean 'Income').
3. **Combine** the results back into a new DataFrame.

```python
import pandas as pd

data = {
    'City': ['NY', 'LA', 'NY', 'LA', 'SF'],
    'Income': [80, 70, 90, 80, 120]
}
df = pd.DataFrame(data)

# Calculate average income per city
avg_income = df.groupby('City')['Income'].mean()
print(avg_income)
```

### 3.2 Tidy Data (`melt` and `pivot`)
In ML, we need "Tidy Data": every column is a feature, and every row is a sample.
Sometimes data comes in "wide" format (e.g., columns for `Year_2020`, `Year_2021`). We need to flatten this into a "long" format using `pd.melt()`.

If we have long format data and need to create a matrix out of it, we use `df.pivot_table()`.

---

## Part 4: Practice Exercises

### Exercise 1: The Wrangling Pipeline
You have a messy CSV string. Use Pandas to clean it up.

```python
import pandas as pd
import io

csv_data = """Patient_ID,Age,Blood_Pressure,Group
1,45,120/80,Control
2,50,NaN,Treatment
3,invalid,130/85,Control
4,55,140/90,Treatment
5,45,120/80,Control
""" # Note that patient 5 is a duplicate of patient 1!

df = pd.read_csv(io.StringIO(csv_data))

# 1. Drop duplicate rows based on Patient_ID
df = df.drop_duplicates(subset=['Patient_ID'])

# 2. Fix the Age column (force 'invalid' to NaN)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# 3. Fill missing Ages with the median Age
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# 4. Split 'Blood_Pressure' into two numeric columns: 'Systolic' and 'Diastolic'
# Hint: df['Blood_Pressure'].str.split('/', expand=True)
bp_split = df['Blood_Pressure'].str.split('/', expand=True)
df['Systolic'] = pd.to_numeric(bp_split[0])
df['Diastolic'] = pd.to_numeric(bp_split[1])
df = df.drop('Blood_Pressure', axis=1) # Drop original

# 5. Group by 'Group' and find the mean Systolic pressure
print(df.groupby('Group')['Systolic'].mean())
```

### Exercise 2: Vectorization vs `apply`
Pandas has an `apply()` function that runs a custom Python function on every row. **Never use it unless absolutely necessary.** It is just a hidden `for` loop and is incredibly slow.

1. Create a DataFrame with 1,000,000 rows of random floats.
2. Calculate the square root of the column using `df['col'].apply(np.sqrt)`. Time it.
3. Calculate the square root using Pandas vectorization: `np.sqrt(df['col'])`. Time it.
4. Marvel at the speed difference.

---

## Self-Test Questions
1. **What is the difference between `merge` and `concat`?** *(`concat` blindly stacks DataFrames on top of each other or side-by-side. `merge` acts like a SQL JOIN, combining rows based on a shared Key column.)*
2. **Why is calling `df.dropna()` without checking your data dangerous?** *(If every row has at least one missing value in some random column, `dropna()` will delete your entire dataset!)*
3. **What does the `axis=1` parameter usually mean in Pandas?** *(It means the operation is applied across columns horizontally, rather than down rows vertically. e.g., `df.drop('ColA', axis=1)` drops a column, whereas `axis=0` would attempt to drop a row with index 'ColA'.)*
