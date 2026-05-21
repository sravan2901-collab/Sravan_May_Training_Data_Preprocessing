import pandas as pd
import numpy as np
df = pd.read_csv('student_scores.csv')

print("=== PART 1 — NUMPY OPERATIONS ===")

math_scores = df['math_score'].dropna().to_numpy()
print("Math scores array:", math_scores)


mean_val = np.mean(math_scores)
median_val = np.median(math_scores)
max_val = np.max(math_scores)
min_val = np.min(math_scores)

print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Max: {max_val:.2f}")
print(f"Min: {min_val:.2f}")


normalized_scores = (math_scores - min_val) / (max_val - min_val)
print("Normalized scores (first 5):", normalized_scores[:5])

print("\n=== PART 2 — PANDAS EXPLORATION ===")

print("First 5 rows:")
print(df.head())


print("\nData types:")
print(df.dtypes)


print("\nMissing values per column:")
print(df.isna().sum())


low_attendance = df[df['attendance'] < 70]
print("\nStudents with attendance below 70%:")
print(low_attendance[['student_id', 'name', 'attendance']])

print("\n=== PART 3 — DATA PREPROCESSING ===")

df_clean = df.copy()


df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')

num_cols = ['age', 'math_score', 'science_score', 'attendance']
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())


df_clean['gender'] = df_clean['gender'].fillna(df_clean['gender'].mode()[0])


df_clean['exam_date'] = pd.to_datetime(df_clean['exam_date'], errors='coerce')
print("Exam date converted to datetime. Sample:")
print(df_clean['exam_date'].head())


def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower_bound, upper_bound)

df_clean['math_score'] = cap_outliers(df_clean['math_score'])
df_clean['science_score'] = cap_outliers(df_clean['science_score'])
print("Outliers handled using IQR capping.")


print(f"Shape before removing duplicates: {df_clean.shape}")
df_clean = df_clean.drop_duplicates()
print(f"Shape after removing duplicates: {df_clean.shape}")

print("\n=== PART 4 — DATA ANALYSIS ===")

df_clean['average_score'] = (df_clean['math_score'] + df_clean['science_score']) / 2


top_5_students = df_clean.nlargest(5, 'average_score')[['student_id', 'name', 'average_score']]
print("Top 5 students by average_score:")
print(top_5_students)


correlation_matrix = df_clean[['attendance', 'math_score', 'science_score', 'average_score']].corr()
print("\nCorrelation matrix:")
print(correlation_matrix)


gender_group = df_clean.groupby('gender')[['math_score', 'science_score', 'average_score']].mean()
print("\nAverage marks by gender:")
print(gender_group)

print("\nPreprocessing and analysis complete!")
