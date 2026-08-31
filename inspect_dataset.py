import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('RT_IOT2022')

print("=" * 80)
print("RT_IOT2022 DATASET INSPECTION")
print("=" * 80)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n" + "=" * 80)
print("ALL COLUMN NAMES AND TYPES:")
print("=" * 80)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    print(f"{i:3d}. {col:<40s} | {str(dtype):<10s}")

print("\n" + "=" * 80)
print("TARGET/LABEL COLUMN (Last Column):")
print("=" * 80)
target_col = df.columns[-1]
print(f"Target Column Name: '{target_col}'")
print(f"\nClass Distribution:")
for class_val, count in df[target_col].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"  {class_val:<30s} : {count:>7d} ({pct:>6.2f}%)")

print("\n" + "=" * 80)
print("MISSING VALUES:")
print("=" * 80)
missing = df.isnull().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("No missing values found")
else:
    print(f"Total missing values: {total_missing}")
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count}")

print("\n" + "=" * 80)
print("DUPLICATE ROWS:")
print("=" * 80)
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows: {dup_count}")

print("\n" + "=" * 80)
print("DATA TYPES SUMMARY:")
print("=" * 80)
numeric_cols = df.select_dtypes(include=['number']).shape[1]
object_cols = df.select_dtypes(include=['object']).shape[1]
print(f"Numerical columns: {numeric_cols}")
print(f"Categorical/Object columns: {object_cols}")

print("\n" + "=" * 80)
print("FEATURES (X) - ALL COLUMNS EXCEPT TARGET:")
print("=" * 80)
feature_cols = [c for c in df.columns if c != target_col]
print(f"Number of features: {len(feature_cols)}")
print("\nFeature columns:")
for i, col in enumerate(feature_cols, 1):
    dtype = df[col].dtype
    print(f"{i:3d}. {col:<40s} ({dtype})")
