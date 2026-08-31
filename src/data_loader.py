"""
DATA LOADING MODULE FOR RT_IOT2022 DATASET

This module handles:
1. Loading the RT_IOT2022 CSV file from the dataset directory
2. Detecting and removing unnecessary identifier columns
3. Identifying the target column (label for classification)
4. Separating features (X) and labels (y)
5. Providing comprehensive dataset diagnostics

The module does NOT preprocess data - that is handled by preprocessing.py.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Optional, Dict, List

# Set UTF-8 encoding for stdout to handle special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class DataLoader:
    """
    Load and analyze the RT_IOT2022 intrusion detection dataset.
    
    Attributes:
        dataset_path (str): Path to the RT_IOT2022 CSV file
        df (pd.DataFrame): Loaded dataset
        target_column (str): Name of the target/label column
        feature_columns (list): Names of feature columns
        unnecessary_columns (list): Columns removed from the dataset
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_path (str): Path to the RT_IOT2022 CSV file
        """
        self.dataset_path = dataset_path
        self.df = None
        self.target_column = None
        self.feature_columns = None
        self.unnecessary_columns = []
        
    def load(self) -> pd.DataFrame:
        """
        Load the RT_IOT2022 dataset from CSV.
        
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If dataset file does not exist
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        print(f"\n{'='*80}")
        print(f"DATA LOADING: RT_IOT2022 DATASET")
        print(f"{'='*80}")
        print(f"\nLoading dataset from: {self.dataset_path}")
        
        # Load the CSV file
        self.df = pd.read_csv(self.dataset_path)
        
        print(f"[OK] Dataset loaded successfully")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        return self.df
    
    def identify_target_column(self, target_name: Optional[str] = None) -> str:
        """
        Identify the target/label column.
        
        The target column is typically the last column or can be specified explicitly.
        
        Args:
            target_name (str, optional): Explicit name of target column.
                                        If None, uses the last column.
        
        Returns:
            str: Name of the target column
        """
        if target_name:
            if target_name not in self.df.columns:
                raise ValueError(f"Target column '{target_name}' not found in dataset")
            self.target_column = target_name
        else:
            # By convention, the last column is the target (Attack_type)
            self.target_column = self.df.columns[-1]
        
        print(f"\n{'='*80}")
        print(f"TARGET COLUMN IDENTIFICATION")
        print(f"{'='*80}")
        print(f"Target column name: '{self.target_column}'")
        print(f"Target column data type: {self.df[self.target_column].dtype}")
        
        return self.target_column
    
    def identify_unnecessary_columns(self) -> List[str]:
        """
        Identify and remove unnecessary columns.
        
        Removes:
        - Index columns (Unnamed: 0, index, etc.)
        - Identifier columns that don't provide predictive value for classification
        
        IMPORTANT: These columns are identified but only removed when
        the user calls remove_unnecessary_columns(). This allows inspection first.
        
        Returns:
            list: Names of unnecessary columns
        """
        print(f"\n{'='*80}")
        print(f"UNNECESSARY COLUMN DETECTION")
        print(f"{'='*80}")
        
        unnecessary = []
        
        # Check for pandas auto-generated index columns
        for col in self.df.columns:
            if col.startswith('Unnamed'):
                reason = "Auto-generated index column (Unnamed)"
                print(f"[OK] Found: '{col}' - {reason}")
                unnecessary.append(col)
        
        # Other potential unnecessary columns (network identifiers that don't aid classification)
        # id.orig_p and id.resp_p are port numbers - could be useful, but we'll mention them
        potentially_useful = []
        for col in ['id.orig_p', 'id.resp_p']:
            if col in self.df.columns:
                potentially_useful.append(col)
        
        if potentially_useful:
            print(f"\n[WARN] Columns that are identifiers but potentially useful:")
            for col in potentially_useful:
                print(f"  - '{col}' (Type: {self.df[col].dtype})")
                print(f"    → Keep: Source/destination ports may have attack patterns")
        
        self.unnecessary_columns = unnecessary
        return unnecessary
    
    def remove_unnecessary_columns(self) -> None:
        """
        Remove the identified unnecessary columns from the dataset.
        
        This is a separate step to allow inspection before deletion.
        """
        if not self.unnecessary_columns:
            print("\nNo unnecessary columns to remove.")
            return
        
        print(f"\n{'='*80}")
        print(f"REMOVING UNNECESSARY COLUMNS")
        print(f"{'='*80}")
        
        for col in self.unnecessary_columns:
            print(f"Removing: '{col}'")
            self.df = self.df.drop(columns=[col])
        
        print(f"[OK] Removed {len(self.unnecessary_columns)} unnecessary column(s)")
        print(f"  New shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
    
    def analyze_features(self) -> Dict[str, List[str]]:
        """
        Analyze and categorize all features.
        
        Returns:
            dict: Dictionary with feature categories
                - 'numerical': List of numerical feature names
                - 'categorical': List of categorical feature names
                - 'total': Total number of features (excluding target)
        """
        print(f"\n{'='*80}")
        print(f"FEATURE ANALYSIS")
        print(f"{'='*80}")
        
        # Get all columns except target
        all_columns = [c for c in self.df.columns if c != self.target_column]
        
        # Categorize features
        numerical_features = self.df[all_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df[all_columns].select_dtypes(include=['object']).columns.tolist()
        
        self.feature_columns = all_columns
        
        print(f"\nTotal features (excluding target): {len(all_columns)}")
        print(f"\nNumerical features: {len(numerical_features)}")
        for i, col in enumerate(numerical_features, 1):
            dtype = self.df[col].dtype
            print(f"  {i:3d}. {col:<40s} ({dtype})")
        
        print(f"\nCategorical features: {len(categorical_features)}")
        for i, col in enumerate(categorical_features, 1):
            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique()
            print(f"  {i:3d}. {col:<40s} ({dtype}, {n_unique} unique values)")
        
        return {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'total': len(all_columns)
        }
    
    def analyze_target(self) -> Dict:
        """
        Analyze the target column and class distribution.
        
        Returns:
            dict: Dictionary with target statistics
        """
        print(f"\n{'='*80}")
        print(f"TARGET ANALYSIS - CLASS DISTRIBUTION")
        print(f"{'='*80}")
        
        class_dist = self.df[self.target_column].value_counts()
        class_pct = (class_dist / len(self.df) * 100).round(2)
        
        print(f"\nTotal samples: {len(self.df):,}")
        print(f"Number of classes: {len(class_dist)}")
        print(f"\nClass distribution:")
        
        for idx, (class_label, count) in enumerate(class_dist.items(), 1):
            pct = class_pct[class_label]
            bar = '#' * int(pct / 2)  # Visual bar (50 chars = 100%)
            print(f"  {idx:2d}. {str(class_label):<30s} : {count:>7,} ({pct:>6.2f}%) {bar}")
        
        # Check for class imbalance
        max_pct = class_pct.max()
        min_pct = class_pct.min()
        imbalance_ratio = max_pct / min_pct
        
        print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 10:
            print(f"  [WARN] WARNING: Significant class imbalance detected!")
            print(f"    → May require stratified splitting or resampling techniques")
        
        return {
            'class_distribution': class_dist.to_dict(),
            'class_percentages': class_pct.to_dict(),
            'num_classes': len(class_dist),
            'imbalance_ratio': float(imbalance_ratio)
        }
    
    def check_data_quality(self) -> Dict:
        """
        Check data quality issues: missing values, duplicates, etc.
        
        Returns:
            dict: Dictionary with quality metrics
        """
        print(f"\n{'='*80}")
        print(f"DATA QUALITY CHECK")
        print(f"{'='*80}")
        
        # Missing values
        missing_total = self.df.isnull().sum().sum()
        print(f"\nMissing values:")
        if missing_total == 0:
            print(f"  [OK] No missing values found")
        else:
            print(f"  [WARN] Found {missing_total} missing value(s)")
            missing_by_col = self.df.isnull().sum()
            for col, count in missing_by_col[missing_by_col > 0].items():
                print(f"    - {col}: {count} missing")
        
        # Duplicates
        dup_count = self.df.duplicated().sum()
        print(f"\nDuplicate rows:")
        if dup_count == 0:
            print(f"  [OK] No duplicate rows found")
        else:
            print(f"  [WARN] Found {dup_count} duplicate row(s)")
        
        # Data types
        print(f"\nData type distribution:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count} columns")
        
        return {
            'missing_values': int(missing_total),
            'duplicate_rows': int(dup_count),
            'data_types': dtype_counts.to_dict()
        }
    
    def separate_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target (y) from the dataset.
        
        Returns:
            tuple: (X, y) where:
                - X is a DataFrame containing all feature columns
                - y is a Series containing the target labels
        """
        print(f"\n{'='*80}")
        print(f"SEPARATING FEATURES AND TARGET")
        print(f"{'='*80}")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        print(f"\nFeatures (X) shape: {X.shape}")
        print(f"Target (y) shape: {y.shape}")
        print(f"\nFeature columns ({len(X.columns)}):")
        for i, col in enumerate(X.columns, 1):
            dtype = X[col].dtype
            if dtype in ['float64', 'int64']:
                print(f"  {i:3d}. {col:<40s} ({dtype})")
            else:
                unique = X[col].nunique()
                print(f"  {i:3d}. {col:<40s} ({dtype}, {unique} unique)")
        
        print(f"\nTarget column: '{self.target_column}'")
        print(f"Target data type: {y.dtype}")
        
        return X, y
    
    def print_summary(self) -> Dict:
        """
        Print a comprehensive summary of the loaded and analyzed dataset.
        
        Returns:
            dict: Summary statistics
        """
        print(f"\n{'='*80}")
        print(f"DATASET SUMMARY - RT_IOT2022")
        print(f"{'='*80}")
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'total_features': len(self.feature_columns) if self.feature_columns else 0,
            'target_column': self.target_column,
            'dataset_path': self.dataset_path,
            'unnecessary_columns_removed': len(self.unnecessary_columns)
        }
        
        print(f"\nDataset shape: {summary['total_rows']:,} samples × {summary['total_columns']} columns")
        print(f"Features: {summary['total_features']}")
        print(f"Target: {summary['target_column']}")
        print(f"Unnecessary columns removed: {summary['unnecessary_columns_removed']}")
        
        return summary


def load_rt_iot2022_dataset(
    dataset_path: str = "RT_IOT2022",
    target_column: Optional[str] = None,
    remove_unnecessary: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Convenience function to load and prepare the RT_IOT2022 dataset.
    
    This is the main entry point for data loading.
    
    Args:
        dataset_path (str): Path to RT_IOT2022 CSV file. Default: "RT_IOT2022"
        target_column (str, optional): Name of target column. Default: None (uses last column)
        remove_unnecessary (bool): Whether to remove unnecessary columns. Default: True
    
    Returns:
        tuple: (X, y, summary) where:
            - X is a DataFrame of features
            - y is a Series of labels
            - summary is a Dict with dataset statistics
    
    Example:
        >>> X, y, summary = load_rt_iot2022_dataset("RT_IOT2022")
        >>> print(f"Features: {X.shape}, Labels: {y.shape}")
    """
    loader = DataLoader(dataset_path)
    
    # Load dataset
    loader.load()
    
    # Identify target
    loader.identify_target_column(target_column)
    
    # Analyze data quality
    quality = loader.check_data_quality()
    
    # Identify and remove unnecessary columns
    loader.identify_unnecessary_columns()
    if remove_unnecessary and loader.unnecessary_columns:
        loader.remove_unnecessary_columns()
    
    # Analyze features
    features_info = loader.analyze_features()
    
    # Analyze target distribution
    target_info = loader.analyze_target()
    
    # Separate X and y
    X, y = loader.separate_features_and_target()
    
    # Print summary
    summary = loader.print_summary()
    
    # Add additional info to summary
    summary.update({
        'features_info': features_info,
        'target_info': target_info,
        'quality_info': quality
    })
    
    print(f"\n{'='*80}")
    print(f"[OK] DATA LOADING COMPLETE")
    print(f"{'='*80}\n")
    
    return X, y, summary


if __name__ == "__main__":
    """
    Main entry point for testing the data loader.
    
    Run with: python src/data_loader.py
    """
    import sys
    
    # Determine dataset path (assuming we're in the project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "RT_IOT2022")
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Looking for dataset at: {dataset_path}")
    
    try:
        # Load the dataset
        X, y, summary = load_rt_iot2022_dataset(dataset_path)
        
        # Display additional information
        print(f"\n{'='*80}")
        print(f"FINAL DATA READY FOR PREPROCESSING")
        print(f"{'='*80}")
        print(f"\nX (features): {X.shape}")
        print(f"y (target): {y.shape}")
        print(f"\nFirst 5 samples of X:")
        print(X.head())
        print(f"\nFirst 5 samples of y:")
        print(y.head())
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
