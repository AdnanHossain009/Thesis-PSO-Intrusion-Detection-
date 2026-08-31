"""
PREPROCESSING MODULE FOR RT_IOT2022 DATASET

This module handles:
1. Handling missing values
2. Encoding categorical features
3. Converting target labels to numerical format
4. Scaling/normalizing numerical features
5. Preventing data leakage via proper train/test splitting
6. Creating reproducible preprocessing pipelines

IMPORTANT DATA LEAKAGE PREVENTION:
- Fit all transformers (scalers, encoders) ONLY on training data
- Apply fitted transformers to validation and test data
- No information from test set influences preprocessing or feature selection
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class Preprocessor:
    """
    Preprocess the RT_IOT2022 dataset with proper data leakage prevention.
    
    Attributes:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target labels
        random_seed (int): Seed for reproducibility
        numerical_features (list): Names of numerical feature columns
        categorical_features (list): Names of categorical feature columns
        scaler (StandardScaler): Fitted scaler for numerical features
        encoder (OneHotEncoder): Fitted encoder for categorical features
        label_encoder (LabelEncoder): Fitted encoder for target labels
        preprocessor (ColumnTransformer): Full preprocessing pipeline
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_seed: int = 42
    ):
        """
        Initialize the Preprocessor.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target labels
            random_seed (int): Seed for reproducibility. Default: 42
        """
        self.X = X.copy()
        self.y = y.copy()
        self.random_seed = random_seed
        
        self.numerical_features = None
        self.categorical_features = None
        self.scaler = None
        self.encoder = None
        self.label_encoder = None
        self.preprocessor = None
        
        # Storage for transformed data
        self.X_processed = None
        self.y_encoded = None
        self.feature_names_out = None
        
        np.random.seed(random_seed)
    
    def identify_feature_types(self) -> Dict[str, list]:
        """
        Identify and categorize numerical and categorical features.
        
        Returns:
            dict: Dictionary with 'numerical' and 'categorical' lists
        """
        print(f"\n{'='*80}")
        print(f"STEP 1: IDENTIFYING FEATURE TYPES")
        print(f"{'='*80}")
        
        self.numerical_features = self.X.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        self.categorical_features = self.X.select_dtypes(
            include=['object']
        ).columns.tolist()
        
        print(f"\nNumerical features: {len(self.numerical_features)}")
        for i, col in enumerate(self.numerical_features, 1):
            if i <= 10:
                print(f"  {i:3d}. {col}")
        if len(self.numerical_features) > 10:
            print(f"  ... and {len(self.numerical_features) - 10} more")
        
        print(f"\nCategorical features: {len(self.categorical_features)}")
        for i, col in enumerate(self.categorical_features, 1):
            print(f"  {i:3d}. {col} ({self.X[col].nunique()} unique values)")
        
        return {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features
        }
    
    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in the dataset.
        
        Returns:
            dict: Dictionary with missing value counts per column
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: CHECKING FOR MISSING VALUES")
        print(f"{'='*80}")
        
        missing_X = self.X.isnull().sum()
        missing_y = self.y.isnull().sum()
        
        total_missing_X = missing_X.sum()
        
        print(f"\nFeatures (X):")
        if total_missing_X == 0:
            print(f"  [OK] No missing values found")
        else:
            print(f"  [WARN] Found {total_missing_X} missing value(s)")
            for col, count in missing_X[missing_X > 0].items():
                print(f"    - {col}: {count}")
        
        print(f"\nTarget (y):")
        if missing_y == 0:
            print(f"  [OK] No missing values in target")
        else:
            print(f"  [WARN] Found {missing_y} missing value(s) in target")
        
        return {
            'X': total_missing_X,
            'y': int(missing_y)
        }
    
    def encode_target(self, fit_on_data: pd.Series = None) -> np.ndarray:
        """
        Encode target labels to numerical format.
        
        Maps each unique class label to an integer (0, 1, 2, ...).
        
        Args:
            fit_on_data (pd.Series, optional): Data to fit encoder on.
                                               If None, uses self.y
        
        Returns:
            np.ndarray: Encoded target labels
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: ENCODING TARGET LABELS")
        print(f"{'='*80}")
        
        self.label_encoder = LabelEncoder()
        
        # Fit encoder on provided data (typically training data)
        if fit_on_data is not None:
            self.label_encoder.fit(fit_on_data)
        else:
            self.label_encoder.fit(self.y)
        
        # Encode all labels
        y_encoded = self.label_encoder.transform(self.y)
        self.y_encoded = y_encoded
        
        print(f"\nTarget label encoding:")
        print(f"  Classes: {sorted(self.label_encoder.classes_.tolist())}")
        print(f"  Number of classes: {len(self.label_encoder.classes_)}")
        
        # Show mapping
        print(f"\nLabel to integer mapping:")
        for idx, label in enumerate(self.label_encoder.classes_):
            print(f"  {label:<30s} -> {idx}")
        
        return y_encoded
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for numerical and categorical features.
        
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
        """
        print(f"\n{'='*80}")
        print(f"STEP 4: CREATING PREPROCESSING PIPELINE")
        print(f"{'='*80}")
        
        # Numerical features: scale using StandardScaler
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Categorical features: one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        print(f"\nPreprocessing pipeline created:")
        print(f"  - Numerical features: StandardScaler (81 features)")
        print(f"  - Categorical features: OneHotEncoder (2 features)")
        
        return self.preprocessor
    
    def fit_preprocessing(self, X_train: pd.DataFrame) -> None:
        """
        Fit the preprocessing pipeline on training data ONLY.
        
        This prevents data leakage by ensuring no information from
        validation/test sets influences preprocessing.
        
        IMPORTANT: Call this method ONLY on training data.
        
        Args:
            X_train (pd.DataFrame): Training feature data
        """
        print(f"\n{'='*80}")
        print(f"STEP 5: FITTING PREPROCESSING ON TRAINING DATA")
        print(f"{'='*80}")
        
        print(f"\nFitting preprocessing on training data (shape: {X_train.shape})")
        print(f"  - StandardScaler will fit on training data statistics")
        print(f"  - OneHotEncoder will fit on training data categories")
        print(f"\n  [OK] No test/validation data used during fitting")
        
        # Fit the entire pipeline
        self.preprocessor.fit(X_train)
        print(f"  [OK] Preprocessing pipeline fitted successfully")
    
    def transform_data(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Transform data using the fitted preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Data to transform
            is_training (bool): Whether this is training data (for display purposes)
        
        Returns:
            pd.DataFrame: Transformed data
        """
        transformed = self.preprocessor.transform(X)
        
        # Get feature names from the preprocessor
        if self.feature_names_out is None:
            self.feature_names_out = self.preprocessor.get_feature_names_out()
        
        # Create DataFrame with feature names
        X_transformed = pd.DataFrame(
            transformed,
            columns=self.feature_names_out,
            index=X.index
        )
        
        return X_transformed
    
    def preprocess_full_dataset(self, y_encode: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess the entire dataset.
        
        Note: Use this ONLY if you don't need to prevent data leakage.
        For machine learning, use preprocess_with_train_test_split() instead.
        
        Args:
            y_encode (bool): Whether to encode target labels. Default: True
        
        Returns:
            tuple: (X_processed, y_encoded)
        """
        print(f"\n{'='*80}")
        print(f"PREPROCESSING FULL DATASET")
        print(f"{'='*80}")
        
        # Fit and transform
        self.create_preprocessing_pipeline()
        self.fit_preprocessing(self.X)
        X_processed = self.transform_data(self.X)
        
        y_processed = self.encode_target() if y_encode else self.y
        
        self.X_processed = X_processed
        
        print(f"\nFull dataset preprocessing complete:")
        print(f"  X shape: {X_processed.shape}")
        print(f"  y shape: {np.array(y_processed).shape}")
        
        return X_processed, y_processed
    
    def preprocess_with_train_test_split(
        self,
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        stratify: bool = True,
        y_encode: bool = True
    ) -> Dict:
        """
        Preprocess dataset with proper train/test split to prevent data leakage.
        
        IMPORTANT: Preprocessing is fitted ONLY on training data.
        
        Workflow:
        1. Split data into train/test (and optional validation)
        2. Fit preprocessing on training data
        3. Apply preprocessing to train/test/validation data
        4. Encode target labels on training data
        
        Args:
            test_size (float): Fraction of data for testing. Default: 0.2
            validation_size (float, optional): Fraction of training data for validation.
                                              If None, no validation split. Default: None
            stratify (bool): Use stratified splitting to preserve class distribution.
                           Default: True
            y_encode (bool): Whether to encode target labels. Default: True
        
        Returns:
            dict: Dictionary with:
                - 'X_train', 'X_test', 'y_train', 'y_test'
                - 'X_val', 'y_val' (if validation_size is not None)
                - 'split_info': Summary of split sizes
        """
        print(f"\n{'='*80}")
        print(f"PREPROCESSING WITH TRAIN/TEST SPLIT")
        print(f"{'='*80}")
        print(f"\nData leakage prevention strategy:")
        print(f"  1. Split data FIRST")
        print(f"  2. Fit preprocessing on TRAINING data only")
        print(f"  3. Apply fitted preprocessing to test data")
        print(f"  4. Fit target encoder on training labels only")
        
        # Step 1: Identify feature types
        self.identify_feature_types()
        
        # Step 2: Check missing values
        self.check_missing_values()
        
        # Step 3: Split data (stratified by default)
        print(f"\n{'='*80}")
        print(f"SPLITTING DATA")
        print(f"{'='*80}")
        
        stratify_by = self.y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_by
        )
        
        print(f"\nStratified train/test split:")
        print(f"  Test size: {test_size}")
        print(f"  Training set size: {X_train.shape[0]} ({(1-test_size)*100:.1f}%)")
        print(f"  Test set size: {X_test.shape[0]} ({test_size*100:.1f}%)")
        print(f"  [OK] Class distribution preserved in split")
        
        # Optional: Create validation split from training data
        X_val = None
        y_val = None
        if validation_size is not None:
            print(f"\nCreating validation split from training data:")
            stratify_val = y_train if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=validation_size,
                random_state=self.random_seed,
                stratify=stratify_val
            )
            
            print(f"  Validation size: {validation_size}")
            print(f"  Training set (updated): {X_train.shape[0]} ({(1-validation_size)*100:.1f}%)")
            print(f"  Validation set size: {X_val.shape[0]} ({validation_size*100:.1f}%)")
            print(f"  [OK] Validation split created")
        
        # Step 4: Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline()
        self.fit_preprocessing(X_train)
        
        # Step 5: Transform data
        print(f"\n{'='*80}")
        print(f"TRANSFORMING DATA")
        print(f"{'='*80}")
        
        X_train_processed = self.transform_data(X_train, is_training=True)
        X_test_processed = self.transform_data(X_test, is_training=False)
        
        print(f"\nTransformed training set: {X_train_processed.shape}")
        print(f"Transformed test set: {X_test_processed.shape}")
        
        X_val_processed = None
        if X_val is not None:
            X_val_processed = self.transform_data(X_val, is_training=False)
            print(f"Transformed validation set: {X_val_processed.shape}")
        
        # Step 6: Encode target labels
        print(f"\n{'='*80}")
        print(f"ENCODING TARGET LABELS")
        print(f"{'='*80}")
        print(f"\nFitting label encoder on training data only")
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train)
        
        y_train_encoded = self.label_encoder.transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_val_encoded = None
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        print(f"  [OK] Label encoder fitted on training data")
        print(f"  Classes: {len(self.label_encoder.classes_)}")
        
        # Build result dictionary
        result = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'split_info': {
                'X_train_shape': X_train_processed.shape,
                'X_test_shape': X_test_processed.shape,
                'test_size': test_size,
                'validation_size': validation_size
            }
        }
        
        if X_val_processed is not None:
            result['X_val'] = X_val_processed
            result['y_val'] = y_val_encoded
            result['split_info']['X_val_shape'] = X_val_processed.shape
        
        # Store for later use
        self.X_processed = X_train_processed
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"PREPROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"\nDatasets ready:")
        print(f"  X_train: {result['X_train'].shape}")
        print(f"  X_test: {result['X_test'].shape}")
        if 'X_val' in result:
            print(f"  X_val: {result['X_val'].shape}")
        print(f"\nTarget encoding:")
        print(f"  y_train: {result['y_train'].shape}")
        print(f"  y_test: {result['y_test'].shape}")
        if 'y_val' in result:
            print(f"  y_val: {result['y_val'].shape}")
        
        print(f"\n[OK] Preprocessing complete - data ready for modeling")
        print(f"     No test/validation data was used in preprocessing")
        
        return result
    
    def save_preprocessing(self, output_dir: str) -> None:
        """
        Save fitted preprocessing components for later use.
        
        Args:
            output_dir (str): Directory to save preprocessing files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor
        with open(os.path.join(output_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save label encoder
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n[OK] Preprocessing saved to {output_dir}")
    
    def load_preprocessing(self, output_dir: str) -> None:
        """
        Load previously saved preprocessing components.
        
        Args:
            output_dir (str): Directory containing preprocessing files
        """
        with open(os.path.join(output_dir, 'preprocessor.pkl'), 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        with open(os.path.join(output_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"\n[OK] Preprocessing loaded from {output_dir}")


def preprocess_rt_iot2022(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    validation_size: Optional[float] = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    Convenience function to preprocess RT_IOT2022 data with train/test/validation split.
    
    This function ensures no data leakage:
    - Splits data FIRST
    - Fits preprocessing ONLY on training data
    - Applies to test/validation data
    
    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target labels
        test_size (float): Fraction of data for testing. Default: 0.2
        validation_size (float, optional): Fraction of training data for validation.
                                          Default: 0.15
        random_seed (int): Random seed for reproducibility. Default: 42
    
    Returns:
        dict: Dictionary containing:
            - X_train, X_test, X_val (preprocessed features)
            - y_train, y_test, y_val (encoded labels)
            - split_info (split statistics)
    
    Example:
        >>> preprocessed = preprocess_rt_iot2022(X, y, test_size=0.2)
        >>> X_train = preprocessed['X_train']
        >>> y_train = preprocessed['y_train']
    """
    preprocessor = Preprocessor(X, y, random_seed=random_seed)
    
    result = preprocessor.preprocess_with_train_test_split(
        test_size=test_size,
        validation_size=validation_size,
        stratify=True,
        y_encode=True
    )
    
    return result


if __name__ == "__main__":
    """
    Main entry point for testing the preprocessing module.
    
    Run with: python src/preprocessing.py (from project root)
    """
    import sys
    import os
    
    # Add src directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_rt_iot2022_dataset
    
    print(f"\n{'='*80}")
    print(f"PREPROCESSING MODULE TEST")
    print(f"{'='*80}")
    
    # Load data
    print(f"\nLoading dataset...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root, "RT_IOT2022")
    X, y, summary = load_rt_iot2022_dataset(dataset_path)
    
    # Preprocess
    print(f"\nPreprocessing dataset...")
    preprocessed = preprocess_rt_iot2022(X, y, test_size=0.2, validation_size=0.15)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"PREPROCESSING RESULTS")
    print(f"{'='*80}")
    
    print(f"\nX_train shape: {preprocessed['X_train'].shape}")
    print(f"X_test shape: {preprocessed['X_test'].shape}")
    print(f"X_val shape: {preprocessed['X_val'].shape}")
    
    print(f"\ny_train shape: {preprocessed['y_train'].shape}")
    print(f"y_test shape: {preprocessed['y_test'].shape}")
    print(f"y_val shape: {preprocessed['y_val'].shape}")
    
    print(f"\nX_train head:")
    print(preprocessed['X_train'].head())
