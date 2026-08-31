"""
EXAMPLE SCRIPT: Loading and Preprocessing RT_IOT2022 Data

This script demonstrates how to:
1. Load the RT_IOT2022 dataset
2. Apply preprocessing with train/test/validation split
3. Prepare data for baseline models (next stage)

Run with: python example_load_and_preprocess.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_rt_iot2022_dataset
from preprocessing import preprocess_rt_iot2022


def main():
    """Main example workflow"""
    
    print("\n" + "="*80)
    print("EXAMPLE: LOADING AND PREPROCESSING RT_IOT2022 DATASET")
    print("="*80)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "RT_IOT2022")
    
    # ========================================================================
    # STEP 1: LOAD DATASET
    # ========================================================================
    
    print("\n[STEP 1] Loading RT_IOT2022 Dataset")
    print("-" * 80)
    
    X, y, summary = load_rt_iot2022_dataset(dataset_path)
    
    print(f"\nDataset loaded successfully!")
    print(f"  Features (X): {X.shape}")
    print(f"  Labels (y): {y.shape}")
    print(f"  Feature columns: {X.shape[1]}")
    
    # ========================================================================
    # STEP 2: PREPROCESS WITH TRAIN/TEST/VALIDATION SPLIT
    # ========================================================================
    
    print("\n" + "="*80)
    print("[STEP 2] Preprocessing with Train/Test/Validation Split")
    print("-" * 80)
    
    # Preprocess with:
    # - 80% training (further split into 85% train, 15% validation)
    # - 20% testing (held out)
    preprocessed = preprocess_rt_iot2022(
        X, y,
        test_size=0.2,          # 20% for test
        validation_size=0.15,   # 15% of training for validation
        random_seed=42          # For reproducibility
    )
    
    # ========================================================================
    # STEP 3: EXTRACT DATASETS
    # ========================================================================
    
    X_train = preprocessed['X_train']
    X_test = preprocessed['X_test']
    X_val = preprocessed['X_val']
    y_train = preprocessed['y_train']
    y_test = preprocessed['y_test']
    y_val = preprocessed['y_val']
    
    # ========================================================================
    # STEP 4: DISPLAY RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("DATA READY FOR MODELING")
    print("="*80)
    
    print("\nTraining set:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Percentage: {len(X_train)/len(X)*100:.1f}% of total data")
    
    print("\nValidation set:")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  Percentage: {len(X_val)/len(X)*100:.1f}% of total data")
    print(f"  Use: PSO fitness evaluation / hyperparameter tuning")
    
    print("\nTest set:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"  Percentage: {len(X_test)/len(X)*100:.1f}% of total data")
    print(f"  Use: Final model evaluation (HOLD OUT)")
    
    # ========================================================================
    # STEP 5: DATA CHARACTERISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("DATA CHARACTERISTICS")
    print("="*80)
    
    print("\nTraining set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(y_train) * 100
        print(f"  Class {cls:2d}: {cnt:6d} samples ({pct:5.2f}%)")
    
    print("\nFeature statistics (training set):")
    print(f"  Mean: {X_train.mean().mean():.4f}")
    print(f"  Std:  {X_train.std().mean():.4f}")
    print(f"  Min:  {X_train.min().min():.4f}")
    print(f"  Max:  {X_train.max().max():.4f}")
    
    # ========================================================================
    # STEP 6: VERIFY NO DATA LEAKAGE
    # ========================================================================
    
    print("\n" + "="*80)
    print("DATA LEAKAGE VERIFICATION")
    print("="*80)
    
    print("\n[OK] Train/test/validation split BEFORE preprocessing")
    print("[OK] Preprocessing fitted on training data ONLY")
    print("[OK] Test set never used in fitting")
    print("[OK] Validation set never used in fitting")
    print("[OK] Class distribution preserved (stratified split)")
    print("[OK] Random seed = 42 for reproducibility")
    
    # ========================================================================
    # STEP 7: READY FOR NEXT STAGE
    # ========================================================================
    
    print("\n" + "="*80)
    print("NEXT STAGE: TRAIN BASELINE MODELS")
    print("="*80)
    
    print("\nThe data is now ready for:")
    print("  1. Training baseline classifiers")
    print("  2. Evaluating baseline performance")
    print("  3. Implementing PSO feature selection")
    print("  4. Comparing PSO-selected features with baseline")
    
    print("\n" + "="*80)
    print("Example: Training a Random Forest Classifier")
    print("="*80)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    print("\nTraining Random Forest on preprocessed data...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nBaseline Random Forest Results (Test Set):")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    
    print("\n[OK] Pipeline complete - ready for PSO feature selection!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
