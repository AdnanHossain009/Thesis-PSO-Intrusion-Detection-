import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Step 1: Importing data_loader...")
    from src.data_loader import load_rt_iot2022_dataset
    print("OK")
    
    print("\nStep 2: Loading dataset...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "RT_IOT2022")
    X, y, summary = load_rt_iot2022_dataset(dataset_path)
    print("OK")
    
    print("\nStep 3: Importing preprocessing...")
    from src.preprocessing import preprocess_rt_iot2022
    print("OK")
    
    print("\nStep 4: Running preprocessing...")
    result = preprocess_rt_iot2022(X, y, test_size=0.2, validation_size=0.15)
    print("OK")
    
    print("\nStep 5: Displaying results...")
    print(f"X_train: {result['X_train'].shape}")
    print(f"X_test: {result['X_test'].shape}")
    print(f"X_val: {result['X_val'].shape}")
    print(f"y_train: {result['y_train'].shape}")
    print(f"y_test: {result['y_test'].shape}")
    print(f"y_val: {result['y_val'].shape}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
