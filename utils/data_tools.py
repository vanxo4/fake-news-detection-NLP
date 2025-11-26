import pandas as pd

#Some useful tools for data processing

def underSample2Min(df, label_name):
    """
    Undersamples the dataframe so that all label groups have the same size,
    corresponding to the size of the minority class.

    Args:
        df (pd.DataFrame): The input dataframe to be balanced.
        label_name (str): The name of the column containing the target labels.

    Returns:
        pd.DataFrame: A new dataframe with balanced classes and a reset index.
    """
    
    # 1. Count label frequencies
    # We get the count of each unique value in the target column
    vc = df[label_name].value_counts()
    lab2freq = dict(zip(vc.index.tolist(), vc.values.tolist()))
    
    # 2. Find the minimum frequency (size of the minority class)
    min_freq = min(lab2freq.values())
    print(f"--- Undersampling: Reducing all classes to {min_freq} samples ---")

    # 3. Random sampling
    idx_sample = []
    for selected_label, _ in lab2freq.items():
        # Filter by class and sample 'min_freq' random indices
        # random_state=42 ensures reproducibility
        sel_indexes = df[df[label_name] == selected_label].sample(n=min_freq, random_state=42).index.tolist()
        idx_sample += sel_indexes
    
    # Sort indices to maintain some original order (optional)
    idx_sample.sort()

    # 4. Construct the new DataFrame
    # Extract rows based on the selected indices
    df_balanced = df.loc[idx_sample, :].copy()
    
    # Reset index to avoid gaps in the index sequence
    # drop=True removes the old index column entirely
    df_balanced = df_balanced.reset_index(drop=True)
    
    return df_balanced

def check_overfitting(model, X_train, y_train, X_test, y_test, threshold=0.05):
    """
    Calculates and compares accuracy scores for training and test sets 
    to diagnose potential overfitting.

    Args:
        model: The trained scikit-learn estimator (must have a .score() method).
        X_train: Training feature set.
        y_train: Training labels.
        X_test: Test feature set.
        y_test: Test labels.
        threshold (float): The maximum acceptable difference between train and test 
                           accuracy before flagging overfitting (default: 0.05 for 5%).

    Returns:
        dict: A dictionary containing 'train_accuracy', 'test_accuracy', and 'gap'.
    """
    
    print("\n--- Overfitting Diagnosis ---")
    
    # 1. Calculate scores
    # .score() returns accuracy for classification models by default
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # 2. Calculate the gap (How much better is the model on training data?)
    gap = train_acc - test_acc
    
    # 3. Print report
    print(f"🎯 Train Accuracy: {train_acc:.4f} ({(train_acc*100):.2f}%)")
    print(f"🎯 Test Accuracy:  {test_acc:.4f} ({(test_acc*100):.2f}%)")
    print(f"📉 Gap:            {gap:.4f} ({(gap*100):.2f}%)")
    
    # 4. Diagnostic logic
    if gap > threshold:
        print(f"⚠️  WARNING: High overfitting detected! The gap is larger than {threshold}.")
    elif gap < -0.02:
        # Negative gap means Test > Train (rare, usually means data mismatch or small test set)
        print(f"❓ NOTICE: Underfitting or Data Mismatch? Test score is higher than Train.")
    else:
        print(f"✅ STATUS: Model is robust. Generalization gap is within acceptable limits.")
        
    print("-----------------------------\n")

    return {"train_acc": train_acc, "test_acc": test_acc, "gap": gap}