import json
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from llm_engine import analyze_review_batch

# --- Configuration Area ---
# Ensure this file exists in your data folder
TEST_FILE = "data/uploaded_reviews-101-200.xlsx" 

# Test Gradients: We will test these specific batch sizes
TEST_SIZES = [1, 2, 3, 4] 

# Sample Limit: Test all 100 rows
SAMPLE_LIMIT = 100 

def load_test_data():
    """Load test data, supports JSON and Excel"""
    if not os.path.exists(TEST_FILE):
        print(f"Error: File not found {TEST_FILE}")
        return []

    try:
        # --- 1. If Excel file ---
        if TEST_FILE.endswith(".xlsx"):
            print(f"Reading Excel file: {TEST_FILE}...")
            df = pd.read_excel(TEST_FILE)
            
            # Smart column detection (Reusing logic)
            text_col = None
            priority_cols = ['text', 'content', 'body', 'comment', 'review_text', 'review', 'Review']
            for col in df.columns:
                if col.lower() in priority_cols:
                    text_col = col
                    break
            if not text_col:
                text_col = df.columns[0] # Fallback
                
            print(f"   -> Selected Column: {text_col}")
            
            # Convert to list format
            data = []
            for _, row in df.iterrows():
                # Convert to string to avoid errors
                data.append({"text": str(row[text_col])})
            
            return data[:SAMPLE_LIMIT]

        # --- 2. If JSON file ---
        else:
            with open(TEST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data[:SAMPLE_LIMIT]

    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def run_batch_test(data, batch_size):
    """
    Run data with specific Batch Size, return results dict.
    """
    print(f"\n>>> Testing Batch Size: {batch_size} ...")
    
    # Construct input with temp IDs
    batch_input = [{"id": idx, "text": item['text']} for idx, item in enumerate(data)]
    
    results = {}
    start_time = time.time()
    
    # Loop through batches
    for i in range(0, len(batch_input), batch_size):
        chunk = batch_input[i : i + batch_size]
        try:
            # Call Core Engine
            batch_out = analyze_review_batch(chunk)
            
            # Parse Output
            for res in batch_out:
                if res.insights:
                    # Simplify result for comparison: Aspect|Sentiment
                    val = f"{res.insights[0].aspect}|{res.insights[0].sentiment}"
                    results[res.id] = val
                else:
                    results[res.id] = "none|none"
        except Exception as e:
            print(f"Batch Error: {e}")
            
    elapsed = time.time() - start_time
    print(f"    Time: {elapsed:.2f} s")
    return results, elapsed

def main():
    data = load_test_data()
    if not data:
        return
    
    print(f"Start Stress Test. Samples: {len(data)}")
    print("--------------------------------------------------")

    # 1. Establish Baseline (Batch Size=1)
    print("Establishing Baseline (Size=1)...")
    print("(This takes time as it runs one by one...)")
    baseline_results, base_time = run_batch_test(data, 1)
    
    report = []

    # 2. Loop through test sizes
    for size in TEST_SIZES:
        if size == 1: continue # Skip baseline
        
        curr_results, curr_time = run_batch_test(data, size)
        
        # Calculate Consistency
        matches = 0
        total = len(data)
        
        for idx in range(total):
            base_val = baseline_results.get(idx, "missing_base")
            curr_val = curr_results.get(idx, "missing_curr")
            
            if base_val == curr_val:
                matches += 1
        
        consistency = (matches / total) * 100
        speedup = base_time / curr_time if curr_time > 0 else 0
        
        print(f"    -> Consistency: {consistency:.2f}%")
        print(f"    -> Speedup: {speedup:.1f} x")
        
        report.append({
            "Batch Size": size,
            "Consistency (%)": round(consistency, 2),
            "Speedup (x)": round(speedup, 1)
        })

    # 3. Generate Final Report
    print("\n" + "="*50)
    print("Final Optimization Report")
    print("="*50)
    df = pd.DataFrame(report)
    print(df.to_string(index=False))
    
    # 4. Auto Recommendation
    valid_configs = df[df["Consistency (%)"] >= 95]
    
    if not valid_configs.empty:
        best = valid_configs.sort_values("Speedup (x)", ascending=False).iloc[0]
        print("\n✅ Recommended Optimal Batch Size:")
        print(f"   Size: {int(best['Batch Size'])}")
        print(f"   Consistency: {best['Consistency (%)']}%")
        print(f"   Speedup: {best['Speedup (x)']} x")
    else:
        print("\n⚠️ Warning: No batch size achieved 95% consistency.")
        print("Suggestion: Stick to smaller batch sizes.")

    # [Optional] Plotting
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Consistency (%)', color=color)
        ax1.plot(df['Batch Size'], df['Consistency (%)'], color=color, marker='o', label='Consistency')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=95, color='r', linestyle='--', label='95% Threshold')
        
        ax2 = ax1.twinx()  
        color = 'tab:green'
        ax2.set_ylabel('Speedup (x)', color=color)
        ax2.plot(df['Batch Size'], df['Speedup (x)'], color=color, marker='x', linestyle=':', label='Speedup')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title(f'Batch Size Stress Test (N={len(data)})')
        plt.tight_layout() # Fix layout
        plt.show()
    except Exception as e:
        print(f"Plotting failed (ignore if running on server): {e}")

if __name__ == "__main__":
    main()