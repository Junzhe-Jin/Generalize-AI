import pandas as pd
import time
from llm_engine import analyze_review_batch

def process_excel_file(filepath, batch_size=20):
    """
    Reads an Excel file and processes it using API-Level Batching.
    """
    df = pd.read_excel(filepath)
    
    # --- Intelligent Column Matching Logic ---
    text_col = None
    priority_cols = ['text', 'content', 'body', 'comment', 'review_text', 'review text']
    
    # Check for exact matches in priority list
    for col in df.columns:
        if col.lower() in priority_cols:
            text_col = col
            break
            
    # If no priority column found, search for 'review' or 'feedback' keywords
    if not text_col:
        for col in df.columns:
            c_low = col.lower()
            if ('review' in c_low or 'feedback' in c_low) and 'id' not in c_low and 'date' not in c_low:
                text_col = col
                break
    
    # Fallback: Select the column with the longest average string length (likely to be the review text)
    if not text_col:
        try:
            text_col = df.astype(str).apply(lambda x: x.str.len()).mean().idxmax()
        except:
            text_col = df.columns[0]
            
    print(f"DEBUG: Selected column for analysis: {text_col}")
    # -----------------------

    analyzed_data = []
    total_rows = len(df)
    print(f"Total reviews: {total_rows}. Processing with Batch Size: {batch_size}")

    # --- Batch Processing Loop ---
    for i in range(0, total_rows, batch_size):
        end_idx = min(i + batch_size, total_rows)
        chunk_df = df.iloc[i:end_idx]
        
        # 1. Prepare Batch Data Package (List of Dicts)
        batch_input = []
        text_map = {} # For validation: ID -> Original Text
        
        for idx, row in chunk_df.iterrows():
            text = str(row[text_col])
            # Simple filtering
            if len(text) < 5 or text.lower() == 'nan':
                continue
            
            # Use DataFrame Index as unique ID
            batch_input.append({"id": idx, "text": text})
            text_map[idx] = text
        
        if not batch_input:
            continue

        print(f"Processing Batch {i//batch_size + 1}: Sending {len(batch_input)} reviews to LLM...")

        # 2. Call LLM (Send the entire list)
        batch_results = analyze_review_batch(batch_input)
        
        # 3. Map results back to data
        # Create a temporary dictionary for easy lookup: result_id -> insights
        results_map = {res.id: res.insights for res in batch_results}
        
        # Iterate through the original sent IDs to ensure every item has a result (even if LLM missed it)
        for item in batch_input:
            original_id = item['id']
            original_text = item['text']
            
            insights = results_map.get(original_id, [])
            
            if not insights:
                # LLM did not return analysis for this item, or analysis was empty
                analyzed_data.append({
                    "original_text": original_text,
                    "aspect": "other", "sentiment": "neutral",
                    "evidence": "", "rationale": "No insight detected (or skipped by LLM)"
                })
            else:
                for insight in insights:
                    analyzed_data.append({
                        "original_text": original_text,
                        "aspect": insight.aspect,
                        "sentiment": insight.sentiment,
                        "evidence": insight.evidence,
                        "rationale": insight.rationale
                    })
        
        # Optional: Brief pause to respect rate limits if needed
        # time.sleep(0.5)

    return pd.DataFrame(analyzed_data)

def calculate_stats(results_df):
    if results_df.empty:
        return pd.DataFrame()
    stats = pd.crosstab(results_df['aspect'], results_df['sentiment'])
    return stats