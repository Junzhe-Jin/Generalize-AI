import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from llm_engine import analyze_review_batch
from config import BATCH_SIZE # Import constant

# ... detect_complexity_tags function remains unchanged ...
def detect_complexity_tags(text):
    """
    Automatically tag text complexity based on linguistic features.
    """
    tags = []
    text_lower = text.lower()
    
    # 1. Detect Negation
    if re.search(r"\b(not|no|never|n't|cannot)\b", text_lower):
        tags.append("Negation")
        
    # 2. Detect Mixed/Contrast
    if re.search(r"\b(but|however|although|though|while)\b", text_lower):
        tags.append("Mixed/Contrast")
        
    # 3. Detect Long Text
    if len(text) > 150:
        tags.append("Long Text")
        
    if not tags:
        tags.append("Simple")
        
    return tags

def run_gold_standard_validation(json_filepath, dual_mode=False):
    """
    Reads the Gold Standard JSON and validates LLM performance.
    Batch Size is now fixed to the value in Config.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load JSON: {e}"}

    print(f"Starting validation on {len(gold_data)} samples. Fixed Batch Size: {BATCH_SIZE}. Dual Mode: {dual_mode}")

    # --- 1. Prepare Data ---
    batch_input = []
    ground_truth_map = {} 

    for idx, item in enumerate(gold_data):
        text = item.get('text', '')
        ground_truth_map[idx] = {
            "aspect": item.get('label_aspect', 'other'),
            "sentiment": item.get('label_sentiment', 'neutral'),
            "text": text,
            "tags": detect_complexity_tags(text) 
        }
        batch_input.append({"id": idx, "text": text})

    # --- 2. Get Predictions ---
    print(">>> Running Validation Pass 1...")
    predictions_pass_1 = _get_predictions(batch_input)
    
    consistency_score = None
    if dual_mode:
        print(">>> Running Validation Pass 2 (Consistency Check)...")
        predictions_pass_2 = _get_predictions(batch_input)
        
        # Calculate Consistency
        consistent_count = 0
        total_checks = len(gold_data)
        for idx in range(total_checks):
            p1 = predictions_pass_1.get(idx, [])
            p2 = predictions_pass_2.get(idx, [])
            
            res1 = (p1[0].aspect, p1[0].sentiment) if p1 else ("none", "none")
            res2 = (p2[0].aspect, p2[0].sentiment) if p2 else ("none", "none")
            
            if res1 == res2:
                consistent_count += 1
                
        consistency_score = round((consistent_count / total_checks) * 100, 2) if total_checks > 0 else 0.0

    # --- 3. Compare Line by Line ---
    y_true_aspect = []
    y_pred_aspect = []
    details = []
    edge_case_stats = {} 

    for idx in range(len(gold_data)):
        truth = ground_truth_map[idx]
        tags = truth['tags']
        predicted_insights = predictions_pass_1.get(idx, [])
        
        if predicted_insights:
            pred_aspect = predicted_insights[0].aspect
            pred_sentiment = predicted_insights[0].sentiment
        else:
            pred_aspect = 'other'
            pred_sentiment = 'neutral'
            
        y_true_aspect.append(truth['aspect'])
        y_pred_aspect.append(pred_aspect)
        
        is_correct = (truth['aspect'] == pred_aspect) and (truth['sentiment'] == pred_sentiment)
        
        # Update Edge Case Stats
        for tag in tags:
            if tag not in edge_case_stats:
                edge_case_stats[tag] = {"total": 0, "correct": 0}
            
            edge_case_stats[tag]["total"] += 1
            if is_correct:
                edge_case_stats[tag]["correct"] += 1

        details.append({
            "text": truth['text'][:50] + "...",
            "tags": ", ".join(tags),
            "true": f"{truth['aspect']} | {truth['sentiment']}",
            "pred": f"{pred_aspect} | {pred_sentiment}",
            "status": "✅ Match" if is_correct else "❌ Mismatch"
        })

    # --- 4. Calculate Metrics ---
    try:
        acc = accuracy_score(y_true_aspect, y_pred_aspect)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_aspect, y_pred_aspect, average='weighted', zero_division=0
        )
        
        report_dict = classification_report(
            y_true_aspect, y_pred_aspect, output_dict=True, zero_division=0
        )
        
        assert isinstance(report_dict, dict)
        aspect_breakdown = {k: v for k, v in report_dict.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}

        # Calculate Edge Case Accuracy
        final_edge_stats = []
        for tag, data in edge_case_stats.items():
            acc_rate = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
            final_edge_stats.append({
                "type": tag,
                "total": data['total'],
                "accuracy": round(acc_rate, 2)
            })

        metrics = {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(precision) * 100, 2),
            "recall": round(float(recall) * 100, 2),
            "f1": round(float(f1) * 100, 2),
            "aspect_breakdown": aspect_breakdown,
            "edge_case_stats": final_edge_stats,
            "details": details,
            "consistency": consistency_score
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        metrics = {"error": str(e)}
    
    return metrics

def _get_predictions(batch_input):
    """Helper: Execute predictions (Using BATCH_SIZE from Config)"""
    predictions_map = {}
    # Use imported constant BATCH_SIZE
    for i in range(0, len(batch_input), BATCH_SIZE):
        chunk = batch_input[i : i + BATCH_SIZE]
        batch_results = analyze_review_batch(chunk)
        for res in batch_results:
            predictions_map[res.id] = res.insights
    return predictions_map