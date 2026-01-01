import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from models import BatchResponse
from config import MODEL_ANALYSIS, MODEL_SUMMARY, SYSTEM_PROMPT

load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_review_batch(reviews_batch):
    """
    Receives a batch of review data and sends it to the LLM for analysis in one go.
    
    Optimized Prompt structure to reduce result discrepancies caused by Batch Size (Robustness),
    and added seed parameter to improve determinism (Repeatability).
    
    Args:
        reviews_batch: List[dict], format like [{"id": 1, "text": "Good..."}, {"id": 2, "text": "Bad..."}]
    """
    # 1. Construct Batch Prompt: Use strong separators
    # Compared to simple newlines, this explicit marking helps the LLM better isolate different reviews in its attention mechanism.
    # It's like adding an "envelope" to each review.
    user_content = "Here is the batch of reviews to analyze. Remember to treat each one independently:\n"
    
    for item in reviews_batch:
        user_content += f"\n<<< START REVIEW ID: {item['id']} >>>\n"
        user_content += f"{item['text']}\n"
        user_content += f"<<< END REVIEW ID: {item['id']} >>>\n"
        user_content += "-" * 20  # Dashed line separator for further visual isolation

    try:
        # 2. Call API (Use Structured Outputs)
        completion = client.beta.chat.completions.parse(
            model=MODEL_ANALYSIS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            # Expecting the new BatchResponse defined in models.py (containing reviews list)
            response_format=BatchResponse,
            
            # [Important Change 1] Set to 0 to ensure consistency for repeatability testing
            temperature=0,
            
            # [Important Change 2] Fixed random seed
            # This tells OpenAI backend to use the same deterministic sampling path as much as possible
            # Greatly reduces the "butterfly effect" caused by Batch Size changes
            seed=42 
        )
        
        message = completion.choices[0].message
        
        # 3. Check for Refusal
        if message.refusal:
            print(f"Model refused to answer: {message.refusal}")
            return []

        # 4. Return parsed result list
        if message.parsed:
            # Note: Current structure is BatchResponse -> reviews (List[ReviewResult])
            return message.parsed.reviews
            
        # 5. Return empty on exception
        return []

    except Exception as e:
        print(f"LLM Batch Error: {e}")
        return []

def generate_marketing_summary(stats_text):
    """
    Generates a summary report for the CMO based on statistical results.
    """
    prompt = f"""
    You are a Strategic Marketing Consultant. 
    Based on the following analysis of customer reviews (Counts of Sentiment per Aspect):
    
    {stats_text}
    
    Please write a concise executive summary formatted with HTML tags (<h3>, <ul>, <li>, <p>).
    Structure it as:
    1. <h3>Executive Overview</h3>
    2. <h3>Top Pain Points (Risk Areas)</h3>
    3. <h3>Strongest Assets (What users love)</h3>
    4. <h3>Strategic Recommendations</h3>
    
    Keep it professional and actionable. Do NOT wrap the output in markdown code blocks (like ```html). Just return the raw HTML string.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_SUMMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 
        )
        content = response.choices[0].message.content
        
        # --- Fix: Clean up Markdown tags ---
        if content:
            # Remove leading ```html or ```
            content = content.replace("```html", "").replace("```", "")
            # Remove possible leading/trailing whitespace
            content = content.strip()
            
        return content
        
    except Exception as e:
        return f"Error generating summary: {e}"