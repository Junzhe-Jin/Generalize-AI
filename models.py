from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Define Enums to ensure consistent classification
AspectEnum = Literal['product_quality', 'usability', 'delivery_shipping', 'price_value', 'service', 'other']
SentimentEnum = Literal['positive', 'negative', 'neutral', 'mixed']

class ReviewInsight(BaseModel):
    """
    A specific insight extracted from a single review (e.g., a complaint about slow shipping).
    """
    aspect: AspectEnum = Field(..., description="The primary category of the review.")
    sentiment: SentimentEnum = Field(..., description="The sentiment toward that aspect.")
    evidence: str = Field(..., description="A direct quote from the review supporting the finding.")
    rationale: str = Field(..., description="Brief explanation of why this classification was made.")

class ReviewResult(BaseModel):
    """
    [New Structure] The complete analysis result for a single review.
    Includes an ID to map the result back to the original Excel row.
    """
    id: int = Field(..., description="The unique ID provided in the prompt for this specific review.")
    insights: List[ReviewInsight] = Field(default_factory=list, description="List of insights found in this review. Can be empty if no relevant info.")

class BatchResponse(BaseModel):
    """
    [Modified Structure] The top-level container for the API response.
    It now contains a list of ReviewResults, rather than flattened ReviewInsights.
    """
    reviews: List[ReviewResult] = Field(..., description="The list of analyzed reviews, matched by their IDs.")