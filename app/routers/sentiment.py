from io import StringIO
from typing import Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

router = APIRouter(
    prefix="/sentiment",
    tags=["sentiment"],
    responses={404: {"description": "Not found"}},
)

# Download NLTK resources (run this once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define request models
class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

# Define response models
class SentimentResponse(BaseModel):
    text: str
    sentiment: Dict[str, float]
    sentiment_label: str

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

# Helper function to get sentiment label
def get_sentiment_label(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Helper function to analyze sentiment
def analyze_sentiment(text: str) -> Dict:
    if not text.strip():
        return {
            "neg": 0.0,
            "neu": 1.0,
            "pos": 0.0,
            "compound": 0.0
        }
    
    return sentiment_analyzer.polarity_scores(text)

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_text(input_data: TextInput):
    try:
        sentiment = analyze_sentiment(input_data.text)
        sentiment_label = get_sentiment_label(sentiment["compound"])
        
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment,
            sentiment_label=sentiment_label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@router.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_texts(input_data: BatchTextInput):
    try:
        results = []
        
        for text in input_data.texts:
            sentiment = analyze_sentiment(text)
            sentiment_label = get_sentiment_label(sentiment["compound"])
            
            results.append(
                SentimentResponse(
                    text=text,
                    sentiment=sentiment,
                    sentiment_label=sentiment_label
                )
            )
        
        return BatchSentimentResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing batch sentiment: {str(e)}")

@router.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        # Read CSV file from uploaded content
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode('utf-8')))

        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'text' column")

        # Analyze sentiment
        df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(x))
        df['sentiment_label'] = df['sentiment'].apply(lambda x: get_sentiment_label(x['compound']))
        
        # Extract sentiment scores into separate columns
        df['negative'] = df['sentiment'].apply(lambda x: x['neg'])
        df['neutral'] = df['sentiment'].apply(lambda x: x['neu'])
        df['positive'] = df['sentiment'].apply(lambda x: x['pos'])
        df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
        
        # Drop the dictionary column
        df = df.drop('sentiment', axis=1)
        textList = []
        for txt in df['text']:
            cleanText = txt.replace(",", "")
            textList.append(cleanText)
        sample_df = pd.DataFrame({
        'sentiment': textList,
        'sentiment_label': df['sentiment_label'],
        'negative': df['negative'],
        'neutral': df['neutral'],
        'positive': df['positive'],
        'compound': df['compound']
    })

        return {"message": "Analysis complete", 
                "columns": list(sample_df.columns),
                "sample_rows": sample_df.to_dict(orient='records'), 
                "data": df}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing CSV: {str(e)}")