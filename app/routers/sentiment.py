import csv
from io import StringIO
from typing import Dict, List, Optional, Union
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, Query, BackgroundTasks
import pandas as pd
from pydantic import BaseModel, Field
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
import logging
from datetime import datetime
import asyncio
import os
import re
from collections import Counter
import spacy
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_api")

router = APIRouter(
    prefix="/sentiment",
    tags=["sentiment"],
    responses={
        404: {"description": "Not found"},
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    },
)

# Download required NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

# Initialize analyzers
sentiment_analyzer_vader = SentimentIntensityAnalyzer()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.warning(f"Failed to load spaCy model: {str(e)}")
    nlp = None

# Initialize transformers model (only when needed to save memory)
transformer_model = None
transformer_tokenizer = None

def get_transformer_analyzer():
    global transformer_model, transformer_tokenizer
    if transformer_model is None or transformer_tokenizer is None:
        try:
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Transformer model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load transformer model: {str(e)}")
            return None
    
    return pipeline("sentiment-analysis", model=transformer_model, tokenizer=transformer_tokenizer)

# Define request models
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    model: str = Field("vader", description="Model to use for sentiment analysis (vader, transformer, ensemble)")
    include_aspects: bool = Field(False, description="Whether to extract aspect-based sentiment")

class BatchTextInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of texts to analyze")
    model: str = Field("vader", description="Model to use for sentiment analysis (vader, transformer, ensemble)")
    include_aspects: bool = Field(False, description="Whether to extract aspect-based sentiment")

class CSVOptions(BaseModel):
    text_column: str = Field("text", description="Column name containing text to analyze")
    model: str = Field("vader", description="Model to use for sentiment analysis")
    include_aspects: bool = Field(False, description="Whether to extract aspect-based sentiment")
    batch_size: int = Field(1000, description="Number of rows to process in a batch")

# Define response models
class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    score: float

class SentimentResponse(BaseModel):
    text: str
    sentiment: Dict[str, float]
    sentiment_label: str
    confidence: float
    language: str = Field(None, description="Detected language")
    aspects: Optional[List[AspectSentiment]] = None
    emotion_analysis: Optional[Dict[str, float]] = None

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    summary: Dict[str, Union[float, Dict]]

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    result_path: Optional[str] = None

# In-memory job storage (in production, use Redis or similar)
job_store = {}

# Helper function to get sentiment label
def get_sentiment_label(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Helper function to determine confidence score
def get_confidence(scores: Dict[str, float]) -> float:
    # For VADER, confidence is based on the magnitude of the compound score
    compound = abs(scores["compound"])
    if compound > 0.75:
        return 0.9 + (compound - 0.75) * 0.4  # Max 0.9 + 0.1 = 1.0
    elif compound > 0.5:
        return 0.7 + (compound - 0.5) * 0.8  # Between 0.7-0.9
    elif compound > 0.25:
        return 0.5 + (compound - 0.25) * 0.8  # Between 0.5-0.7
    else:
        return 0.3 + compound * 0.8  # Between 0.3-0.5

# Helper function to detect language
def detect_language(text: str) -> str:
    try:
        # Simple language detection heuristic 
        # In production, use a proper language detection library like langdetect
        # This is a placeholder implementation
        common_english = set(['the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'with'])
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return "unknown"
        
        # Check for overlap with common English words
        if len(set(words).intersection(common_english)) > 0:
            return "en"
        
        # Add more language detection logic as needed
        return "unknown"
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return "unknown"

# Extract emotion scores from text
def analyze_emotions(text: str) -> Dict[str, float]:
    emotions = {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0
    }
    
    try:
        # Using TextBlob for emotion analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Simple mapping based on polarity and subjectivity
        if polarity > 0.3:
            emotions["joy"] = min(1.0, polarity * 1.5)
        elif polarity < -0.3:
            emotions["sadness"] = min(1.0, abs(polarity) * 1.2)
            emotions["anger"] = min(1.0, abs(polarity) * subjectivity)
        
        # Check for exclamation marks and question marks for surprise
        if "!" in text:
            emotions["surprise"] = min(1.0, text.count("!") * 0.2)
        if "?" in text:
            emotions["surprise"] = min(1.0, text.count("?") * 0.1)
            
        # Check for fear-related words
        fear_words = ["afraid", "scared", "terrified", "frightened", "fear", "worried", "anxious"]
        for word in fear_words:
            if re.search(r'\b' + word + r'\b', text.lower()):
                emotions["fear"] += 0.3
        emotions["fear"] = min(1.0, emotions["fear"])
        
    except Exception as e:
        logger.warning(f"Emotion analysis failed: {str(e)}")
    
    return emotions

# Extract aspects and their sentiment
def extract_aspects(text: str, sentiment_scores: Dict[str, float]) -> List[AspectSentiment]:
    if not nlp:
        return []
    
    try:
        aspects = []
        doc = nlp(text)
        
        # Extract noun phrases as potential aspects
        noun_phrases = []
        for chunk in doc.noun_chunks:
            noun_phrases.append(chunk.text)
        
        # If no noun phrases, try using nouns
        if not noun_phrases:
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            noun_phrases = nouns
        
        # Analyze sentiment for each aspect by looking at nearby adjectives
        for aspect in noun_phrases:
            # Find the aspect's position in the text
            aspect_doc = nlp(aspect)
            aspect_root = aspect_doc[-1]  # Use the last word as the root of the aspect
            
            # Look for related adjectives or sentiment-bearing words
            related_tokens = []
            for token in doc:
                if token.text.lower() == aspect_root.text.lower() or any(t.text.lower() == aspect_root.text.lower() for t in token.children):
                    # Look for adjectives in a window around the token
                    start_idx = max(0, token.i - 5)
                    end_idx = min(len(doc), token.i + 5)
                    for i in range(start_idx, end_idx):
                        if doc[i].pos_ in ["ADJ", "ADV"] or doc[i].text in ["good", "bad", "great", "terrible", "excellent", "poor"]:
                            related_tokens.append(doc[i].text)
            
            # If we have related tokens, analyze their sentiment
            if related_tokens:
                aspect_text = " ".join([aspect] + related_tokens)
                aspect_sentiment = sentiment_analyzer_vader.polarity_scores(aspect_text)
                aspect_label = get_sentiment_label(aspect_sentiment["compound"])
                aspects.append(AspectSentiment(
                    aspect=aspect,
                    sentiment=aspect_label,
                    score=aspect_sentiment["compound"]
                ))
            else:
                # Fall back to using the overall sentiment
                aspects.append(AspectSentiment(
                    aspect=aspect,
                    sentiment=get_sentiment_label(sentiment_scores["compound"]),
                    score=sentiment_scores["compound"]
                ))
                
        return aspects
    except Exception as e:
        logger.error(f"Aspect extraction failed: {str(e)}")
        return []

# VADER sentiment analysis
def analyze_sentiment_vader(text: str) -> Dict:
    if not text.strip():
        return {
            "neg": 0.0,
            "neu": 1.0,
            "pos": 0.0,
            "compound": 0.0
        }
    
    return sentiment_analyzer_vader.polarity_scores(text)

# Transformer-based sentiment analysis
def analyze_sentiment_transformer(text: str) -> Dict:
    analyzer = get_transformer_analyzer()
    if not analyzer or not text.strip():
        return {
            "neg": 0.0,
            "neu": 1.0,
            "pos": 0.0,
            "compound": 0.0
        }
    
    try:
        result = analyzer(text)
        if result[0]['label'] == 'POSITIVE':
            return {
                "neg": 0.0,
                "neu": 1.0 - result[0]['score'],
                "pos": result[0]['score'],
                "compound": result[0]['score'] * 2 - 1  # Convert 0-1 to -1 to 1 scale
            }
        else:
            return {
                "neg": result[0]['score'],
                "neu": 1.0 - result[0]['score'],
                "pos": 0.0,
                "compound": -result[0]['score']  # Convert to negative for NEGATIVE sentiment
            }
    except Exception as e:
        logger.error(f"Transformer analysis failed: {str(e)}")
        return analyze_sentiment_vader(text)  # Fallback to VADER

# TextBlob sentiment analysis
def analyze_sentiment_textblob(text: str) -> Dict:
    if not text.strip():
        return {
            "neg": 0.0,
            "neu": 1.0,
            "pos": 0.0,
            "compound": 0.0
        }
    
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Range: -1 to 1
        subjectivity = blob.sentiment.subjectivity  # Range: 0 to 1
        
        # Convert TextBlob output to VADER-like format
        if polarity > 0:
            pos = (polarity + 1) / 2  # Convert from -1,1 to 0,1
            return {
                "neg": 0.0,
                "neu": 1.0 - pos,
                "pos": pos,
                "compound": polarity
            }
        elif polarity < 0:
            neg = (-polarity + 1) / 2  # Convert from -1,1 to 0,1
            return {
                "neg": neg,
                "neu": 1.0 - neg,
                "pos": 0.0,
                "compound": polarity
            }
        else:
            return {
                "neg": 0.0,
                "neu": 1.0,
                "pos": 0.0,
                "compound": 0.0
            }
    except Exception as e:
        logger.error(f"TextBlob analysis failed: {str(e)}")
        return analyze_sentiment_vader(text)  # Fallback to VADER

# Ensemble sentiment analysis (combining multiple models)
def analyze_sentiment_ensemble(text: str) -> Dict:
    try:
        # Get scores from different models
        vader_scores = analyze_sentiment_vader(text)
        blob_scores = analyze_sentiment_textblob(text)
        
        # Try transformer if available
        transformer_scores = None
        try:
            transformer_scores = analyze_sentiment_transformer(text)
        except:
            pass
        
        # Calculate weighted average 
        weights = {"vader": 0.5, "textblob": 0.3, "transformer": 0.2}
        
        if transformer_scores:
            # Use all three models
            compound = (
                vader_scores["compound"] * weights["vader"] +
                blob_scores["compound"] * weights["textblob"] +
                transformer_scores["compound"] * weights["transformer"]
            )
            pos = (
                vader_scores["pos"] * weights["vader"] +
                blob_scores["pos"] * weights["textblob"] +
                transformer_scores["pos"] * weights["transformer"]
            )
            neg = (
                vader_scores["neg"] * weights["vader"] +
                blob_scores["neg"] * weights["textblob"] +
                transformer_scores["neg"] * weights["transformer"]
            )
        else:
            # Just use VADER and TextBlob with adjusted weights
            adjusted_weights = {"vader": weights["vader"] / (weights["vader"] + weights["textblob"]),
                              "textblob": weights["textblob"] / (weights["vader"] + weights["textblob"])}
            
            compound = (
                vader_scores["compound"] * adjusted_weights["vader"] +
                blob_scores["compound"] * adjusted_weights["textblob"]
            )
            pos = (
                vader_scores["pos"] * adjusted_weights["vader"] +
                blob_scores["pos"] * adjusted_weights["textblob"]
            )
            neg = (
                vader_scores["neg"] * adjusted_weights["vader"] +
                blob_scores["neg"] * adjusted_weights["textblob"]
            )
        
        neu = 1.0 - pos - neg
        
        return {
            "neg": neg,
            "neu": neu,
            "pos": pos,
            "compound": compound
        }
    except Exception as e:
        logger.error(f"Ensemble analysis failed: {str(e)}")
        return analyze_sentiment_vader(text)  # Fallback to VADER

# Main analyzer function that dispatches to the right model
def analyze_sentiment(text: str, model: str = "vader", include_aspects: bool = False) -> Dict:
    try:
        # Choose model based on input
        if model == "transformer":
            sentiment = analyze_sentiment_transformer(text)
        elif model == "textblob":
            sentiment = analyze_sentiment_textblob(text)
        elif model == "ensemble":
            sentiment = analyze_sentiment_ensemble(text)
        else:  # Default to VADER
            sentiment = analyze_sentiment_vader(text)
        
        sentiment_label = get_sentiment_label(sentiment["compound"])
        confidence = get_confidence(sentiment)
        language = detect_language(text)
        
        response = {
            "text": text,
            "sentiment": sentiment,
            "sentiment_label": sentiment_label,
            "confidence": confidence,
            "language": language,
            "emotion_analysis": analyze_emotions(text)
        }
        
        # Add aspect-based sentiment if requested
        if include_aspects:
            response["aspects"] = extract_aspects(text, sentiment)
            
        return response
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        # Return a safe fallback
        return {
            "text": text,
            "sentiment": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
            "sentiment_label": "neutral",
            "confidence": 0.3,
            "language": "unknown",
            "emotion_analysis": {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0}
        }

# Function to create a summary of batch analysis
def create_summary(results: List[SentimentResponse]) -> Dict:
    total = len(results)
    if total == 0:
        return {
            "count": 0,
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "average_scores": {"compound": 0, "positive": 0, "negative": 0, "neutral": 0},
            "confidence": {"average": 0, "min": 0, "max": 0},
            "emotions": {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0}
        }
    
    # Count sentiment labels
    sentiment_counts = Counter([r.sentiment_label for r in results])
    
    # Average scores
    avg_compound = sum(r.sentiment["compound"] for r in results) / total
    avg_pos = sum(r.sentiment["pos"] for r in results) / total
    avg_neg = sum(r.sentiment["neg"] for r in results) / total
    avg_neu = sum(r.sentiment["neu"] for r in results) / total
    
    # Confidence stats
    confidences = [r.confidence for r in results]
    avg_confidence = sum(confidences) / total
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    
    # Emotion averages
    emotions = {
        "joy": sum(r.emotion_analysis["joy"] for r in results) / total if results[0].emotion_analysis else 0,
        "sadness": sum(r.emotion_analysis["sadness"] for r in results) / total if results[0].emotion_analysis else 0,
        "anger": sum(r.emotion_analysis["anger"] for r in results) / total if results[0].emotion_analysis else 0,
        "fear": sum(r.emotion_analysis["fear"] for r in results) / total if results[0].emotion_analysis else 0,
        "surprise": sum(r.emotion_analysis["surprise"] for r in results) / total if results[0].emotion_analysis else 0
    }
    
    return {
        "count": total,
        "sentiment_distribution": {
            "positive": sentiment_counts.get("positive", 0) / total,
            "neutral": sentiment_counts.get("neutral", 0) / total,
            "negative": sentiment_counts.get("negative", 0) / total
        },
        "average_scores": {
            "compound": avg_compound,
            "positive": avg_pos,
            "negative": avg_neg,
            "neutral": avg_neu
        },
        "confidence": {
            "average": avg_confidence,
            "min": min_confidence,
            "max": max_confidence
        },
        "emotions": emotions
    }

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_text(input_data: TextInput):
    try:
        logger.info(f"Analyzing text with model: {input_data.model}")
        result = analyze_sentiment(
            input_data.text, 
            model=input_data.model,
            include_aspects=input_data.include_aspects
        )
        
        return SentimentResponse(
            text=input_data.text,
            sentiment=result["sentiment"],
            sentiment_label=result["sentiment_label"],
            confidence=result["confidence"],
            language=result["language"],
            aspects=result.get("aspects"),
            emotion_analysis=result["emotion_analysis"]
        )
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@router.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_texts(input_data: BatchTextInput):
    try:
        results = []
        
        # Process texts in smaller chunks to avoid timeouts
        for text in input_data.texts:
            result = analyze_sentiment(
                text, 
                model=input_data.model,
                include_aspects=input_data.include_aspects
            )
            
            results.append(
                SentimentResponse(
                    text=text,
                    sentiment=result["sentiment"],
                    sentiment_label=result["sentiment_label"],
                    confidence=result["confidence"],
                    language=result["language"],
                    aspects=result.get("aspects"),
                    emotion_analysis=result["emotion_analysis"]
                )
            )
        
        # Create summary
        summary = create_summary(results)
        
        return BatchSentimentResponse(results=results, summary=summary)
    except Exception as e:
        logger.error(f"Error analyzing batch sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing batch sentiment: {str(e)}")

async def process_csv_async(job_id: str, file_content: bytes, options: CSVOptions):
    try:
        job_store[job_id]["status"] = "processing"
        
        # Read CSV file
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        if options.text_column not in df.columns:
            job_store[job_id]["status"] = "failed"
            job_store[job_id]["error"] = f"CSV must contain a '{options.text_column}' column"
            return
        
        total_rows = len(df)
        processed_rows = 0
        
        # Process in batches
        result_dfs = []
        for i in range(0, total_rows, options.batch_size):
            batch = df.iloc[i:i+options.batch_size].copy()
            
            # Apply sentiment analysis to each row
            sentiment_results = []
            for text in batch[options.text_column]:
                sentiment_results.append(
                    analyze_sentiment(text, model=options.model, include_aspects=options.include_aspects)
                )
            
            # Extract sentiment scores and other fields
            batch['sentiment_label'] = [r["sentiment_label"] for r in sentiment_results]
            batch['negative'] = [r["sentiment"]["neg"] for r in sentiment_results]
            batch['neutral'] = [r["sentiment"]["neu"] for r in sentiment_results]
            batch['positive'] = [r["sentiment"]["pos"] for r in sentiment_results]
            batch['compound'] = [r["sentiment"]["compound"] for r in sentiment_results]
            batch['confidence'] = [r["confidence"] for r in sentiment_results]
            batch['language'] = [r["language"] for r in sentiment_results]
            
            # Extract emotions
            batch['joy'] = [r["emotion_analysis"]["joy"] for r in sentiment_results]
            batch['sadness'] = [r["emotion_analysis"]["sadness"] for r in sentiment_results]
            batch['anger'] = [r["emotion_analysis"]["anger"] for r in sentiment_results]
            batch['fear'] = [r["emotion_analysis"]["fear"] for r in sentiment_results]
            batch['surprise'] = [r["emotion_analysis"]["surprise"] for r in sentiment_results]
            
            # Handle aspects if included
            if options.include_aspects:
                # Store as JSON string since CSV can't handle nested structures
                batch['aspects'] = [
                    json.dumps([{"aspect": a["aspect"], "sentiment": a["sentiment"], "score": a["score"]} 
                              for a in r.get("aspects", [])]) 
                    for r in sentiment_results
                ]
            
            result_dfs.append(batch)
            
            processed_rows += len(batch)
            job_store[job_id]["progress"] = processed_rows / total_rows
            
            # Small delay to avoid blocking
            await asyncio.sleep(0.01)
        
        # Combine all batches
        result_df = pd.concat(result_dfs)
        
        # Fixed: Explicitly set line ending to '\r\n' for Windows compatibility
        # This ensures proper line breaks in Excel and other spreadsheet software
        csv_content = result_df.to_csv(index=False, lineterminator='\r\n', quoting=csv.QUOTE_ALL)
        
        # Update job status
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["progress"] = 1.0
        job_store[job_id]["result_content"] = csv_content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_store[job_id]["filename"] = f"sentiment_analysis_{timestamp}.csv"
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)

@router.post("/analyze-csv")
async def analyze_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    model: str = Form("vader"),
    include_aspects: bool = Form(False),
    batch_size: int = Form(1000)
):
    try:
        content = await file.read()
        
        # Create job
        job_id = f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename.replace('.', '_')}"
        job_store[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "file_name": file.filename
        }
        
        # Setup options
        options = CSVOptions(
            text_column=text_column,
            model=model,
            include_aspects=include_aspects,
            batch_size=batch_size
        )
        
        # Start processing in background
        background_tasks.add_task(process_csv_async, job_id, content, options)
        
        return {
            "message": "CSV analysis started",
            "job_id": job_id,
            "status": "queued"
        }
    except Exception as e:
        logger.error(f"Error initiating CSV analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error initiating CSV analysis: {str(e)}")
    
@router.get("/download-result/{job_id}")
async def download_result(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job['status']}")
    
    if "result_content" not in job:
        raise HTTPException(status_code=404, detail="Result not found")
    
    filename = job.get("filename", f"sentiment_analysis_result.csv")

    # 💥 Do not re-encode! Just add BOM at the start
    bom = b'\xef\xbb\xbf'
    content_with_bom = bom + job["result_content"].encode("utf-8")

    return Response(
        content=content_with_bom,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result_path=job.get("result_path")
    )

@router.get("/models")
async def list_models():
    models = [
        {
            "id": "vader",
            "name": "VADER",
            "description": "Rule-based sentiment analyzer optimized for social media",
            "strengths": ["Fast", "No dependencies", "Works well with short texts", "Handles emojis and slang"],
            "weaknesses": ["Less accurate on formal text", "Limited context understanding"]
        },
        {
            "id": "transformer",
            "name": "DistilBERT",
            "description": "Transformer-based deep learning model fine-tuned for sentiment",
            "strengths": ["Higher accuracy", "Better context understanding", "Good with complex sentences"],
            "weaknesses": ["Slower", "Requires more resources", "May not handle emojis well"]
        },
        {
            "id": "textblob",
            "name": "TextBlob",
            "description": "Simple, pattern-based sentiment analyzer",
            "strengths": ["Simple", "Fast", "Easy to understand"],
            "weaknesses": ["Less accurate than modern methods", "Limited vocabulary"]
        },
        {
            "id": "ensemble",
            "name": "Ensemble (Recommended)",
            "description": "Combines multiple models for better overall accuracy",
            "strengths": ["Most balanced approach", "Handles a wide range of text types", "More robust"],
            "weaknesses": ["Slower than single models", "More resource intensive"]
        }
    ]
    
    return {"models": models}

@router.get("/stats")
async def get_stats():
    try:
        return {
            "total_jobs": len(job_store),
            "completed_jobs": sum(1 for job in job_store.values() if job["status"] == "completed"),
            "failed_jobs": sum(1 for job in job_store.values() if job["status"] == "failed"),
            "processing_jobs": sum(1 for job in job_store.values() if job["status"] == "processing"),
            "queued_jobs": sum(1 for job in job_store.values() if job["status"] == "queued")
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")

@router.post("/compare-models")
async def compare_models(input_data: TextInput):
    try:
        models = ["vader", "transformer", "textblob", "ensemble"]
        results = {}
        
        for model in models:
            results[model] = analyze_sentiment(input_data.text, model=model)
        
        return {
            "text": input_data.text,
            "results": {
                model: {
                    "sentiment_label": results[model]["sentiment_label"],
                    "compound": results[model]["sentiment"]["compound"],
                    "confidence": results[model]["confidence"]
                } for model in models
            },
            "recommendation": "ensemble"  # Default recommendation
        }
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")
        
@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    try:
        if job_id not in job_store:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Delete result file if it exists
        if "result_path" in job_store[job_id] and job_store[job_id]["result_path"]:
            try:
                result_path = job_store[job_id]["result_path"]
                if os.path.exists(result_path):
                    os.remove(result_path)
                    logger.info(f"Deleted result file: {result_path}")
            except Exception as e:
                logger.warning(f"Could not delete result file: {str(e)}")
        
        # Remove from job store
        del job_store[job_id]
        
        return {"message": f"Job {job_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting job: {str(e)}")

@router.get("/emotions")
async def get_emotion_info():
    """
    Returns information about the emotions detected by the API
    """
    return {
        "emotions": {
            "joy": {
                "description": "Feelings of happiness, pleasure, or contentment",
                "indicators": ["positive polarity", "exclamation marks", "happy/joyful terms"]
            },
            "sadness": {
                "description": "Feelings of sorrow, grief, or unhappiness",
                "indicators": ["negative polarity", "sad/unhappy terms"]
            },
            "anger": {
                "description": "Feelings of annoyance, frustration, or rage",
                "indicators": ["negative polarity with high subjectivity", "angry terms"]
            },
            "fear": {
                "description": "Feelings of worry, anxiety, or terror",
                "indicators": ["fear-related words", "anxious sentiment"]
            },
            "surprise": {
                "description": "Reaction to unexpected events or information",
                "indicators": ["exclamation marks", "question marks", "terms indicating shock"]
            }
        },
        "methodology": "Emotion detection uses a rule-based approach combined with polarity and subjectivity scores. For more accurate emotion detection, consider using a specialized emotion detection model."
    }

@router.get("/languages")
async def get_supported_languages():
    """
    Returns information about language detection and supported languages
    """
    return {
        "primary_support": ["en"],
        "limited_support": ["fr", "es", "de", "it"],
        "notes": "Language detection is currently based on simple heuristics and best suited for English. For production use with multilingual content, consider integrating a dedicated language detection library like langdetect.",
        "language_handling": "When non-English texts are detected, sentiment analysis will still be applied but may be less accurate than with English content."
    }

@router.post("/clean-jobs")
async def clean_old_jobs(days: int = Query(7, description="Clean jobs older than this many days")):
    """
    Clean up old jobs and their result files
    """
    try:
        if not 1 <= days <= 90:
            raise HTTPException(status_code=400, detail="Days must be between 1 and 90")
        
        current_time = datetime.now()
        deleted_count = 0
        file_count = 0
        
        job_ids_to_delete = []
        
        for job_id, job_data in job_store.items():
            try:
                # Check if job is old enough to be deleted
                created_at = datetime.fromisoformat(job_data.get("created_at", ""))
                age_days = (current_time - created_at).days
                
                if age_days >= days:
                    # Delete result file
                    if "result_path" in job_data and job_data["result_path"]:
                        result_path = job_data["result_path"]
                        if os.path.exists(result_path):
                            os.remove(result_path)
                            file_count += 1
                    
                    job_ids_to_delete.append(job_id)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Error processing job {job_id} for cleanup: {str(e)}")
        
        # Remove jobs from store
        for job_id in job_ids_to_delete:
            del job_store[job_id]
        
        return {
            "message": f"Cleaned up {deleted_count} jobs older than {days} days",
            "deleted_jobs": deleted_count,
            "deleted_files": file_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning jobs: {str(e)}")

@router.post("/feedback")
async def submit_feedback(
    text: str = Form(...),
    predicted_sentiment: str = Form(...),
    correct_sentiment: str = Form(...),
    model: str = Form("vader"),
    notes: str = Form(None)
):
    """
    Submit feedback for sentiment analysis predictions to help improve the system
    """
    try:
        # Validate inputs
        if predicted_sentiment not in ["positive", "neutral", "negative"]:
            raise HTTPException(status_code=400, detail="Invalid predicted sentiment")
        
        if correct_sentiment not in ["positive", "neutral", "negative"]:
            raise HTTPException(status_code=400, detail="Invalid correct sentiment")
        
        if model not in ["vader", "transformer", "textblob", "ensemble"]:
            raise HTTPException(status_code=400, detail="Invalid model name")
        
        # Create output directory if it doesn't exist
        os.makedirs("feedback", exist_ok=True)
        
        # Create feedback entry
        feedback = {
            "text": text,
            "predicted_sentiment": predicted_sentiment,
            "correct_sentiment": correct_sentiment,
            "model": model,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        # Append to feedback file
        feedback_file = "feedback/sentiment_feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback) + "\n")
        
        return {
            "message": "Thank you for your feedback!",
            "feedback_id": datetime.now().strftime("%Y%m%d%H%M%S")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@router.get("/docs/examples")
async def get_documentation_examples():
    """
    Returns example API usage to help users get started
    """
    return {
        "simple_analysis": {
            "endpoint": "/sentiment/analyze",
            "method": "POST",
            "body": {
                "text": "I really enjoyed the movie! The acting was superb.",
                "model": "vader",
                "include_aspects": True
            },
            "description": "Analyze a single text with aspect-based sentiment"
        },
        "batch_analysis": {
            "endpoint": "/sentiment/analyze-batch",
            "method": "POST",
            "body": {
                "texts": [
                    "The customer service was excellent!",
                    "The product broke after one day of use.",
                    "It's an okay product, not great but not terrible."
                ],
                "model": "ensemble",
                "include_aspects": True
            },
            "description": "Analyze multiple texts at once"
        },
        "csv_analysis": {
            "endpoint": "/sentiment/analyze-csv",
            "method": "POST",
            "form_data": {
                "file": "(your CSV file)",
                "text_column": "review_text",
                "model": "ensemble",
                "include_aspects": "true",
                "batch_size": "500"
            },
            "description": "Process a CSV file with one text per row"
        },
        "model_comparison": {
            "endpoint": "/sentiment/compare-models",
            "method": "POST",
            "body": {
                "text": "This restaurant has amazing food but terrible service",
                "include_aspects": True
            },
            "description": "Compare results across all available models"
        }
    }

# Optional: Advanced analysis endpoints

@router.post("/analyze-trends")
async def analyze_sentiment_trends(
    file: UploadFile = File(...),
    text_column: str = Form("text"),
    date_column: str = Form("date"),
    date_format: str = Form("%Y-%m-%d"),
    model: str = Form("vader")
):
    """
    Analyze sentiment trends over time from a CSV file
    """
    try:
        content = await file.read()
        
        # Read CSV
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        
        # Validate columns
        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain a '{text_column}' column")
        
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV must contain a '{date_column}' column")
        
        # Convert date column
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Could not parse dates with format '{date_format}'")
        
        # Sort by date
        df = df.sort_values(by=date_column)
        
        # Apply sentiment analysis
        sentiment_results = []
        for text in df[text_column]:
            result = analyze_sentiment(text, model=model)
            sentiment_results.append(result)
        
        # Extract sentiment scores
        df['sentiment_label'] = [r["sentiment_label"] for r in sentiment_results]
        df['compound'] = [r["sentiment"]["compound"] for r in sentiment_results]
        
        # Group by date and calculate average
        daily_sentiment = df.groupby(df[date_column].dt.date)['compound'].mean().reset_index()
        daily_sentiment.columns = ['date', 'average_sentiment']
        
        # Count sentiment labels by date
        sentiment_counts = df.groupby([df[date_column].dt.date, 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        sentiment_counts.columns = ['date'] + list(sentiment_counts.columns[1:])
        
        # Create a full result
        result = pd.merge(daily_sentiment, sentiment_counts, on='date')
        
        # Convert to dict for JSON response
        trend_data = result.to_dict(orient='records')
        
        return {
            "trend_data": trend_data,
            "total_analyzed": len(df),
            "date_range": {
                "start": df[date_column].min().strftime("%Y-%m-%d"),
                "end": df[date_column].max().strftime("%Y-%m-%d")
            },
            "overall_sentiment": {
                "average_compound": df['compound'].mean(),
                "sentiment_distribution": {
                    "positive": (df['sentiment_label'] == 'positive').mean(),
                    "neutral": (df['sentiment_label'] == 'neutral').mean(),
                    "negative": (df['sentiment_label'] == 'negative').mean()
                }
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing sentiment trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment trends: {str(e)}")

@router.post("/analyze-advanced")
async def analyze_advanced(input_data: TextInput):
    """
    Provide more advanced sentiment analysis with additional metrics
    """
    try:
        # Basic sentiment analysis
        result = analyze_sentiment(input_data.text, model=input_data.model, include_aspects=True)
        
        # Get additional metrics
        
        # Word count and complexity metrics
        words = re.findall(r'\b\w+\b', input_data.text)
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Sentiment intensity
        intensity = abs(result["sentiment"]["compound"])
        
        # Mixed sentiment detection
        mixed_sentiment = (
            result["sentiment"]["pos"] > 0.2 and 
            result["sentiment"]["neg"] > 0.2
        )
        
        # Subjectivity detection (using TextBlob)
        blob = TextBlob(input_data.text)
        subjectivity = blob.sentiment.subjectivity
        
        # Extract entities if spaCy is available
        entities = []
        entity_sentiments = {}
        if nlp:
            doc = nlp(input_data.text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
                
                # Get sentiment for context around entity
                start_idx = max(0, ent.start - 5)
                end_idx = min(len(doc), ent.end + 5)
                context = doc[start_idx:end_idx].text
                entity_sentiment = sentiment_analyzer_vader.polarity_scores(context)
                entity_sentiments[ent.text] = {
                    "compound": entity_sentiment["compound"],
                    "label": get_sentiment_label(entity_sentiment["compound"])
                }
        
        return {
            "basic_sentiment": {
                "text": input_data.text,
                "sentiment": result["sentiment"],
                "sentiment_label": result["sentiment_label"],
                "confidence": result["confidence"]
            },
            "aspects": result.get("aspects", []),
            "emotions": result["emotion_analysis"],
            "advanced_metrics": {
                "word_count": word_count,
                "avg_word_length": avg_word_length,
                "sentiment_intensity": intensity,
                "has_mixed_sentiment": mixed_sentiment,
                "subjectivity": subjectivity,
            },
            "entities": entities,
            "entity_sentiments": entity_sentiments
        }
    except Exception as e:
        logger.error(f"Error in advanced analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in advanced analysis: {str(e)}")