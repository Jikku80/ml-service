import io
import re
import tempfile
from typing import List, Optional
import pypdf  # Changed from PyPDF2 to pypdf
from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(
    prefix="/cv",
    tags=["cv"],
    responses={404: {"description": "Not found"}},
)

class CandidateMatch(BaseModel):
    file_name: str
    match_score: float
    key_skills_matched: List[str]
    missing_skills: List[str]
    experience_years: Optional[float] = None

class CandidateRanking(BaseModel):
    job_title: str
    matches: List[CandidateMatch]

def extract_text_from_pdf(pdf_content):
    """Extract text content from PDF bytes."""
    try:
        # Create BytesIO object from PDF content
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = pypdf.PdfReader(pdf_file)  # Updated to pypdf.PdfReader
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_skills(text, skill_keywords):
    """Extract skills from text based on a predefined list of skills."""
    found_skills = []
    for skill in skill_keywords:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    return found_skills

def extract_experience_years(text):
    """Extract years of experience from CV text."""
    experience_patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience\s*(?:of|:)?\s*(\d+)\+?\s*years?',
        r'worked\s*(?:for)?\s*(\d+)\+?\s*years?'
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the highest number of years mentioned
            return max([int(year) for year in matches])
    return None

@router.post("/upload-job-description/", response_model=dict)
async def upload_job_description(
    job_title: str = Form(...),
    job_description: UploadFile = File(...),
    required_skills: str = Form(...)
):
    """Upload and store a job description."""
    job_desc_content = await job_description.read()
    job_desc_text = extract_text_from_pdf(job_desc_content)
    
    # Parse the required skills
    skills_list = [skill.strip() for skill in required_skills.split(',')]
    
    # Store the job description in a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as job_file:
        job_file.write(job_desc_text.encode())
        job_path = job_file.name
    
    return {
        "job_title": job_title,
        "job_description_path": job_path,
        "required_skills": skills_list,
        "message": "Job description uploaded successfully"
    }

@router.post("/match-cvs/", response_model=CandidateRanking)
async def match_cvs(
    job_title: str = Form(...),
    job_description_path: str = Form(...),
    required_skills: str = Form(...),
    min_experience: Optional[int] = Form(None),
    cv_files: List[UploadFile] = File(...)
):
    """Match uploaded CVs against a job description."""
    # Load the job description
    with open(job_description_path, 'r') as f:
        job_desc_text = f.read()
    
    # Parse the required skills
    skills_list = [skill.strip() for skill in required_skills.split(',')]
    
    # Process each CV
    all_texts = [job_desc_text]  # Start with job description
    file_names = ["job_description"]
    
    # Extract text from all CVs first
    cv_texts = []
    for cv_file in cv_files:
        cv_content = await cv_file.read()
        cv_text = extract_text_from_pdf(cv_content)
        cv_texts.append(cv_text)
        all_texts.append(cv_text)
        file_names.append(cv_file.filename)
    
    # Calculate TF-IDF vectors and similarity scores
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between job description and each CV
    cos_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Create match objects for each CV
    matches = []
    for i, (cv_file, cv_text, similarity) in enumerate(zip(cv_files, cv_texts, cos_similarities)):
        # Extract skills
        matched_skills = extract_skills(cv_text, skills_list)
        missing_skills = [skill for skill in skills_list if skill not in matched_skills]
        
        # Extract years of experience
        experience_years = extract_experience_years(cv_text)
        
        # Skip candidates with insufficient experience if specified
        if min_experience is not None and (experience_years is None or experience_years < min_experience):
            continue
        
        match = CandidateMatch(
            file_name=cv_file.filename,
            match_score=float(similarity),
            key_skills_matched=matched_skills,
            missing_skills=missing_skills,
            experience_years=experience_years
        )
        matches.append(match)
    
    # Sort matches by score (descending)
    matches.sort(key=lambda x: x.match_score, reverse=True)
    
    return CandidateRanking(
        job_title=job_title,
        matches=matches
    )