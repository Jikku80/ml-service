import io
import re
import os
import json
import logging
import shutil
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime, time

import nltk
import pypdf
import spacy
import pytz
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLP components
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()

# Load spaCy model for NER and better text processing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cv_matcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("cv_matcher")

# Database models and dependencies would be imported here
# from database import get_db, Job, CV, MatchResult

# Router configuration
router = APIRouter(
    prefix="/cv",
    tags=["cv"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    },
)

# Pydantic models
class Skill(BaseModel):
    name: str
    category: Optional[str] = None
    weight: float = 1.0

class EducationDetail(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[int] = None
    field_of_study: Optional[str] = None

class WorkExperience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[float] = None  # in years
    description: Optional[str] = None

class CandidateProfile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    education: List[EducationDetail] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    total_experience_years: Optional[float] = None
    skills: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

class CandidateMatch(BaseModel):
    file_name: str
    match_score: float
    profile: CandidateProfile
    key_skills_matched: List[str]
    missing_skills: List[str]
    experience_match: bool = True
    education_match: bool = True

class JobRequirements(BaseModel):
    title: str
    required_skills: List[str]
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience: Optional[float] = None
    education_level: Optional[str] = None

class CandidateRanking(BaseModel):
    job_title: str
    requirements: JobRequirements
    matches: List[CandidateMatch]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))

class JobDescription(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    requirements: JobRequirements
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))

class AnalysisResult(BaseModel):
    parsed_text: str
    total_words: int
    skills_found: List[str]
    education: List[EducationDetail]
    experience: List[WorkExperience]
    total_experience: Optional[float] = None

# Helper classes and functions
class TextProcessor:
    """Class for processing and analyzing text content."""
    
    def __init__(self, skill_keywords: List[str]):
        self.skill_keywords = skill_keywords
        self.education_patterns = [
            r'(?:(?:Bachelor|Master|PhD|Doctorate|BSc|MSc|BA|MA|MBA|MD|BBA|BEng|MEng|B\.A\.|M\.A\.|B\.S\.|M\.S\.|Ph\.D\.)\s+(?:of|in|on)?\s+[A-Za-z\s]+)',
            r'(?:University|College|Institute|School)\s+of\s+[A-Za-z\s]+',
        ]
        # Add more patterns as needed for education extraction
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text by removing extra whitespace and special characters."""
        # Replace multiple spaces, newlines, and tabs with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s@.,-:;]', '', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing stopwords."""
        try:
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token.isalpha() and token not in STOPWORDS]
        except:
            # Fallback to simple tokenization if nltk fails
            words = text.lower().split()
            return [word for word in words if word.isalpha() and word not in STOPWORDS]
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text based on a predefined list of skills."""
        found_skills = []
        clean_text = self.clean_text(text.lower())
        
        for skill in self.skill_keywords:
            skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(skill_pattern, clean_text):
                found_skills.append(skill)
        
        return found_skills
    
    def extract_education(self, text: str) -> List[EducationDetail]:
        """Extract education details from text."""
        education_details = []
        
        # If spaCy is available, use it for better entity recognition
        if nlp:
            doc = nlp(text)
            # Process entities that might be educational institutions
            # This is a simplified approach; would need refinement in production
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Check if it might be an educational institution
                    context = text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)]
                    if re.search(r'\b(university|college|institute|school|degree|bachelor|master|phd)\b', context, re.IGNORECASE):
                        education_details.append(EducationDetail(
                            institution=ent.text,
                            # Extract other details like degree, year, etc.
                        ))
        
        # Use regex patterns as fallback or additional method
        for pattern in self.education_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract the education information - this is simplified
                education_text = match.group(0)
                # Try to extract degree, institution, year, etc.
                
                education_details.append(EducationDetail(
                    institution=education_text,
                    # Other fields would be extracted with more specific patterns
                ))
        
        return education_details
    
    def extract_experience_years(self, text: str) -> Optional[float]:
        """Extract years of experience from CV text."""
        string_to_int = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20
        }
        
        # Patterns for numeric years
        numeric_patterns = [
            r'(\d+[\.\d]*)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*(?:of|:)?\s*(\d+[\.\d]*)\+?\s*years?',
            r'worked\s*(?:for)?\s*(\d+[\.\d]*)\+?\s*years?',
            r'(\d+[\.\d]*)\+?\s*years?\s+in\s+(?:the\s+)?industry'
        ]
        
        # Patterns for text-based years
        text_patterns = [
            r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s*(?:of|:)?\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\+?\s*years?',
            r'worked\s*(?:for)?\s*(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\+?\s*years?'
        ]
        
        years = []
        
        # Extract numeric years
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years.append(float(match))
                except ValueError:
                    continue
        
        # Extract text-based years
        for pattern in text_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower()
                if match_lower in string_to_int:
                    years.append(string_to_int[match_lower])
        
        # Return the highest number of years mentioned, or None if no years were found
        return max(years) if years else None
    
    def extract_work_experience(self, text: str) -> List[WorkExperience]:
        """Extract work experience details from text."""
        experiences = []
        
        # This is a simplified approach - in production, you would use more sophisticated
        # techniques like segmentation and contextual analysis
        
        # If spaCy is available, use it for entity recognition
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    # Look for potential job titles near organization entities
                    context = text[max(0, ent.start_char - 100):min(len(text), ent.end_char + 100)]
                    
                    # This is a simplistic approach to find job titles
                    title_match = re.search(r'(?:(?:Senior|Junior|Lead|Chief|Principal|Associate)\s+)?(?:Developer|Engineer|Manager|Director|Analyst|Designer|Architect|Consultant|Specialist)', context, re.IGNORECASE)
                    
                    if title_match:
                        experiences.append(WorkExperience(
                            company=ent.text,
                            title=title_match.group(0)
                            # Other details would need more complex extraction logic
                        ))
        
        # Add regex-based extraction as fallback or enhancement
        # This would ideally identify job titles, companies, dates, etc.
        
        return experiences
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from text."""
        contact_info = {}
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact_info['email'] = email_match.group(0)
        
        # Extract phone number (various formats)
        phone_patterns = [
            r'\b\+?[0-9]{1,3}[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            r'\b\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                contact_info['phone'] = phone_match.group(0)
                break
        
        # Extract name (This is highly complex and might require more sophisticated NLP)
        # A very basic approach might be to look at the beginning of the document
        # but this would need significant refinement in a real system
        
        return contact_info
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """Perform comprehensive analysis of document text."""
        cleaned_text = self.clean_text(text)
        
        # Extract all relevant information
        skills = self.extract_skills(cleaned_text)
        education = self.extract_education(cleaned_text)
        experience_details = self.extract_work_experience(cleaned_text)
        experience_years = self.extract_experience_years(cleaned_text)
        
        return AnalysisResult(
            parsed_text=cleaned_text,
            total_words=len(cleaned_text.split()),
            skills_found=skills,
            education=education,
            experience=experience_details,
            total_experience=experience_years
        )

class PDFProcessor:
    """Class for handling PDF documents."""
    
    @staticmethod
    def extract_text(pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + " "
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    @staticmethod
    def get_metadata(pdf_content: bytes) -> Dict[str, Any]:
        """Extract metadata from PDF file."""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = pypdf.PdfReader(pdf_file)
            
            # Extract available metadata
            metadata = {}
            if pdf_reader.metadata:
                for key, value in pdf_reader.metadata.items():
                    if key.startswith('/'):
                        clean_key = key[1:]  # Remove the leading slash
                        metadata[clean_key] = value
            
            # Add basic document info
            metadata['pages'] = len(pdf_reader.pages)
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {e}")
            return {"error": str(e)}

class CVMatcher:
    """Class for matching CVs against job descriptions."""
    
    def __init__(self, job_requirements: JobRequirements):
        self.job_requirements = job_requirements
        self.text_processor = TextProcessor(
            skill_keywords=job_requirements.required_skills + job_requirements.preferred_skills
        )
    
    def calculate_similarity(self, job_desc_text: str, cv_texts: List[str]) -> List[float]:
        """Calculate TF-IDF based similarity between job description and CVs."""
        all_texts = [job_desc_text] + cv_texts
        
        # Create and configure vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            max_features=10000,
            min_df=1
        )
        
        # Calculate TF-IDF vectors and similarity scores
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            cos_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            return cos_similarities.tolist()
        except Exception as e:
            logger.error(f"Error in similarity calculation: {e}")
            return [0.0] * len(cv_texts)  # Return zeros in case of error
    
    def analyze_cv(self, cv_text: str) -> CandidateProfile:
        """Extract comprehensive profile information from CV text."""
        analysis = self.text_processor.analyze_text(cv_text)
        contact_info = self.text_processor.extract_contact_info(cv_text)
        
        profile = CandidateProfile(
            name=contact_info.get('name'),
            email=contact_info.get('email'),
            phone=contact_info.get('phone'),
            location=contact_info.get('location'),
            education=analysis.education,
            work_experience=analysis.experience,
            total_experience_years=analysis.total_experience,
            skills=analysis.skills_found
        )
        
        return profile
    
    def match_cv(self, cv_text: str, file_name: str, similarity_score: float) -> CandidateMatch:
        """Match a single CV against job requirements."""
        # Analyze the CV to extract profile information
        profile = self.analyze_cv(cv_text)
        
        # Extract matching and missing skills
        matched_skills = [skill for skill in profile.skills if skill in self.job_requirements.required_skills]
        missing_skills = [skill for skill in self.job_requirements.required_skills if skill not in profile.skills]
        
        # Check experience requirements
        experience_match = True
        if self.job_requirements.min_experience is not None:
            if profile.total_experience_years is None or profile.total_experience_years < self.job_requirements.min_experience:
                experience_match = False
        
        # Check education requirements (simplified - would need more sophisticated logic in production)
        education_match = True
        if self.job_requirements.education_level is not None:
            education_level_found = False
            for edu in profile.education:
                if edu.degree and self.job_requirements.education_level.lower() in edu.degree.lower():
                    education_level_found = True
                    break
            education_match = education_level_found
        
        return CandidateMatch(
            file_name=file_name,
            match_score=similarity_score,
            profile=profile,
            key_skills_matched=matched_skills,
            missing_skills=missing_skills,
            experience_match=experience_match,
            education_match=education_match
        )

class Storage:
    """Class for handling storage of files and data with user-specific folders."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        
        # Base directories will be created as needed when a user is set
        self.current_user_id = None
        self.user_dir = None
        self.jobs_dir = None
        self.cvs_dir = None
        self.results_dir = None
    
    def set_user(self, user_id: str) -> None:
        """Set current user and initialize their directories with 'cv_' prefix."""
        self.current_user_id = user_id
        self.user_dir = self.base_dir / f"cv_{user_id}"
        self.jobs_dir = self.user_dir / "jobs"
        self.cvs_dir = self.user_dir / "cvs"
        self.results_dir = self.user_dir / "results"
        
        # Create user's directories if they don't exist
        for directory in [self.base_dir, self.user_dir, self.jobs_dir, self.cvs_dir, self.results_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def _ensure_user_set(self) -> None:
        """Ensure a user is set before performing operations."""
        if self.current_user_id is None:
            raise ValueError("User ID must be set before performing storage operations. Call set_user() first.")
    
    async def save_job_description(self, job_title: str, content: str) -> str:
        """Save job description content to file and return path."""
        self._ensure_user_set()
        
        file_name = f"{self.current_user_id}.txt"
        file_path = self.jobs_dir / file_name
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(file_path)
    
    async def save_cv(self, file_name: str, content: bytes) -> str:
        """Save CV content to file and return path."""
        self._ensure_user_set()
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Extract file extension and create new name
        _, ext = os.path.splitext(file_name)
        new_file_name = f"{file_name.replace(ext, '')}_{timestamp}{ext}"
        file_path = self.cvs_dir / new_file_name
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        return str(file_path)
    
    async def save_match_results(self, job_title: str, results: Dict[str, Any]) -> str:
        """Save matching results to JSON file and return path."""
        self._ensure_user_set()
        
        file_name = f"results_{self.current_user_id}.json"
        file_path = self.results_dir / file_name
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, default=str, indent=2)
        
        return str(file_path)
    
    async def cleanup_user_data(self, user_id: Optional[str] = None) -> bool:
        """Remove a user's folder and all its contents.

        Args:
            user_id: Optional user ID to cleanup. If None, uses current user.
            
        Returns:
            bool: True if cleanup was successful, False otherwise.
        """
        target_user_id = user_id or self.current_user_id

        if not target_user_id:
            raise ValueError("No user ID specified for cleanup.")

        target_dir = self.base_dir / f"cv_{target_user_id}"

        if target_dir.exists():
            try:
                shutil.rmtree(target_dir)
                # Reset current user attributes if we're removing the current user
                if target_user_id == self.current_user_id:
                    self.current_user_id = None
                    self.user_dir = None
                    self.jobs_dir = None
                    self.cvs_dir = None
                    self.results_dir = None
                return True
            except Exception as e:
                print(f"Error during cleanup: {e}")
                return False
        return False

# Initialize storage
storage = Storage()

# API Endpoints
@router.post("/{user_id}/upload-job-description/", response_model=Dict[str, Any])
async def upload_job_description(
    user_id: str,
    job_title: str = Form(...),
    job_description: UploadFile = File(...),
    required_skills: str = Form(...),
    preferred_skills: str = Form(""),
    min_experience: Optional[float] = Form(None),
    education_level: Optional[str] = Form(None)
):
    """Upload and store a job description with detailed requirements."""
    try:
        storage.set_user(user_id)
        job_desc_content = await job_description.read()
        
        # Extract text from PDF
        if job_description.content_type == "application/pdf":
            job_desc_text = PDFProcessor.extract_text(job_desc_content)
        else:
            # Assume plain text if not PDF
            job_desc_text = job_desc_content.decode('utf-8')
        
        # Parse the skills
        required_skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
        preferred_skills_list = [skill.strip() for skill in preferred_skills.split(',') if skill.strip()]
        
        # Create job requirements object
        job_reqs = JobRequirements(
            title=job_title,
            required_skills=required_skills_list,
            preferred_skills=preferred_skills_list,
            min_experience=min_experience,
            education_level=education_level
        )
        
        # Store the job description
        job_desc_path = await storage.save_job_description(job_title, job_desc_text)
        
        # Create full job description object
        job_desc = JobDescription(
            title=job_title,
            description=job_desc_text,
            requirements=job_reqs
        )
        
        # In a real application, we would save this to a database
        # db.add(job_desc)
        # db.commit()
        
        return {
            "job_title": job_title,
            "job_description_path": job_desc_path,
            "requirements": job_reqs.model_dump(),
            "message": "Job description uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading job description: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing job description: {str(e)}"
        )

@router.post("/analyze-cv/", response_model=AnalysisResult)
async def analyze_cv(
    cv_file: UploadFile = File(...),
    skill_keywords: str = Form("")
):
    """Analyze a CV without matching it to a job description."""
    try:
        cv_content = await cv_file.read()
        
        # Extract text from PDF
        if cv_file.content_type == "application/pdf":
            cv_text = PDFProcessor.extract_text(cv_content)
        else:
            # Assume plain text if not PDF
            cv_text = cv_content.decode('utf-8')
        
        # Parse skill keywords
        skills_list = [skill.strip() for skill in skill_keywords.split(',') if skill.strip()]
        
        # Create text processor and analyze CV
        text_processor = TextProcessor(skill_keywords=skills_list)
        analysis = text_processor.analyze_text(cv_text)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing CV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing CV: {str(e)}"
        )

@router.post("/{user_id}/match-cvs/", response_model=CandidateRanking)
async def match_cvs(
    user_id: str,
    background_tasks: BackgroundTasks,
    job_title: str = Form(...),
    job_description_path: str = Form(...),
    required_skills: str = Form(...),
    preferred_skills: str = Form(""),
    min_experience: Optional[float] = Form(None),
    education_level: Optional[str] = Form(None),
    cv_files: List[UploadFile] = File(...)
):
    """Match uploaded CVs against a job description with detailed analysis."""
    try:
        storage.set_user(user_id)
        # Load the job description
        with open(job_description_path, 'r', encoding='utf-8') as f:
            job_desc_text = f.read()
        
        # Parse the skills
        required_skills_list = [skill.strip() for skill in required_skills.split(',') if skill.strip()]
        preferred_skills_list = [skill.strip() for skill in preferred_skills.split(',') if skill.strip()]
        
        # Create job requirements
        job_reqs = JobRequirements(
            title=job_title,
            required_skills=required_skills_list,
            preferred_skills=preferred_skills_list,
            min_experience=min_experience,
            education_level=education_level
        )
        
        # Initialize CV matcher
        cv_matcher = CVMatcher(job_requirements=job_reqs)
        
        # Process each CV
        all_texts = [job_desc_text]  # Start with job description
        cv_texts = []
        cv_file_names = []
        
        # Extract text from all CVs first
        for cv_file in cv_files:
            cv_content = await cv_file.read()
            
            # Extract text based on file type
            if cv_file.filename.lower().endswith('.pdf'):
                cv_text = PDFProcessor.extract_text(cv_content)
            else:
                # Assume plain text for other file types
                cv_text = cv_content.decode('utf-8')
            
            cv_texts.append(cv_text)
            all_texts.append(cv_text)
            cv_file_names.append(cv_file.filename)
            
            # Save CV file (in a real app, we'd do this asynchronously)
            # background_tasks.add_task(storage.save_cv, cv_file.filename, cv_content)
        
        # Calculate similarities between job description and CVs
        similarities = cv_matcher.calculate_similarity(job_desc_text, cv_texts)
        
        # Create match objects for each CV
        matches = []
        for i, (file_name, cv_text, similarity) in enumerate(zip(cv_file_names, cv_texts, similarities)):
            # Match the CV
            match = cv_matcher.match_cv(cv_text, file_name, similarity)
            
            # Add to matches if it meets basic criteria
            if match.experience_match and match.education_match:
                matches.append(match)
        
        # Sort matches by score (descending)
        matches.sort(key=lambda x: x.match_score, reverse=True)
        
        # Create ranking result
        ranking = CandidateRanking(
            job_title=job_title,
            requirements=job_reqs,
            matches=matches
        )
        
        # Save results in the background
        background_tasks.add_task(
            storage.save_match_results,
            job_title,
            ranking.model_dump()
        )
        
        return ranking
    
    except Exception as e:
        logger.error(f"Error matching CVs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in CV matching process: {str(e)}"
        )
    
@router.delete("/{user_id}/erase")
async def erase_files(user_id:str):
    try:
        success = await storage.cleanup_user_data(user_id)
        if success:
            return {"status": "deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User data not found or could not be deleted."
            )
    except Exception as e:
        logger.error(f"Error Erasing Files: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error retrieving job listings: {str(e)}"
    )


@router.get("/{user_id}/jobs/", response_model=List[Dict[str, Any]])
async def list_jobs(user_id:str):
    """List all stored job descriptions."""
    try:
        storage.set_user(user_id)

        jobs = []
        for file_path in storage.jobs_dir.glob("*.txt"):
            # Extract job title from filename
            job_title = file_path.stem.split('_')[0].replace('_', ' ')
            
            # In a real app, we'd retrieve this from a database
            jobs.append({
                "title": job_title,
                "path": str(file_path),
                "created_at": datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        
        return jobs
    
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving job listings: {str(e)}"
        )

@router.get("/{user_id}/results/", response_model=List[Dict[str, Any]])
async def list_results(user_id: str):
    """List all stored matching results."""
    try:
        storage.set_user(user_id)

        results = []
        for file_path in storage.results_dir.glob("*.json"):
            # Extract job title from filename
            job_title = file_path.stem.split('_')[0].replace('_', ' ')
            
            results.append({
                "job_title": job_title,
                "path": str(file_path),
                "created_at": datetime.fromtimestamp(file_path.stat().st_mtime)
            })
        
        return results
    
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving results: {str(e)}"
        )

@router.get("/{user_id}/result/{result_id}", response_model=CandidateRanking)
async def get_result(user_id:str, result_id: str):
    """Get a specific matching result by ID."""
    try:
        storage.set_user(user_id)

        file_path = storage.results_dir / f"{result_id}.json"
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Result with ID {result_id} not found"
            )
        
        with open(file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            
            # Convert the loaded JSON to a CandidateRanking object
            return CandidateRanking(**result_data)
    
    except Exception as e:
        logger.error(f"Error retrieving result: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving result: {str(e)}"
        )

@router.post("/bulk-upload/", response_model=Dict[str, Any])
async def bulk_upload_cvs(
    background_tasks: BackgroundTasks,
    job_id: str = Form(...),
    cv_files: List[UploadFile] = File(...)
):
    """Bulk upload CVs for later processing."""
    try:
        # In a real system, we'd verify the job_id exists in the database
        upload_results = []
        
        for cv_file in cv_files:
            cv_content = await cv_file.read()
            
            # Save the CV file
            file_path = await storage.save_cv(cv_file.filename, cv_content)
            
            # Add to results
            upload_results.append({
                "filename": cv_file.filename,
                "path": file_path,
                "size": len(cv_content)
            })
            
            # Add task to process the CV in the background
            background_tasks.add_task(
                process_cv_background,
                job_id,
                file_path
            )
        
        return {
            "job_id": job_id,
            "uploaded_files": len(upload_results),
            "files": upload_results,
            "message": "Files uploaded successfully and queued for processing"
        }
    
    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )

# Background task for CV processing
async def process_cv_background(job_id: str, cv_path: str):
    """Process a CV in the background and update results."""
    try:
        # In a real system, this would:
        # 1. Load the job description from the database by job_id
        # 2. Process the CV and match it against the job
        # 3. Update the database with results
        # 4. Potentially send notifications
        
        logger.info(f"Background processing of {cv_path} for job {job_id}")
        
        # Simulate processing time
        import time
        time.sleep(2)
        
        # Update processing status
        logger.info(f"Completed processing {cv_path} for job {job_id}")
    
    except Exception as e:
        logger.error(f"Error in background CV processing: {e}")

@router.post("/smart-match/", response_model=Dict[str, Any])
async def smart_match_candidates(
    job_id: str = Form(...),
    top_n: int = Form(10),
    min_score: float = Form(0.5)
):
    """Advanced matching with customizable parameters and optimization."""
    try:
        # In a real system, we would:
        # 1. Load job details from database
        # 2. Apply enhanced matching algorithms with weighting
        # 3. Filter and rank candidates based on custom criteria
        
        return {
            "job_id": job_id,
            "matches_found": top_n,
            "message": "Smart matching completed",
            "results_url": f"/cv/results/{job_id}"
        }
    
    except Exception as e:
        logger.error(f"Error in smart matching: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during smart matching: {str(e)}"
        )

@router.post("/extract-skills/", response_model=Dict[str, Any])
async def extract_skills_from_job(
    job_description: str = Form(...),
    custom_skills: Optional[str] = Form(None)
):
    """Extract skills from job description text."""
    try:
        # Load common skills library
        common_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "TypeScript",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
            "SQL", "PostgreSQL", "MySQL", "MongoDB", "AWS", "Azure",
            "Docker", "Kubernetes", "Git", "Agile", "Scrum", "DevOps",
            "Machine Learning", "Data Science", "AI", "Deep Learning",
            "TensorFlow", "PyTorch", "NLP", "Computer Vision",
            "REST API", "GraphQL", "CI/CD", "Linux", "Windows", "MacOS",
            "HTML", "CSS", "SASS", "LESS", "Redux", "RxJS", "Spring Boot",
            ".NET", "ASP.NET", "PHP", "Laravel", "Ruby", "Ruby on Rails"
        ]
        
        # Add custom skills if provided
        if custom_skills:
            custom_skills_list = [skill.strip() for skill in custom_skills.split(',') if skill.strip()]
            all_skills = common_skills + custom_skills_list
        else:
            all_skills = common_skills
        
        # Create text processor and extract skills
        text_processor = TextProcessor(skill_keywords=all_skills)
        found_skills = text_processor.extract_skills(job_description)
        
        # Classify skills by category (simplified example)
        categorized_skills = {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "cloud": [],
            "tools": [],
            "methodologies": [],
            "other": []
        }
        
        # Simple categorization logic - would be much more sophisticated in production
        for skill in found_skills:
            if skill in ["Python", "Java", "JavaScript", "C++", "C#", "TypeScript", "Ruby", "PHP"]:
                categorized_skills["programming_languages"].append(skill)
            elif skill in ["React", "Angular", "Vue.js", "Django", "Flask", "Spring Boot", ".NET", "Laravel"]:
                categorized_skills["frameworks"].append(skill)
            elif skill in ["SQL", "PostgreSQL", "MySQL", "MongoDB"]:
                categorized_skills["databases"].append(skill)
            elif skill in ["AWS", "Azure", "Google Cloud"]:
                categorized_skills["cloud"].append(skill)
            elif skill in ["Docker", "Kubernetes", "Git", "Jenkins"]:
                categorized_skills["tools"].append(skill)
            elif skill in ["Agile", "Scrum", "DevOps", "CI/CD"]:
                categorized_skills["methodologies"].append(skill)
            else:
                categorized_skills["other"].append(skill)
        
        return {
            "extracted_skills": found_skills,
            "categorized_skills": categorized_skills,
            "total_skills_found": len(found_skills)
        }
    
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting skills: {str(e)}"
        )

@router.get("/statistics/", response_model=Dict[str, Any])
async def get_system_statistics():
    """Get statistics about the CV matching system."""
    try:
        # Count stored jobs
        job_count = len(list(storage.jobs_dir.glob("*.txt")))
        
        # Count stored CVs
        cv_count = len(list(storage.cvs_dir.glob("*.*")))
        
        # Count results
        result_count = len(list(storage.results_dir.glob("*.json")))
        
        # In a real system, we'd get more sophisticated metrics from a database
        
        return {
            "jobs_stored": job_count,
            "cvs_processed": cv_count,
            "matching_results": result_count,
            "system_status": "healthy",
            "last_updated": datetime.now()
        }
    
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving system statistics: {str(e)}"
        )

# Enhanced application configuration and setup
def configure_cv_matcher():
    """Configure and initialize the CV matching system."""
    try:
        # Ensure required directories exist
        storage.base_dir.mkdir(exist_ok=True, parents=True)
        storage.jobs_dir.mkdir(exist_ok=True, parents=True)
        storage.cvs_dir.mkdir(exist_ok=True, parents=True)
        storage.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize NLP components if not already done
        if not nlp:
            logger.warning("spaCy model not available. Some advanced NLP features will be limited.")
        
        # Load custom skill dictionaries if available
        skills_path = Path("./data/skills_dictionary.json")
        if skills_path.exists():
            with open(skills_path, 'r', encoding='utf-8') as f:
                # This would load a comprehensive skills dictionary
                pass
        
        logger.info("CV Matcher system initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error configuring CV Matcher: {e}")
        return False
