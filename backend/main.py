"""
FastAPI Application Entry Point.

Main application file that:
- Configures FastAPI app
- Registers routes
- Sets up middleware
- Configures CORS and error handlers
"""
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env file from current directory

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

from schemas.resume import ResumeUploadResponse
from schemas.job import JobSearchRequest, JobSearchResponse
from schemas.match import MatchResponse
from services.resume_parser import get_resume_parser
from services.job_service import get_job_service
from services.matcher import get_job_matcher
from config import settings


# Create FastAPI app
app = FastAPI(
    title="Resume-to-Jobs Matching Platform",
    description="Upload your resume and find matching jobs automatically using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Root route
@app.get("/")
async def root():
    """Redirect to documentation."""
    return RedirectResponse(url="/docs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class MatchRequest(BaseModel):
    """Request for matching a resume to jobs."""
    resume_id: str
    role: Optional[str] = Query(None, description="Optional role filter")
    location: Optional[str] = Query(None, description="Optional location filter")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "resume-matcher",
        "version": "1.0.0"
    }


# Upload and parse resume endpoint
@app.post("/upload-resume", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file (PDF or DOCX)")
):
    """
    Upload and parse a resume file.

    Accepts PDF or DOCX files and extracts structured data using LLM.

    Returns:
    - resume_id: Unique identifier for the parsed resume
    - extracted_data: Structured resume information
    """
    # Validate file type
    content_type = file.content_type
    file_extension = file.filename.split(".")[-1].lower()

    valid_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    valid_extensions = ["pdf", "docx"]

    if content_type not in valid_types and file_extension not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Supported formats: PDF, DOCX"
        )

    # Read file content
    file_content = await file.read()

    if len(file_content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )

    # Parse resume
    parser = get_resume_parser()

    try:
        result = await parser.parse_file(
            file_path=file.filename,
            file_content=file_content,
            file_type=file_extension
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")


# Parse resume from text endpoint
@app.post("/parse-text")
async def parse_resume_text(
    text: str,
    resume_id: Optional[str] = Query(None, description="Optional existing resume ID")
):
    """
    Parse a resume from raw text.

    Useful for copy-pasting resume content or importing from other sources.

    Returns:
    - resume_id: Unique identifier
    - extracted_data: Structured resume information
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty")

    if len(text) > 50000:
        raise HTTPException(status_code=400, detail="Text exceeds 50000 character limit")

    parser = get_resume_parser()

    try:
        result = await parser.parse_text(text, resume_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse resume: {str(e)}")


# Get jobs endpoint
@app.get("/jobs", response_model=JobSearchResponse)
async def get_jobs(
    role: str = Query(..., description="Job role to search for"),
    location: Optional[str] = Query(None, description="Optional location filter"),
    skills: Optional[str] = Query(None, description="Comma-separated list of skills"),
    max_results: int = Query(20, ge=1, le=100, description="Maximum number of results")
):
    """
    Search for job listings.

    Searches across multiple job sources (Indeed, RemoteOK, LinkedIn).

    Query Parameters:
    - role: Job title/role (required)
    - location: Optional location filter
    - skills: Optional comma-separated skills
    - max_results: Number of results (default: 20)

    Returns:
    - total_jobs: Total number of matching jobs
    - jobs: List of job listings
    - query: The search query parameters
    """
    # Parse skills from comma-separated string
    skill_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else []

    query = JobSearchRequest(
        role=role,
        location=location,
        skills=skill_list,
        max_results=max_results
    )

    service = get_job_service()

    try:
        result = await service.search_jobs(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search jobs: {str(e)}")


# Match resume to jobs endpoint
@app.post("/match-jobs")
async def match_jobs(
    match_request: MatchRequest
):
    """
    Match a parsed resume to job listings.

    Uses a two-stage approach:
    1. Heuristic filtering (skills overlap)
    2. LLM-based scoring (detailed analysis)

    Returns:
    - total_matches: Number of matching jobs
    - matched_jobs: Ranked list of jobs with match scores
    - processing_time_ms: Time taken to process
    """
    # Get parsed resume from cache
    parser = get_resume_parser()
    resume_data = await parser.get_parsed_resume(match_request.resume_id)

    if not resume_data:
        raise HTTPException(
            status_code=404,
            detail="Resume not found. Please upload/parse the resume first."
        )

    # Search for jobs
    query = JobSearchRequest(
        role=match_request.role or resume_data.preferred_roles[0],
        location=match_request.location,
        skills=resume_data.skills,
        max_results=50  # Get more jobs to match
    )

    job_service = get_job_service()
    jobs = await job_service.search_jobs(query)

    if not jobs.jobs:
        return MatchResponse(
            total_matches=0,
            matched_jobs=[],
            processing_time_ms=0
        )

    # Match resume to jobs
    matcher = get_job_matcher()
    result = await matcher.match_resume_to_jobs_full(resume_data, jobs.jobs)

    return result


# Apply redirect endpoint
@app.get("/apply")
async def apply_to_job(url: str = Query(..., description="Apply URL to redirect to")):
    """
    Redirect to job application URL.

    Validates and redirects the user to the job's apply page.

    Query Parameters:
    - url: The job's apply URL
    """
    if not url:
        raise HTTPException(status_code=400, detail="Apply URL is required")

    # Basic URL validation
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL format")

    # Redirect to the apply URL
    return RedirectResponse(url=url)


# Get resume endpoint
@app.get("/resume/{resume_id}")
async def get_resume(resume_id: str):
    """
    Get a previously parsed resume by ID.

    Query Parameters:
    - resume_id: The ID of the parsed resume

    Returns:
    - resume_data: The structured resume information
    """
    parser = get_resume_parser()
    resume_data = await parser.get_parsed_resume(resume_id)

    if not resume_data:
        raise HTTPException(status_code=404, detail="Resume not found")

    return {"resume_id": resume_id, "data": resume_data.model_dump()}


# Stats endpoint
@app.get("/stats")
async def get_stats():
    """
    Get system statistics.

    Returns:
    - Cached items count
    - Enabled scrapers
    """
    from services.cache import get_cache
    cache = get_cache()

    return {
        "enabled_scrapers": settings.ENABLED_SOURCES,
        "job_age_threshold_days": settings.JOB_AGE_DAYS_THRESHOLD,
        "max_jobs_per_source": settings.MAX_JOBS_PER_SOURCE,
        "max_jobs_for_llm_scoring": settings.MAX_JOBS_FOR_LLM_SCORING,
    }


# Error handlers
@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
