"""
Video Generation MCP Server
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import mcp.types as types
import asyncio
import logging
import os
import uuid
import time
from typing import Dict, Optional, Literal, Any
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from enum import Enum

# Import our video generation functions
from script_processing.generate_assets import generate_assets, generate_audio_and_caption, export_mp3
from script_processing.generate_paper import process_article_firecrawl
from script_processing.generate_script import process_script
from script_processing.type import Text
from video_processing.generate_video import process_video

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Job tracking system
class JobStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class JobInfo:
    job_id: str
    status: JobStatus
    job_type: str
    progress: float = 0.0
    start_time: float = 0.0
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "job_type": self.job_type,
            "progress": self.progress,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "result": self.result,
            "error_message": self.error_message,
            "duration": (self.end_time - self.start_time) if self.end_time else None
        }

# Global job storage
jobs: Dict[str, JobInfo] = {}

def create_job(job_type: str) -> str:
    """Create a new job and return its ID"""
    job_id = str(uuid.uuid4())
    job = JobInfo(
        job_id=job_id,
        status=JobStatus.PENDING,
        job_type=job_type,
        start_time=time.time()
    )
    jobs[job_id] = job
    logger.info(f"Created job {job_id} of type {job_type}")
    return job_id

def update_job_status(job_id: str, status: JobStatus, progress: float = None, result: Dict[str, Any] = None, error_message: str = None):
    """Update job status and related fields"""
    if job_id not in jobs:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return
    
    job = jobs[job_id]
    job.status = status
    
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result = result
    if error_message is not None:
        job.error_message = error_message
    if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        job.end_time = time.time()
    
    logger.info(f"Updated job {job_id}: status={status.value}, progress={job.progress}")

def get_job(job_id: str) -> Optional[JobInfo]:
    """Get job information by ID"""
    return jobs.get(job_id)

# Async wrapper functions for background processing
async def _async_generate_assets(job_id: str, script: str, method: str, output_dir: str):
    """Background task for asset generation"""
    try:
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=0.0)
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up file paths
        audio_file = str(output_path / "audio.wav")
        subtitles_file = str(output_path / "subtitles.srt")
        rich_content_file = str(output_path / "rich.json")
        
        logger.info(f"Job {job_id}: Generating assets with {method} method...")
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=25.0)
        
        # Generate assets (this is the time-consuming part)
        duration = await asyncio.get_event_loop().run_in_executor(
            None, 
            generate_assets,
            script, method, audio_file, subtitles_file, rich_content_file
        )
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=100.0)
        
        result = {
            "status": "success",
            "audio_file": audio_file,
            "subtitles_file": subtitles_file,
            "rich_content_file": rich_content_file,
            "duration": str(duration)
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, result=result)
        logger.info(f"Job {job_id}: Successfully generated assets. Duration: {duration}s")
        
    except Exception as e:
        error_message = f"Error generating assets: {str(e)}"
        logger.error(f"Job {job_id}: {error_message}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_message)

async def _async_generate_video(job_id: str, input_dir: str, output_video: str, concurrency: int):
    """Background task for video generation"""
    try:
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=0.0)
        
        input_path = Path(input_dir)
        output_path = Path(output_video)
        
        # Validate input directory and required files
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        required_files = ["audio.wav", "subtitles.srt", "rich.json"]
        missing_files = []
        
        for file in required_files:
            if not (input_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise ValueError(f"Missing required files in {input_dir}: {', '.join(missing_files)}")
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=20.0)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Job {job_id}: Generating video from {input_dir} to {output_video}")
        
        # Generate video (this is the time-consuming part)
        final_output = await asyncio.get_event_loop().run_in_executor(
            None,
            process_video,
            input_path, output_path, concurrency
        )
        
        result = {
            "status": "success",
            "output_video": str(final_output)
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, progress=100.0, result=result)
        logger.info(f"Job {job_id}: Successfully generated video: {output_video}")
        
    except Exception as e:
        error_message = f"Error generating video: {str(e)}"
        logger.error(f"Job {job_id}: {error_message}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_message)

async def _async_generate_complete_video_from_url(job_id: str, content_url: str, method: str, output_dir: str, output_video: str, concurrency: int):
    """Background task for complete video generation pipeline from URL"""
    try:
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=0.0)
        
        logger.info(f"Job {job_id}: Processing content from URL: {content_url}")
        
        # Step 1: Process article from URL (0-20% progress)
        contain_markdown = await asyncio.get_event_loop().run_in_executor(
            None, 
            process_article_firecrawl, 
            content_url
        )
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=10.0)
        
        # Step 2: Generate script from markdown (10-20% progress)
        script = await asyncio.get_event_loop().run_in_executor(
            None,
            process_script,
            "openrouter",  # method
            contain_markdown,  # paper_markdown
            False,  # from_pdf
            "video"  # mode
        )
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=20.0)
        
        logger.info(f"Job {job_id}: Starting complete video generation pipeline...")
        
        # Step 3: Generate assets (20-60% progress)
        assets_job_id = create_job("asset_generation_sub")
        await _async_generate_assets(assets_job_id, script, method, output_dir)
        
        assets_job = get_job(assets_job_id)
        if assets_job.status != JobStatus.COMPLETED:
            raise Exception(f"Asset generation failed: {assets_job.error_message}")
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=60.0)
        
        # Step 4: Generate video from assets (60-100% progress)
        video_job_id = create_job("video_generation_sub")
        await _async_generate_video(video_job_id, output_dir, output_video, concurrency)
        
        video_job = get_job(video_job_id)
        if video_job.status != JobStatus.COMPLETED:
            raise Exception(f"Video generation failed: {video_job.error_message}")
        
        # Combine results
        result = {
            "status": "success",
            "output_video": video_job.result["output_video"],
            "duration": assets_job.result["duration"],
            "script": script,
            "source_url": content_url,
            "assets": {
                "audio_file": assets_job.result["audio_file"],
                "subtitles_file": assets_job.result["subtitles_file"],
                "rich_content_file": assets_job.result["rich_content_file"]
            }
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, progress=100.0, result=result)
        logger.info(f"Job {job_id}: Complete video generation pipeline from URL finished successfully")
        
    except Exception as e:
        error_message = f"Error in complete video generation from URL: {str(e)}"
        logger.error(f"Job {job_id}: {error_message}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_message)

async def _async_generate_complete_video(job_id: str, script: str, method: str, output_dir: str, output_video: str, concurrency: int):
    """Background task for complete video generation pipeline"""
    try:
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=0.0)
        
        logger.info(f"Job {job_id}: Starting complete video generation pipeline...")
        
        # Step 1: Generate assets (0-50% progress)
        assets_job_id = create_job("asset_generation_sub")
        await _async_generate_assets(assets_job_id, script, method, output_dir)
        
        assets_job = get_job(assets_job_id)
        if assets_job.status != JobStatus.COMPLETED:
            raise Exception(f"Asset generation failed: {assets_job.error_message}")
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=50.0)
        
        # Step 2: Generate video from assets (50-100% progress)
        video_job_id = create_job("video_generation_sub")
        await _async_generate_video(video_job_id, output_dir, output_video, concurrency)
        
        video_job = get_job(video_job_id)
        if video_job.status != JobStatus.COMPLETED:
            raise Exception(f"Video generation failed: {video_job.error_message}")
        
        # Combine results
        result = {
            "status": "success",
            "output_video": video_job.result["output_video"],
            "duration": assets_job.result["duration"],
            "assets": {
                "audio_file": assets_job.result["audio_file"],
                "subtitles_file": assets_job.result["subtitles_file"],
                "rich_content_file": assets_job.result["rich_content_file"]
            }
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, progress=100.0, result=result)
        logger.info(f"Job {job_id}: Complete video generation pipeline finished successfully")
        
    except Exception as e:
        error_message = f"Error in complete video generation: {str(e)}"
        logger.error(f"Job {job_id}: {error_message}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_message)

async def _async_generate_complete_podcast(job_id: str, content_url: str, output_file: str, voice: str = "female"):
    """Background task for complete podcast generation from URL"""
    try:
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=0.0)
        
        logger.info(f"Job {job_id}: Processing content from URL: {content_url}")
        
        # Step 1: Process article from URL (0-25% progress)
        contain_markdown = await asyncio.get_event_loop().run_in_executor(
            None, 
            process_article_firecrawl, 
            content_url
        )
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=25.0)
        
        # Step 2: Generate script from markdown (25-50% progress)
        script = await asyncio.get_event_loop().run_in_executor(
            None,
            process_script,
            "openrouter",  # method
            contain_markdown,  # paper_markdown
            "podcast"  # mode
        )
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=50.0)
        
        # Step 3: Generate podcast from script (50-100% progress)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Job {job_id}: Generating podcast with {voice} voice...")
        
        # Generate audio and caption from script (this also parses the script internally)
        text_content = await asyncio.get_event_loop().run_in_executor(
            None,
            generate_audio_and_caption,
            script  # Pass the script string directly
        )
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=90.0)
        
        # Filter only Text objects for audio export
        text_only_content = [item for item in text_content if isinstance(item, Text)]
        
        # Export to final MP3 file
        await asyncio.get_event_loop().run_in_executor(
            None,
            export_mp3,
            text_only_content,
            str(output_file)
        )
        
        update_job_status(job_id, JobStatus.IN_PROGRESS, progress=100.0)
        
        result = {
            "status": "success",
            "podcast_file": str(output_file),
            "voice": voice,
            "script": script,
            "source_url": content_url
        }
        
        update_job_status(job_id, JobStatus.COMPLETED, result=result)
        logger.info(f"Job {job_id}: Successfully generated complete podcast from URL")
        
    except Exception as e:
        error_message = f"Error generating complete podcast: {str(e)}"
        logger.error(f"Job {job_id}: {error_message}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_message)

mcp = FastMCP("Video Generator", port=3000, stateless_http=True, debug=True)

@mcp.tool(
    title="Check Job Status",
    description="Check if a video generation job is finished and get its status, progress, and results. Use this to bypass client-side timeouts by polling job completion."
)
async def check_job_status(
    job_id: str = Field(description="The job ID returned from a video generation function")
) -> Dict[str, Any]:
    """
    Check the status of a video generation job.
    
    Input format:
    {
        "job_id": "uuid-string-here"
    }
    
    Output format:
    {
        "job_id": "uuid-string-here",
        "status": "pending|in_progress|completed|failed",
        "job_type": "asset_generation|video_generation|complete_video",
        "progress": 75.0,
        "start_time": 1234567890.0,
        "end_time": 1234567895.0,
        "duration": 5.0,
        "result": {...},
        "error_message": "error details if failed"
    }
    """
    job = get_job(job_id)
    
    if not job:
        return {
            "status": "error",
            "message": f"Job {job_id} not found"
        }
    
    return job.to_dict()

@mcp.tool(
    title="List All Jobs", 
    description="List all video generation jobs and their current status. Useful for monitoring and debugging."
)
async def list_jobs(
    status_filter: Optional[str] = Field(default=None, description="Filter jobs by status: pending, in_progress, completed, failed")
) -> Dict[str, Any]:
    """
    List all jobs with optional status filtering.
    
    Input format:
    {
        "status_filter": "completed"  // optional
    }
    
    Output format:
    {
        "jobs": [
            {
                "job_id": "uuid1",
                "status": "completed",
                "job_type": "complete_video",
                "progress": 100.0,
                ...
            },
            ...
        ],
        "total_count": 5,
        "filtered_count": 2
    }
    """
    all_jobs = list(jobs.values())
    
    if status_filter:
        try:
            filter_status = JobStatus(status_filter.lower())
            filtered_jobs = [job for job in all_jobs if job.status == filter_status]
        except ValueError:
            return {
                "status": "error",
                "message": f"Invalid status filter '{status_filter}'. Valid options: pending, in_progress, completed, failed"
            }
    else:
        filtered_jobs = all_jobs
    
    return {
        "jobs": [job.to_dict() for job in filtered_jobs],
        "total_count": len(all_jobs),
        "filtered_count": len(filtered_jobs)
    }

@mcp.tool(
    title="Generate Video Assets",
    description="Generate audio, subtitles, and rich content from a script text. Returns a job ID for tracking progress. Use check_job_status to monitor completion."
)
async def generate_video_assets(
    script: str = Field(description="The video script text containing \\Headline:, \\Text:, \\Figure: markers"),
    method: Optional[str] = Field(default="kokoro", description="Audio generation method: 'kokoro' (default), 'elevenlabs', or 'lmnt'"),
    output_dir: Optional[str] = Field(default="public", description="Output directory for generated assets"),
    async_mode: Optional[bool] = Field(default=True, description="If True, returns job ID immediately. If False, waits for completion (may timeout)")
) -> Dict[str, str]:
    """
    Generate video assets from script text.
    
    Input format (minimal):
    {
        "script": "\\Headline: Title\\nText: Content..."
    }
    
    Input format (full):
    {
        "script": "\\Headline: Title\\nText: Content...",
        "method": "kokoro", 
        "output_dir": "public",
        "async_mode": true
    }
    
    Output format (async_mode=true):
    {
        "status": "started",
        "job_id": "uuid-string-here",
        "message": "Asset generation started. Use check_job_status to monitor progress."
    }
    
    Output format (async_mode=false):
    {
        "status": "success",
        "audio_file": "public/audio.wav",
        "subtitles_file": "public/subtitles.srt", 
        "rich_content_file": "public/rich.json",
        "duration": "12.5"
    }
    """
    try:
        # Validate method
        valid_methods = ["elevenlabs", "lmnt", "kokoro"]
        if method not in valid_methods:
            return {
                "status": "error",
                "message": f"Invalid method '{method}'. Available methods: {', '.join(valid_methods)}"
            }
        
        if async_mode:
            # Create job and start background task
            job_id = create_job("asset_generation")
            asyncio.create_task(_async_generate_assets(job_id, script, method, output_dir))
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Asset generation started. Use check_job_status to monitor progress."
            }
        else:
            # Synchronous mode (original behavior)
            # Ensure output directory exists
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Set up file paths
            audio_file = str(output_path / "audio.wav")
            subtitles_file = str(output_path / "subtitles.srt")
            rich_content_file = str(output_path / "rich.json")
            
            logger.info(f"Generating assets with {method} method...")
            
            # Generate assets
            duration = generate_assets(
                script=script,
                method=method,
                mp3_output=audio_file,
                srt_output=subtitles_file,
                rich_output=rich_content_file
            )
            
            logger.info(f"Successfully generated assets. Duration: {duration}s")
            
            return {
                "status": "success",
                "audio_file": audio_file,
                "subtitles_file": subtitles_file,
                "rich_content_file": rich_content_file,
                "duration": str(duration)
            }
        
    except Exception as e:
        error_message = f"Error generating assets: {str(e)}"
        logger.error(error_message)
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.tool(
    title="Generate Video",
    description="Generate a video from pre-existing assets (audio.wav, subtitles.srt, rich.json). Returns a job ID for tracking progress. Use check_job_status to monitor completion."
)
async def generate_video_from_assets(
    input_dir: str = Field(description="Directory containing audio.wav, subtitles.srt, and rich.json files"),
    output_video: Optional[str] = Field(default="public/output.mp4", description="Path for the output video file"),
    concurrency: Optional[int] = Field(default=6, description="Rendering concurrency level (1-10)"),
    async_mode: Optional[bool] = Field(default=True, description="If True, returns job ID immediately. If False, waits for completion (may timeout)")
) -> Dict[str, str]:
    """
    Generate video from existing assets.
    
    Input format (minimal):
    {
        "input_dir": "public"
    }
    
    Input format (full):
    {
        "input_dir": "public",
        "output_video": "public/output.mp4",
        "concurrency": 6,
        "async_mode": true
    }
    
    Output format (async_mode=true):
    {
        "status": "started",
        "job_id": "uuid-string-here",
        "message": "Video generation started. Use check_job_status to monitor progress."
    }
    
    Output format (async_mode=false):
    {
        "status": "success",
        "output_video": "public/output.mp4"
    }
    """
    try:
        # Validate concurrency
        if not 1 <= concurrency <= 10:
            concurrency = 6
            logger.warning("Concurrency out of range (1-10), using default value of 6")
        
        if async_mode:
            # Create job and start background task
            job_id = create_job("video_generation")
            asyncio.create_task(_async_generate_video(job_id, input_dir, output_video, concurrency))
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Video generation started. Use check_job_status to monitor progress."
            }
        else:
            # Synchronous mode (original behavior)
            input_path = Path(input_dir)
            output_path = Path(output_video)
            
            # Validate input directory and required files
            if not input_path.exists() or not input_path.is_dir():
                return {
                    "status": "error",
                    "message": f"Input directory {input_dir} does not exist"
                }
            
            required_files = ["audio.wav", "subtitles.srt", "rich.json"]
            missing_files = []
            
            for file in required_files:
                if not (input_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                return {
                    "status": "error",
                    "message": f"Missing required files in {input_dir}: {', '.join(missing_files)}"
                }
            
            logger.info(f"Generating video from {input_dir} to {output_video}")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate video
            process_video(
                input=input_path,
                output=output_path,
                concurrency=concurrency
            )
            
            logger.info(f"Successfully generated video: {output_video}")
            
            return {
                "status": "success",
                "output_video": str(output_path)
            }
        
    except Exception as e:
        error_message = f"Error generating video: {str(e)}"
        logger.error(error_message)
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool(
    title="Generate Podcast",
    description="Generate a podcast (audio file) from a URL. Returns a job ID for tracking progress. Use check_job_status to monitor completion."
)
async def generate_podcast_from_url(
    content_url: str = Field(description="URL of the content to generate a podcast from"),
    output_file: Optional[str] = Field(default="public/podcast.wav", description="Path for the output podcast audio file"),
    voice: Optional[str] = Field(default="female", description="Voice for the podcast: 'female' or 'male'"),
    async_mode: Optional[bool] = Field(default=True, description="If True, returns job ID immediately. If False, waits for completion (may timeout)")
) -> Dict[str, str]:
    """
    Generate podcast audio from content URL.
    
    Input format (minimal):
    {
        "content_url": "https://example.com/article"
    }
    
    Input format (full):
    {
        "content_url": "https://example.com/article",
        "output_file": "public/my_podcast.wav",
        "voice": "female",
        "async_mode": true
    }
    
    Output format (async_mode=true):
    {
        "status": "started",
        "job_id": "uuid-string-here",
        "message": "Podcast generation started. Use check_job_status to monitor progress."
    }
    
    Output format (async_mode=false):
    {
        "status": "success",
        "podcast_file": "public/podcast.wav",
        "voice": "female",
        "script": "Generated script...",
        "source_url": "https://example.com/article"
    }
    """
    try:
        if not content_url:
            return {
                "status": "error",
                "message": "Content URL is required"
            }
        
        # Validate voice
        valid_voices = ["female", "male"]
        if voice not in valid_voices:
            return {
                "status": "error",
                "message": f"Invalid voice '{voice}'. Available voices: {', '.join(valid_voices)}"
            }
        
        if async_mode:
            # Create job and start background task
            job_id = create_job("complete_podcast")
            asyncio.create_task(_async_generate_complete_podcast(job_id, content_url, output_file, voice))
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Podcast generation started. Use check_job_status to monitor progress."
            }
        else:
            # Synchronous mode 
            logger.info(f"Starting complete podcast generation from URL: {content_url}")
            
            # Step 1: Process article from URL
            contain_markdown = process_article_firecrawl(content_url)
            
            # Step 2: Generate script from markdown
            script = process_script("openrouter", contain_markdown, "podcast")
            
            # Step 3: Generate podcast from script
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Generating podcast with {voice} voice...")
            
            # Generate audio and caption from script (this also parses the script internally)
            text_content = generate_audio_and_caption(script)
            
            # Filter only Text objects for audio export
            text_only_content = [item for item in text_content if isinstance(item, Text)]
            
            # Export to final MP3 file
            export_mp3(text_only_content, str(output_file))
            
            logger.info(f"Successfully generated podcast: {output_file}")
            
            return {
                "status": "success",
                "podcast_file": str(output_file),
                "voice": voice,
                "script": script,
                "source_url": content_url
            }
        
    except Exception as e:
        error_message = f"Error generating podcast: {str(e)}"
        logger.error(error_message)
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool(
    title="Generate Complete Video",
    description="Complete video generation pipeline: generate assets from script and then create video. Returns a job ID for tracking progress. Use check_job_status to monitor completion."
)
async def generate_complete_video(
    containt_url: Optional[str] = Field(description="URL of the content to generate a video from"),
    method: Optional[str] = Field(default="kokoro", description="Audio generation method: 'kokoro' (default), 'elevenlabs', or 'lmnt'"),
    output_dir: Optional[str] = Field(default="public", description="Output directory for generated files"),
    output_video: Optional[str] = Field(default="public/output.mp4", description="Path for the output video file"),
    concurrency: Optional[int] = Field(default=6, description="Video rendering concurrency level (1-10)"),
    async_mode: Optional[bool] = Field(default=True, description="If True, returns job ID immediately. If False, waits for completion (may timeout)")
) -> Dict[str, str]:
    """
    Complete video generation from script to final video.
    
    Input format (minimal):
    {
        "script": "\\Headline: Title\\nText: Content..."
    }
    
    Input format (full):
    {
        "script": "\\Headline: Title\\nText: Content...",
        "method": "kokoro",
        "output_dir": "public",
        "output_video": "public/output.mp4",
        "concurrency": 6,
        "async_mode": true
    }
    
    Output format (async_mode=true):
    {
        "status": "started",
        "job_id": "uuid-string-here",
        "message": "Complete video generation started. Use check_job_status to monitor progress."
    }
    
    Output format (async_mode=false):
    {
        "status": "success",
        "output_video": "public/output.mp4",
        "duration": "12.5",
        "assets": {
            "audio_file": "public/audio.wav",
            "subtitles_file": "public/subtitles.srt",
            "rich_content_file": "public/rich.json"
        }
    }
    """
    try:
        if not containt_url:
            return {
                "status": "error",
                "message": "Containt URL is required"
            }
        
        # Validate method
        valid_methods = ["elevenlabs", "lmnt", "kokoro"]
        if method not in valid_methods:
            return {
                "status": "error",
                "message": f"Invalid method '{method}'. Available methods: {', '.join(valid_methods)}"
            }
        
        if async_mode:
            # Create job and start background task immediately
            job_id = create_job("complete_video")
            asyncio.create_task(_async_generate_complete_video_from_url(job_id, containt_url, method, output_dir, output_video, concurrency))
            
            return {
                "status": "started",
                "job_id": job_id,
                "message": "Complete video generation started. Use check_job_status to monitor progress."
            }
        else:
            # Synchronous mode - process URL and generate script first
            contain_markdown = await asyncio.get_event_loop().run_in_executor(
                None,
                process_article_firecrawl,
                containt_url
            )
            script = await asyncio.get_event_loop().run_in_executor(
                None,
                process_script,
                "openrouter",  # method
                contain_markdown,  # paper_markdown
                False,  # from_pdf
                "video"  # mode
            )
            
            # Synchronous mode (original behavior) - use legacy sync pipeline
            logger.info("Starting complete video generation pipeline...")
            
            # Step 1: Generate assets
            assets_result = await generate_video_assets(script, method, output_dir, async_mode=False)
            
            if assets_result["status"] != "success":
                return assets_result
            
            # Step 2: Generate video from assets
            video_result = await generate_video_from_assets(output_dir, output_video, concurrency, async_mode=False)
            
            if video_result["status"] != "success":
                return video_result
            
            logger.info("Complete video generation pipeline finished successfully")
            
            return {
                "status": "success",
                "output_video": video_result["output_video"],
                "duration": assets_result["duration"],
                "assets": {
                    "audio_file": assets_result["audio_file"],
                    "subtitles_file": assets_result["subtitles_file"],
                    "rich_content_file": assets_result["rich_content_file"]
                }
            }
        
    except Exception as e:
        error_message = f"Error in complete video generation: {str(e)}"
        logger.error(error_message)
        return {
            "status": "error",
            "message": str(e)
        }

@mcp.resource(
    uri="video://info/methods",
    description="Information about available audio generation methods and async job tracking",
    name="Audio Generation Methods & Job Tracking",
)
def get_methods_info() -> str:
    """Get information about all available audio generation methods and job tracking"""
    return """
# Available Audio Generation Methods

## Kokoro (Default)
- **Method ID**: `kokoro`
- **Best for**: Fast, high-quality speech synthesis
- **Strengths**: Fast generation, good quality, offline processing
- **Requirements**: None (bundled)

## ElevenLabs
- **Method ID**: `elevenlabs` 
- **Best for**: Premium, natural-sounding voices
- **Strengths**: Very natural speech, multiple voice options
- **Requirements**: ELEVENLABS_API_KEY environment variable

## LMNT
- **Method ID**: `lmnt`
- **Best for**: Professional speech synthesis
- **Strengths**: High quality, reliable
- **Requirements**: LMNT_API_KEY environment variable

# Async Job Tracking System

To bypass client-side timeouts, all video generation functions now support asynchronous execution:

## How It Works
1. **Start Job**: Call any video generation function with `async_mode: true` (default)
2. **Get Job ID**: Receive a unique job ID immediately
3. **Check Progress**: Use `check_job_status` with the job ID to monitor progress
4. **Get Results**: When status is "completed", retrieve the final results

## Job Statuses
- **pending**: Job created but not started
- **in_progress**: Job is currently running (shows progress percentage)
- **completed**: Job finished successfully (results available)
- **failed**: Job encountered an error (error message available)

## Usage Pattern

```javascript
// 1. Start video generation
const startResult = await generate_complete_video({
    script: "\\Headline: My Video\\nText: Content here...",
    async_mode: true
});

// 2. Get job ID
const jobId = startResult.job_id;

// 3. Poll for completion
let jobStatus;
do {
    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds
    jobStatus = await check_job_status({ job_id: jobId });
} while (jobStatus.status === "pending" || jobStatus.status === "in_progress");

// 4. Get results
if (jobStatus.status === "completed") {
    console.log("Video ready:", jobStatus.result.output_video);
} else {
    console.error("Job failed:", jobStatus.error_message);
}
```

## Script Format

Your script should use these markers:
- `\\Headline: Your title here` - Creates a headline section
- `\\Text: Your content here` - Creates a text section  
- `\\Figure: https://example.com/image.png` - Adds an image/figure

## Example Script

```
\\Headline: Welcome to My Video
\\Text: This is the introduction to our topic.
\\Figure: https://example.com/intro-image.png
\\Text: Let's dive deeper into the subject matter.
\\Headline: Key Points
\\Text: Here are the main points we'll cover today.
```

## Usage Examples

### Async Mode (Recommended - Prevents Timeouts)
Use `async_mode: true` and poll with `check_job_status`:

- `generate_video_assets` - Create audio, subtitles, and rich content
- `generate_video_from_assets` - Create video from existing assets  
- `generate_complete_video` - End-to-end pipeline from script to video

### Sync Mode (Legacy - May Timeout)
Use `async_mode: false` for immediate results (not recommended for long-running tasks).

### Job Management
- `check_job_status` - Check if job is finished and get results
- `list_jobs` - List all jobs with optional status filtering
"""

@mcp.resource(
    uri="video://info/requirements",
    description="System requirements, file formats, and async job system",
    name="Requirements & Job System",
)
def get_requirements_info() -> str:
    """Get information about system requirements and async job system"""
    return """
# System Requirements

## Required Files for Video Generation
- `audio.wav` - Audio track
- `subtitles.srt` - Subtitle file in SRT format
- `rich.json` - Rich content metadata (figures, headlines, etc.)

## Dependencies
- Node.js and pnpm (for Remotion video rendering)
- Python 3.12+ 
- FFmpeg (for video processing)

## Output Formats
- Video: MP4 (1920x1080, 30fps)
- Audio: WAV
- Subtitles: SRT format

## Environment Variables (Optional)
- `ELEVENLABS_API_KEY` - For ElevenLabs TTS
- `LMNT_API_KEY` - For LMNT TTS

## Performance Notes
- Concurrency levels 1-10 (default: 6)
- Higher concurrency = faster rendering but more CPU usage
- Kokoro method is fastest for audio generation

# Async Job System (Timeout Prevention)

## Problem Solved
Client-side timeouts are bypassed by running video generation as background jobs.

## Job Lifecycle
1. **Job Creation**: Each video generation request creates a unique job
2. **Background Processing**: Long-running tasks execute without blocking
3. **Progress Tracking**: Monitor progress percentage and status
4. **Result Retrieval**: Get final results when job completes

## Job Storage
- Jobs stored in-memory during server runtime
- Job data includes: ID, status, progress, start/end times, results, errors
- Jobs persist until server restart

## API Functions
- **check_job_status(job_id)**: Check if job is finished and get results
- **list_jobs(status_filter?)**: List all jobs with optional filtering

## Best Practices
1. Always use `async_mode: true` (default) for long-running operations
2. Poll job status every 5-10 seconds to avoid overwhelming the server
3. Handle all job statuses: pending, in_progress, completed, failed
4. Check for errors in failed jobs via `error_message` field

## Timeout Prevention Strategy
- Start job → Get job ID → Poll status → Retrieve results
- No client-side blocking on long video generation processes
- Suitable for integration with web applications and chatbots
"""

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
