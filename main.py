"""
Video Generation MCP Server
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
import mcp.types as types
import asyncio
import logging
import os
from typing import Dict, Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

# Import our video generation functions
from script_processing.generate_assets import generate_assets
from video_processing.generate_video import process_video

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Video Generator", port=3000, stateless_http=True, debug=True)

@mcp.tool(
    title="Generate Video Assets",
    description="Generate audio, subtitles, and rich content from a script text. This creates the necessary assets (audio.wav, subtitles.srt, rich.json) for video generation."
)
async def generate_video_assets(
    script: str = Field(description="The video script text containing \\Headline:, \\Text:, \\Figure: markers"),
    method: Optional[str] = Field(default="kokoro", description="Audio generation method: 'kokoro' (default), 'elevenlabs', or 'lmnt'"),
    output_dir: Optional[str] = Field(default="public", description="Output directory for generated assets")
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
        "output_dir": "public"
    }
    
    Output format:
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
    description="Generate a video from pre-existing assets (audio.wav, subtitles.srt, rich.json). The input directory must contain these files."
)
async def generate_video_from_assets(
    input_dir: str = Field(description="Directory containing audio.wav, subtitles.srt, and rich.json files"),
    output_video: Optional[str] = Field(default="public/output.mp4", description="Path for the output video file"),
    concurrency: Optional[int] = Field(default=6, description="Rendering concurrency level (1-10)")
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
        "concurrency": 6
    }
    
    Output format:
    {
        "status": "success",
        "output_video": "public/output.mp4"
    }
    """
    try:
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
        
        # Validate concurrency
        if not 1 <= concurrency <= 10:
            concurrency = 6
            logger.warning("Concurrency out of range (1-10), using default value of 6")
        
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
    title="Generate Complete Video",
    description="Complete video generation pipeline: generate assets from script and then create video. This combines both asset generation and video creation in one step."
)
async def generate_complete_video(
    script: str = Field(description="The video script text containing \\Headline:, \\Text:, \\Figure: markers"),
    method: Optional[str] = Field(default="kokoro", description="Audio generation method: 'kokoro' (default), 'elevenlabs', or 'lmnt'"),
    output_dir: Optional[str] = Field(default="public", description="Output directory for generated files"),
    output_video: Optional[str] = Field(default="public/output.mp4", description="Path for the output video file"),
    concurrency: Optional[int] = Field(default=6, description="Video rendering concurrency level (1-10)")
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
        "concurrency": 6
    }
    
    Output format:
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
        logger.info("Starting complete video generation pipeline...")
        
        # Step 1: Generate assets
        assets_result = await generate_video_assets(script, method, output_dir)
        
        if assets_result["status"] != "success":
            return assets_result
        
        # Step 2: Generate video from assets
        video_result = await generate_video_from_assets(output_dir, output_video, concurrency)
        
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
    description="Information about available audio generation methods",
    name="Audio Generation Methods",
)
def get_methods_info() -> str:
    """Get information about all available audio generation methods"""
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

### Generate Assets Only
Use `generate_video_assets` to create audio, subtitles, and rich content files.

### Generate Video from Existing Assets  
Use `generate_video_from_assets` when you already have the required files.

### Complete Pipeline
Use `generate_complete_video` for end-to-end video generation from script.
"""

@mcp.resource(
    uri="video://info/requirements",
    description="System requirements and file formats",
    name="Requirements",
)
def get_requirements_info() -> str:
    """Get information about system requirements"""
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
"""

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
