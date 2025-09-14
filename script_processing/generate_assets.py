import asyncio
import os
import sys
import tempfile
from typing import Literal
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs
import pandas as pd
import whisper
import torch
import torchaudio
import srt
from datetime import timedelta
from pathlib import Path
import logging
import time

from kokoro import KPipeline
import whisper
import torch
import torchaudio



import traceback
import soundfile as sf
import shutil

try :
    import mlx_whisper
except ImportError:
    mlx_whisper = None
    print("mlx_whisper not available")


from script_processing.type import Text, Caption, Figure, Equation, Headline, RichContent, CreatedIllustration

logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()



VOICE_MAPPING = {
    "female": "cgSgspJ2msm6clMCkdW9",
    "male" : "EXAVITQuxST5mfrX0gnT",
}


def _parse_script(script: str) -> list[RichContent | Text]:
    """Parse the script and return a list of RichContent or Text objects

    Parameters
    ----------
    script : str
        The script to parse as a string

    Returns
    -------
    list[RichContent | Text]
        List of RichContent or Text objects
    """
    lines = script.split("\n")
    content = []
    # For each line, parse it and create the corresponding object
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        elif line.startswith(r"\Figure: "):
            figure_content = line.replace(r"\Figure: ", "")
            figure = Figure(content=figure_content)
            content.append(figure)
        elif line.startswith(r"\Text: "):
            text_content = line.replace(r"\Text: ", "")
            text = Text(content=text_content)
            content.append(text)
        elif line.startswith(r"\Equation: "):
            equation_content = line.replace(r"\Equation: ", "")
            equation = Equation(content=equation_content)
            content.append(equation)
        elif line.startswith(r"\Headline: "):
            headline_content = line.replace(r"\Headline: ", "")
            headline = Headline(content=headline_content)
            content.append(headline)
        elif line.startswith(r"\CreatedIllustration: "):
            created_illustration_content = line.replace(r"\CreatedIllustration: ", "")
            created_illustration = CreatedIllustration(content=created_illustration_content)
            content.append(created_illustration)
        else:
            logger.warning(f"Unknown line: {line}")
    return content


def _make_caption_whisper(result: dict) -> list[Caption]:
    """Create a list of Caption objects from the result of the whisper model

    Parameters
    ----------
    result : dict
        Result dictionary from the whisper model

    Returns
    -------
    list[Caption]
        List of Caption objects
    """
    captions: list[Caption] = []
    for segment in result["segments"]:
        for word in segment["words"]:
            _word = word["word"]
            # Remove leading space if there is one
            if _word.startswith(" "):
                _word = _word[1:]
            caption = Caption(word=_word, start=word["start"], end=word["end"])
            captions.append(caption)
    return captions

def _generate_audio_and_caption_kokoro(
    script_contents: list[RichContent | Text],
    temp_dir: Path = Path(tempfile.gettempdir()),
    offset: float = 0.5,
) -> list[RichContent | Text]:
    """Generate audio and caption for each text segment in the script using Kokoro TTS

    Parameters
    ----------
    script_contents : list[RichContent  |  Text]
        List of RichContent or Text objects
    temp_dir : Path, optional
        Temporary directory to store the audio files, by default Path(tempfile.gettempdir())
    offset : float, optional
        Offset between each text segment, by default 0.5

    Returns
    -------
    list[RichContent | Text]
        List of RichContent or Text objects with audio and caption
    """
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
    
    # Initialize Kokoro pipeline with American English
    pipeline = KPipeline(lang_code='a')  # 'a' for American English
    
    # If the temp directory does not exist, create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:
        for i, script_content in enumerate(script_contents):
            match script_content:
                case RichContent(content=content):
                    pass
                case Text(content=content, audio=None, captions=None):
                    logger.info(f"Generating audio and caption for text {i} using Kokoro")
                    audio_path = (temp_dir / f"audio_{i}.wav").absolute().as_posix()
                    
                    logger.info(f"Generating audio {i} at {audio_path}")
                    
                    # Generate audio using Kokoro
                    generator = pipeline(content, voice='af_heart', speed=1.0)
                    
                    # Get the first (and typically only) result from the generator
                    for j, (gs, ps, audio) in enumerate(generator):
                        if j == 0:  # Take the first result
                            # Save audio using soundfile with 24kHz sample rate (Kokoro default)
                            sf.write(audio_path, audio, 24000)
                            break
                    
                    # Load audio for duration calculation
                    audio_tensor, sr = torchaudio.load(audio_path)
                    total_audio_duration = audio_tensor.size(1) / sr
                    total_audio_duration += offset
                    
                    script_content.audio_path = audio_path
                    script_content.end = total_audio_duration
                    

                    if (sys.platform == 'darwin'
                        and hasattr(os, 'uname') 
                        and os.uname().machine in ('arm64', 'aarch64')
                        and mlx_whisper is not None):
                        result = mlx_whisper.transcribe(audio=audio_path, word_timestamps=True)
                        script_content.captions = _make_caption_whisper(result)
                        
                    logger.info(
                        f"Generated audio and caption for text {i}, duration: {total_audio_duration}"
                    )

    except Exception as e:
        logger.error(f"Error generating audio and caption with Kokoro: {e}, {traceback.format_exc()}")
        raise e

    offset_fix = 0
    # Initially all text caption start at time 0
    # We need to offset them by the end of the previous text
    for i, script_content in enumerate(script_contents):
        if not (isinstance(script_content, Text)):
            continue
        if not script_content.captions:
            continue
        for caption in script_content.captions:
            caption.start += offset_fix
            caption.end += offset_fix
        script_content.start = offset_fix
        if script_content.end:
            script_content.end = script_content.end + offset_fix
        else:
            script_content.end = script_content.captions[-1].end
        offset_fix = script_content.end
    return script_contents



def _generate_audio_and_caption_elevenlabs(
    script_contents: list[RichContent | Text],
    temp_dir: Path = Path(tempfile.gettempdir()),
    voice: Literal["female", "male"] = "female",
    podcast: bool = False,
) -> list[RichContent | Text]:
    """Generate audio and caption for each text segment in the script.
    Use Whisper model to generate the captions

    Parameters
    ----------
    script_contents : list[RichContent  |  Text]
        List of RichContent or Text objects
    temp_dir : Path, optional
        Temporary directory to store the audio files, by default Path(tempfile.gettempdir())

    Returns
    -------
    list[RichContent | Text]
        List of RichContent or Text objects with audio and caption
    """
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    # If the temp directory does not exist, create it
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # For each segment,
    # if it is a rich content, do nothing
    # if it is a text, generate audio and caption and store them in the object
    try:
        for i, script_content in enumerate(script_contents):
            match script_content:
                case RichContent(content=content):
                    pass
                case Text(content=content, audio=None, captions=None):
                    audio_path = (temp_dir / f"audio_{i}.wav").absolute().as_posix()#TODO change this for a more unique filename
                    # If audio_path don't exist, generate it
                    #if not os.path.exists(audio_path):
                    logger.info(f"Generating audio {i} at {audio_path}")
                    script_content.audio = elevenlabs_client.text_to_speech.convert(
                        text=content,
                        voice_id=VOICE_MAPPING[voice],
                        voice_settings=VoiceSettings(
                            stability=0.5,
                            similarity_boost=0.8,
                            style=0.0,
                            use_speaker_boost=True,
                        ),
                        model_id="eleven_v3",
                    )
                    save(script_content.audio, audio_path)


                    ##TODO USE vostral
                    if (sys.platform == 'darwin'
                    and hasattr(os, 'uname') 
                    and os.uname().machine in ('arm64', 'aarch64')
                    and mlx_whisper is not None
                    and not podcast):
                        result = mlx_whisper.transcribe(audio=audio_path,word_timestamps=True)
                        script_content.captions = _make_caption_whisper(result)
                        
                        total_audio_duration = float(result["segments"][-1]["end"]-result["segments"][0]["start"])


                    script_content.audio_path = audio_path
                    script_content.end = total_audio_duration if not podcast else 0

    except Exception as e:

        logger.error(f"Error generating audio and caption: {e}, {traceback.format_exc()}")
        raise e

        time.sleep(1) ##not to be ratelimated by providers

    offset = 0
    # Initially all text caption start at time 0
    # We need to offset them by the end of the previous text
    for i, script_content in enumerate(script_contents):
        if not (isinstance(script_content, Text)):
            continue
        if not script_content.captions:
            continue
        for caption in script_content.captions:
            caption.start += offset
            caption.end += offset
        script_content.start = offset
        if script_content.end:
            script_content.end = script_content.end + offset
        else:
            script_content.end = script_content.captions[-1].end
        offset = script_content.end
    return script_contents



def fill_rich_content_time(
    script_contents: list[RichContent | Text],
) -> list[RichContent | Text]:
    """Fill the time for each rich content based on the text duration

    Parameters
    ----------
    script_contents : list[RichContent  |  Text]
        List of RichContent or Text objects

    Returns
    -------
    list[RichContent | Text]
        List of RichContent or Text objects with time assigned
    """
    # For each rich content, assign a time based on the text duration
    k = 0
    while k < len(script_contents):
        current_rich_content_group = []
        while k < len(script_contents) and not isinstance(script_contents[k], Text):
            current_rich_content_group.append(script_contents[k])
            k += 1

        if k >= len(script_contents):
            break

        next_text_group = []
        while k < len(script_contents) and isinstance(script_contents[k], Text):
            next_text_group.append(script_contents[k])
            k += 1

        if not next_text_group:
            break

        # Skip if there are no rich content elements to assign time to
        if not current_rich_content_group:
            continue

        total_duration = next_text_group[-1].end - next_text_group[0].start
        duration_per_rich_content = total_duration / len(current_rich_content_group)
        offset = next_text_group[0].start
        for i, rich_content in enumerate(current_rich_content_group):
            rich_content.start = offset + i * duration_per_rich_content
            rich_content.end = offset + (i + 1) * duration_per_rich_content
            # print(f"Asigning {rich_content.start} - {rich_content.end} to {rich_content}")
    return script_contents


def export_mp3(text_content: list[Text], out_path: str, offset: float = 0.5) -> None:
    """Export the audio of the text content to a single mp3 file

    Parameters
    ----------
    text_content : list[Text]
        List of Text objects
    out_path : str
        Path to save the mp3 file
    """
    # Merge all mp3 files into one
    audio_all = []
    for i, text in enumerate(text_content):
        if not text.audio_path:
            continue

        path = text.audio_path
        audio, sr = torchaudio.load(path)
        if offset > 0:
            # Add offset sec of silence between each audio
            silence = torch.zeros((1, int(sr * offset)))
            audio = torch.cat([audio, silence], dim=1)
        audio_all.append(audio)
    audio_all_torch = torch.cat(audio_all, dim=1)
    torchaudio.save(out_path, audio_all_torch, sr)


def export_srt(full_audio_path: str, out_path: str) -> None:
    """Export the SRT file for the full audio.
    We use the whisper model again to generate the caption for the full audio
    Because concatenating the caption of each text segment may not be accurate

    Parameters
    ----------
    full_audio_path : str
        Path to the full audio file
    out_path : str
        Path to save the SRT file
    """
    # Generate Caption for the full audio
    model = whisper.load_model("base.en")
    option = whisper.DecodingOptions(
        language="en", fp16=True, without_timestamps=False, task="transcribe"
    )
    result = model.transcribe(full_audio_path, word_timestamps=True)
    flatten_caption = _make_caption_whisper(result)
    # Generate SRT file from the caption
    subs = [
        srt.Subtitle(
            index=i,
            start=timedelta(seconds=t.start),
            end=timedelta(seconds=t.end),
            content=t.word
        )
        for i, t in enumerate(flatten_caption)
    ]
    srt_text = srt.compose(subs)
    # Write the SRT file
    with open(out_path, "w") as f:
        f.write(srt_text)


def export_rich_content_json(rich_content: list[RichContent], out_path: str) -> None:
    """Export the rich content to a json file.

    If a Figure has a local file path (e.g. "/Users/foo/bar/image.png") we copy the
    file next to the generated rich.json and rewrite the reference so that
    Remotion can fetch it through the temporary HTTP server (relative URL).
    Remote URLs (starting with http/https) are left untouched.
    """
    """Export the rich content to a json file

    Parameters
    ----------
    rich_content : list[RichContent]
        List of RichContent objects
    out_path : str
        Path to save the json file
    """
    # Prepare directory where we will write the JSON – we also copy any local
    # images next to it so they can be served by the temporary HTTP server.
    out_dir = Path(out_path).parent
    os.makedirs(out_dir, exist_ok=True)

    rich_content_dict = []
    for i, content in enumerate(rich_content):
        # If this is a local Figure (not starting with http/https), copy it next
        # to the json file and rewrite the reference to a relative URL that the
        # browser can fetch through http://localhost:<port>/.
        if isinstance(content, Figure):
            path_obj = Path(content.content)
            if path_obj.is_file() and not str(content.content).lower().startswith(("http://", "https://")):
                destination = out_dir / path_obj.name
                # Only copy if we haven't already.
                if not destination.exists():
                    shutil.copy(path_obj, destination)
                # Use only the filename in the JSON (relative URL).
                content.content = path_obj.name
        rich_content_dict.append({
            "type": content.__class__.__name__.lower(),
            "content": content.content,
            "start": content.start,
            "end": content.end,
        })

    df = pd.DataFrame(rich_content_dict)
    df.to_json(out_path, orient="records")


def generate_audio_and_caption(
    script: str
) -> list[RichContent | Text]:
    """Generate audio and caption for the script

    Parameters
    ----------
    script : str
        Script to generate audio and caption

    Returns
    -------
    list[RichContent | Text]
        List of RichContent or Text objects with audio and caption
    """
    script_contents = _parse_script(script)
    try:
        #script_contents = _generate_audio_and_caption_elevenlabs(script_contents)
        script_contents = _generate_audio_and_caption_kokoro(script_contents)
    except Exception as e:
        logger.error(f"Error generating audio and caption: {e}, {traceback.format_exc()}")
        raise e
    return script_contents

def generate_assets(
    script: str,
    method: Literal["elevenlabs", "lmnt", "kokoro"] = "kokoro",
    mp3_output: str = "public/audio.wav",
    srt_output: str = "public/subtitles.srt",
    rich_output: str = "public/rich.json",
) -> float:
    """Generate audio, caption, and rich content assets from script

    Parameters
    ----------
    script : str
        The video script
    method : "elevenlabs" | "lmnt" | "kokoro", optional
        The method to generate audio, by default "kokoro"
    mp3_output : str, optional
        The output mp3 file path, by default "public/audio.wav"
    srt_output : str, optional
        The output srt file path, by default "public/output.srt"
    rich_output : str, optional
        The output rich content json file path, by default "public/output.json

    Returns
    -------
    float
        The total duration of the audio
    """
    logger.info(f"Generating assets from script: {script}")

    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    logger.info(f"Created temporary directory: {temp_dir}")

    # Create parent directory for mp3_output, srt_output, and rich_output
    os.makedirs(os.path.dirname(mp3_output), exist_ok=True)
    os.makedirs(os.path.dirname(srt_output), exist_ok=True)
    os.makedirs(os.path.dirname(rich_output), exist_ok=True)

    # Generate audio and caption for each text content
    logger.info(f"Generating audio and caption for script: {script}")
    script_contents = generate_audio_and_caption(script)
    # Fill the time for each RichContent
    logger.info(f"Filling time for each RichContent: {script_contents}")
    script_contents = fill_rich_content_time(script_contents)

    # Separate rich content and text content
    rich_content = [c for c in script_contents if isinstance(c, RichContent)]
    text_content = [c for c in script_contents if isinstance(c, Text)]

    # Export mp3
    logger.info(f"Exporting mp3: {text_content}")
    export_mp3(text_content, mp3_output, offset=0.5)

    # Export srt
    logger.info(f"Exporting srt: {mp3_output}")
    export_srt(mp3_output, srt_output)

    # Export rich content
    logger.info(f"Exporting rich content: {rich_content}")
    export_rich_content_json(rich_content, rich_output)

    # Remove temp_dir
    logger.info(f"Removing temporary directory: {temp_dir}")
    temp_dir.cleanup()

    total_duration = text_content[-1].end if text_content[-1].end else 0
    return total_duration
