import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from time import sleep
from typing import Literal
import json
import socket
import re

VIDEO_FPS = 30
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
REMOTION_ROOT_PATH = Path("video_remotion/src/remotion/index.ts")
REMOTION_COMPOSITION_ID = "LaVideo"
# Consider making this configurable or determining it via benchmarking
OPTIMAL_CONCURRENCY = 6 

logger = logging.getLogger(__name__)


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    free_port = str(sock.getsockname()[1])
    sock.close()
    return free_port


@dataclass
class CompositionProps:
    durationInSeconds: int = 5
    subtitlesFileName: str = "video_remotion/public/output.srt"
    audioFileName: str = "video_remotion/public/output.wav"
    richContentFileName: str = "video_remotion/public/output.json"
    waveColor: str = "#a3a5ae"
    subtitlesLinePerPage: int = 2
    subtitlesLineHeight: int = 98
    subtitlesZoomMeasurerSize: int = 10
    onlyDisplayCurrentSentence: bool = True
    mirrorWave: bool = False
    waveLinesToDisplay: int = 300
    waveFreqRangeStartIndex: int = 5
    waveNumberOfSamples: Literal["32", "64", "128", "256", "512"] = "512"
    durationInFrames: int = field(init=False)

    def __post_init__(self):
        self.durationInFrames: int = self.durationInSeconds * VIDEO_FPS + 7 * VIDEO_FPS


def expose_directory(directory: Path):
    # pnpx http-server --cors -a localhost -p 8080
    subprocess.run(
        [
            "pnpx",
            "http-server",
            "--cors",
            "-a",
            "localhost",
            "-p",
            "8080",
        ],
        cwd=directory.absolute().as_posix(),
    )


def process_video(
    input: Path,
    output: Path = Path("video_remotion/public/output.mp4"),
    concurrency: int = OPTIMAL_CONCURRENCY,
):
    # Get the paper id,
    # Pick an available port,
    free_port = get_free_port()
    print(f"Free port: {free_port}")
    # Ensure that figures inside the Rich Content JSON can be fetched by the Remotion bundle.
    # If a Figure has a local filename (e.g. "figure_1.png"), we prefix it with the URL of the
    # temporary static server so that the browser inside Remotion can retrieve it over HTTP.
    rich_json_path = input / "rich.json"
    if rich_json_path.exists():
        try:
            # Replace port numbers in URLs (pattern: :port/) with the free port
            def replace_port(match):
                return f":{free_port}/"

            content = rich_json_path.read_text()
            content = re.sub(r":\d+/", replace_port, content)
            data = json.loads(content)
            # The JSON is expected to be a list of dicts.
            for item in data:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "figure"
                    and isinstance(item.get("content"), str)
                    and not item["content"].lower().startswith(("http://", "https://"))
                ):
                    # Prefer IPv4 to avoid environments where localhost resolves to ::1
                    item["content"] = f"http://127.0.0.1:{free_port}/{item['content']}"
            rich_json_path.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to rewrite {rich_json_path} with absolute URLs: {e}")
    with subprocess.Popen(
        [
            "pnpx",
            "http-server",
            input.absolute().as_posix(),
            "--cors",
            "-a",
            "0.0.0.0",
            "-p",
            free_port,
        ],
        cwd=input.absolute().as_posix(),
        stdout=subprocess.DEVNULL, # Hide http-server logs
        stderr=subprocess.DEVNULL,
    ) as static_server:
        print(f"Exposed directory {input}")
        sleep(2)
        logger.info(f"Exposed directory {input}")
        # Use IPv4 to avoid environments where localhost resolves to IPv6 ::1
        base_url = f"http://127.0.0.1:{free_port}"
        composition_props = CompositionProps(
            subtitlesFileName=f"{base_url}/subtitles.srt",
            audioFileName=f"{base_url}/audio.wav",
            richContentFileName=f"{base_url}/rich.json",
        )
        logger.info(f"Generating video to {output}")
        
        render_command = [
            "npx",
            "remotion",
            "render",
            REMOTION_ROOT_PATH.absolute().as_posix(),
            "--props",
            json.dumps(asdict(composition_props)),
            "--compositionId",
            REMOTION_COMPOSITION_ID,
            "--concurrency",
            str(concurrency),
            "--gl",
            "angle",
            "--hardware-acceleration",
            "if-possible",
            "--output",
            output.absolute().as_posix(),
        ]

        logger.info(f"Running command: {' '.join(render_command)}")

        # Use Popen and let it write directly to the terminal.
        # This allows Remotion to render its interactive progress bar.
        render_proc = subprocess.Popen(
            render_command,
            cwd=Path("video_remotion").absolute().as_posix(),
        )

        # Wait for the process to complete and get the final return code
        return_code = render_proc.wait()
        
        static_server.terminate()
        
        if return_code != 0:
            raise RuntimeError(f"Remotion render failed with exit code {return_code}")
        if not output.exists():
            raise FileNotFoundError(str(output))
            
        logger.info(f"Generated video to {output}")
        return output