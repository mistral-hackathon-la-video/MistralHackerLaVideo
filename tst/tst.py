import sys
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processing.generate_video import process_video
from script_processing.generate_assets import generate_assets

logger = logging.getLogger(__name__)


script = """\Headline: Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts
\Text: Welcome back to Arxflix! Today, we’re diving into a groundbreaking paper that explores new ways to scale Unified Multimodal Large Language Models (MLLMs) using the Mixture of Experts (MoE) architecture. The paper is titled "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts".

\Figure: https://arxiv.org/html/2405.11273v1/extracted/5604403/figure/moe_intro.png
\Text: This illustration provides a snapshot of the Uni-MoE model, highlighting its efficient handling of various modalities like text, images, audio, and video. Let’s break it down.

\Headline: The Challenge
\Text: Scaling multimodal models traditionally incurs high computational costs. Conventional models process each input with all model parameters, leading to dense and inefficient computations.

\Text: Enter the Mixture of Experts (MoE). Unlike dense models, MoE activates only a subset of experts for each input. This sparse activation reduces computational overhead while maintaining performance.

\Text: Previous works have used MoE in text and image-text models but limited their scope to fewer experts and modalities. This paper pioneers a unified MLLM leveraging MoE across multiple modalities.

\Headline: Uni-MoE Architecture
\Text: Uni-MoE introduces a sophisticated architecture featuring modality-specific encoders and connectors. These map diverse modalities into a unified language representation space.

\Figure: https://arxiv.org/html/2405.11273v1/extracted/5604403/figure/model.png
\Text: Here’s an overview of the training methodology for Uni-MoE. The progressive training stages ensure efficient cross-modality alignment and expert tuning.

\Headline: Three-Stage Training Strategy
\Text: The training process for Uni-MoE is divided into three stages. Firstly, cross-modality alignment involves training connectors to map different modalities into a unified language space. Secondly, modality-specific expert training refines each expert’s proficiency within its domain. Lastly, unified MoE training integrates all trained experts and fine-tunes them using Low-Rank Adaptation (LoRA).

\Figure: https://arxiv.org/html/2405.11273v1/extracted/5604403/figure/loss_curve.png
\Text: This figure shows the loss curves for various MoE settings. Notice how the variant with more experts achieves more stable convergence.

\Headline: Evaluation and Results
\Text: Uni-MoE was evaluated on extensive benchmarks, including image-text, video, and audio/speech datasets. The model significantly reduced performance bias and improved multi-expert collaboration.

\Figure: https://arxiv.org/html/2405.11273v1/extracted/5604403/figure/cap/cap_text_audio_v1.png
\Text: This distribution shows expert loading with various cross-modality inputs, demonstrating how Uni-MoE efficiently handles different data types.

\Headline: Key Contributions
\Text: The paper’s key contributions include: Firstly, the framework is unified and integrates multiple modalities with modality-specific encoders. Secondly, it employs a progressive training strategy that enhances expert collaboration and generalization. Lastly, extensive benchmarks have showcased the model’s superior performance in complex multimodal tasks.

\Headline: Conclusion
\Text: Uni-MoE showcases the potential of MoE frameworks in advancing multimodal large language models. By efficiently managing computational resources and leveraging specialized experts, it sets a new standard for multimodal understanding.

\Text: For more detailed insights, check out the paper and the code available on GitHub. Thanks for watching, and don’t forget to like, subscribe, and hit the bell icon for more research updates from Arxflix!
"""



def generate_video(
    input_dir: str,
    output_video: str,
):
    """Generate video from input directory.
    The input directory should contain subtitles.srt, audio.wav, and rich.json files.

    Parameters
    ----------
    input_dir : str
        The input directory containing subtitles.srt, audio.wav, and rich.json
    output_video : str
        Path of the output video
    """
    _input_dir = Path(input_dir)
    _output_video = Path(output_video)

    logger.info(f"Generating video to {_output_video.name} from {_input_dir.name}")

    if not _input_dir.exists() or not _input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {_input_dir} does not exist")

    if not (_input_dir / "subtitles.srt").exists():
        raise FileNotFoundError(f"Subtitles file does not exist in {_input_dir}")
    if not (_input_dir / "audio.wav").exists():
        raise FileNotFoundError(f"Audio file does not exist in {_input_dir}")
    if not (_input_dir / "rich.json").exists():
        raise FileNotFoundError(f"Rich content file does not exist in {_input_dir}")

    process_video(_input_dir, _output_video)




if __name__ == "__main__":
    generate_assets(script, "kokoro")
    generate_video("public", "public/output.mp4")
    logger.info(f"Generated video to public/output.mp4")