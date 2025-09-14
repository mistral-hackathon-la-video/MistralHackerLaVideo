from openai import OpenAI
from typing import Literal, Any
import instructor
from instructor.core.hooks import Hooks, HookName
import requests
import os
import logging
import traceback
from dotenv import load_dotenv
import re
from src.schema.script_podcast import generate_model_with_context_check_podcast, reconstruct_script
from src.schema.script_video import generate_model_with_context_check_video


logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()

# Access the variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OCR_MODEL = os.getenv("OCR_MODEL")
OCR_PROVIDER = os.getenv("OCR_PROVIDER")
OCR_COORDINATE_EXTRACTOR_MODEL = os.getenv("OCR_COORDINATE_EXTRACTOR_MODEL")
OCR_PARSING_MODEL = os.getenv("OCR_PARSING_MODEL")
OCR_FIGURE_DETECTOR_MODEL = os.getenv("OCR_FIGURE_DETECTOR_MODEL")
SCRIPGENETOR_MODEL = os.getenv("SCRIPGENETOR_MODEL")
GEMINI_SCRIP_MODEL = os.getenv("GEMINI_SCRIP_MODEL")





def replace_keys_with_values(text, dict_list):
  """
  Replaces keys found in a text with their corresponding values from a list of dictionaries.

  Args:
    text: The input text string.
    dict_list: A list of dictionaries where keys are patterns to search for in the text 
               and values are the replacements.

  Returns:
    The modified text with keys replaced by values.
  """

  # Combine all dictionaries into a single dictionary for efficiency
  combined_dict = {}
  for d in dict_list:
    combined_dict.update(d)

  # Filter out empty keys to prevent KeyError
  combined_dict = {k: v for k, v in combined_dict.items() if k and k.strip()}
  
  # If no valid keys, return original text
  if not combined_dict:
    return text

  # Sort keys by length in descending order to handle overlapping keys correctly
  sorted_keys = sorted(combined_dict.keys(), key=len, reverse=True)

  # Build a regular expression pattern to match any of the keys
  # Escape special characters in keys for use in regex
  pattern = re.compile("|".join(map(re.escape, sorted_keys)))

  # Perform the replacement using re.sub with a lambda function
  modified_text = pattern.sub(lambda match: combined_dict.get(match.group(0), match.group(0)), text)

  return modified_text


SYSTEM_PROMPT_PODCAST = r"""
<context>
You're Vistral an AI Researcher and Content Creator on Youtube who specializes in summarizing academic papers.
The video will be uploaded on YouTube and is intended for a research-focused audience of academics, students, and professionals of the field of deep learning. 
</context>

<goal>
Generate a script for a mid-short video (5-6 minutes or less than 6000 words) on the research paper you will receve.
</goal>


<style_instructions>
The script should be engaging, clear, and concise, effectively communicating the content of the paper. 
The video should give a good overview of the paper in the least amount of time possible, with short sentences that fit well for a dynamic Youtube video.
The overall goal of the video is to make research papers more accessible and understandable to a wider audience, while maintaining academic rigor.
</style_instructions>

<format_instructions>
The script sould be formated following the followings rules below:
- Your ouput is a JSON with the following keys :
    - title: The title of the video.
    - paper_id: The id of the paper (e.g., '2405.11273') explicitly mensionned in the paper
    - target_duration_minutes : The target duration of the video
    - components : a list of component (component_type, content, position)
        - You should follow this format for each component: Text and Headline
        - The only autorized component_type are : Text,  and Headline
        - The Text will be spoken by a narrator and caption in the video.
        - For Headlines: Write complete, conversational sentences that sound natural when spoken. Avoid title-case phrases, abbreviations, or technical jargon. Use engaging, accessible language like "Let's explore how...", "Now we'll discover...", "Here's why this approach...", etc.
        - Avoid markdown listing (1., 2., or - dash) at all cost. Use full sentences that are easy to understand in spoken language.
        - Don't hallucinate figures.
        - Don't forget to maintain https:// as it is in the link.
</format_instructions>

Attention : 
- The paper_id in the precedent instruction are just exemples. Don't confuse it with the correct paper ID you ll receve. 
- keep the full link of the figure in the figure content value
- Always include at least one figure if present in the text. Viewers like when the video is animated and well commented. 3blue1brown Style


Here is an example of what you need to produce for paper id 2405.11273: 
<exemple>
{
    "title": "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts",
    "paper_id": "2405.11273",
    "target_duration_minutes": 5.5,
    "components": [
        {
            "component_type": "Headline",
            "content": "Today we're exploring how Uni-MoE creates a revolutionary approach to multimodal AI architectures",
            "position": 0
        },
        {
            "component_type": "Text",
            "content": "Welcome back to Vistral! Today, we’re diving into an exciting new paper titled "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts". This research addresses the challenge of efficiently scaling multimodal large language models (MLLMs) to handle a variety of data types like text, images, audio, and video.",
            "position": 1
        },
        {
            "component_type": "Text",
            "content": "Here’s a snapshot of the Uni-MoE model, illustrating its ability to handle multiple modalities using the Mixture of Experts (MoE) architecture. Let’s break down the main points of this paper.",
            "position": 3
        },
        {
            "component_type": "Headline",
            "content": "Let's understand why traditional scaling methods create significant computational challenges",
            "position": 4
        },
        {
            "component_type": "Text",
            "content": "Scaling multimodal models traditionally incurs high computational costs. Conventional models process each input with all model parameters, leading to dense and inefficient computations.",
            "position": 5
        },
        {
            "component_type": "Text",
            "content": "Enter the Mixture of Experts (MoE). Unlike dense models, MoE activates only a subset of experts for each input. This sparse activation reduces computational overhead while maintaining performance.",
            "position": 6
        },
        {
            "component_type": "Text",
            "content": "Previous works have used MoE in text and image-text models but limited their scope to fewer experts and modalities. This paper pioneers a unified MLLM leveraging MoE across multiple modalities.",
            "position": 7
        },
        ...
    ]
}
</exemple>


Your output is a JSON with the following structure : 

{
    "title": "...",
    "paper_id": "...",
    "target_duration_minutes": ...,
    "components": [
        {
            "component_type": "...",
            "content": "...",
            "position": ...
        },
        ...
    ]
}

Attention : No links are allowed. use no links.
Attention : USE NO EQUATIONS. Equations are forbidden from the output. DO NOT USE EQUATIONS. DO NOT USE $ or [ or \begin{...} or \end{...}. DO NOT MAKE TABLES.
https://... is forbidden from the ouput. DO NOT USE FIGURES. DO NOT USE LINKS. LINKS ARE FORBIDDEN FROM THE OUPUT
"""

SYSTEM_PROMPT_VIDEO = r"""
<context>
You're Vistral an AI Researcher and Content Creator on Youtube who specializes in summarizing academic papers.
The video will be uploaded on YouTube and is intended for a research-focused audience of academics, students, and professionals of the field of deep learning. 
</context>

<goal>
Generate a script for a mid-short video (5-6 minutes or less than 6000 words) on the research paper you will receve.
</goal>


<style_instructions>
The script should be engaging, clear, and concise, effectively communicating the content of the paper. 
The video should give a good overview of the paper in the least amount of time possible, with short sentences that fit well for a dynamic Youtube video.
The overall goal of the video is to make research papers more accessible and understandable to a wider audience, while maintaining academic rigor.
</style_instructions>

<format_instructions>
The script sould be formated following the followings rules below:
- Your ouput is a JSON with the following keys :
    - title: The title of the video.
    - paper_id: The id of the paper (e.g., '2405.11273') explicitly mensionned in the paper
    - target_duration_minutes : The target duration of the video
    - components : a list of component (component_type, content, position)
        - You should follow this format for each component: Text, Figure, Equation and Headline
        - The only autorized component_type are : Text, Figure, Equation and Headline
        - Figure, Equation (latex) and Headline will be displayed in the video as *rich content*, in big on the screen. You should incorporate them in the script where they are the most useful and relevant.
        - The Text will be spoken by a narrator and caption in the video.
        - Avoid markdown listing (1., 2., or - dash) at all cost. Use full sentences that are easy to understand in spoken language.
        - For Equation: Don't use $ or [, the latex context is automatically detected.
        - For Equation: Always write everything in the same line, multiple lines will generate an error. Don't make table.
        - Don't hallucinate figures.
        - Don't forget to maintain https:// as it is in the link.
</format_instructions>

<example_figures>
![](https://arxiv.org/html/2405.11273/multi_od/files1/figure/moe_intro.png) is rendered as "https://arxiv.org/html/2405.11273/multi_od/files1/figure/moe_intro.png".
![](ar5iv.labs.arxiv.org//html/5643.43534/assets/x5.png) is rendered as "ar5iv.labs.arxiv.org//html/5643.43534/assets/x5.png"
<example_figures>
Attention : 
- The paper_id in the precedent instruction are just exemples. Don't confuse it with the correct paper ID you ll receve.
- Only extract figure that are present in the paper. Don't use the exemple links. 
- keep the full link of the figure in the figure content value
- Do not forget 'https://' a the start of the figure link.
- Always include at least one figure if present in the text. Viewers like when the video is animated and well commented. 3blue1brown Style


Here is an example of what you need to produce for paper id 2405.11273: 
<exemple>
{
    "title": "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts",
    "paper_id": "2405.11273",
    "target_duration_minutes": 5.5,
    "components": [
        {
            "component_type": "Headline",
            "content": "Uni-MoE: Revolutionary Multimodal Architecture",
            "position": 0
        },
        {
            "component_type": "Text",
            "content": "Welcome back to Vistral! Today, we’re diving into an exciting new paper titled "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts". This research addresses the challenge of efficiently scaling multimodal large language models (MLLMs) to handle a variety of data types like text, images, audio, and video.",
            "position": 1
        },
        {
            "component_type": "Figure",
            "content": "https://arxiv.org/html/2405.11273/multi_od/files1/figure/moe_intro.png",
            "position": 2
        },
        {
            "component_type": "Text",
            "content": "Here’s a snapshot of the Uni-MoE model, illustrating its ability to handle multiple modalities using the Mixture of Experts (MoE) architecture. Let’s break down the main points of this paper.",
            "position": 3
        },
        {
            "component_type": "Headline",
            "content": "The Problem with Traditional Scaling",
            "position": 4
        },
        {
            "component_type": "Text",
            "content": "Scaling multimodal models traditionally incurs high computational costs. Conventional models process each input with all model parameters, leading to dense and inefficient computations.",
            "position": 5
        },
        {
            "component_type": "Text",
            "content": "Enter the Mixture of Experts (MoE). Unlike dense models, MoE activates only a subset of experts for each input. This sparse activation reduces computational overhead while maintaining performance.",
            "position": 6
        },
        {
            "component_type": "Text",
            "content": "Previous works have used MoE in text and image-text models but limited their scope to fewer experts and modalities. This paper pioneers a unified MLLM leveraging MoE across multiple modalities.",
            "position": 7
        },
        ...
    ]
}
</exemple>


Your output is a JSON with the following structure : 

{
    "title": "...",
    "paper_id": "...",
    "target_duration_minutes": ...,
    "components": [
        {
            "component_type": "...",
            "content": "...",
            "position": ...
        },
        ...
    ]
}
"""

def adjust_links(paper_markdown: str, paper_id: str) -> str:
    """Adjust links in the paper markdown (placeholder function)."""
    # For now, just return the paper as-is
    # This function would normally process arxiv links
    return paper_markdown


def create_logging_hooks(tag: str = "instructor") -> Hooks:
    """Create hooks that log each failed attempt (completion + parse errors)."""
    hooks = Hooks()
    state: dict[str, Any] = {"kwargs": None, "response": None}

    def on_kwargs(*args: Any, **kwargs: Any) -> None:
        try:
            state["kwargs"] = {
                "model": kwargs.get("model"),
                "messages": kwargs.get("messages")
                or kwargs.get("contents")
                or kwargs.get("chat_history"),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "stream": kwargs.get("stream"),
            }
        except Exception:
            pass

    def on_response(response: Any) -> None:
        state["response"] = response

    def extract_text_from_response(resp: Any) -> str | None:
        try:
            if hasattr(resp, "choices") and resp.choices:
                choice0 = resp.choices[0]
                if hasattr(choice0, "message") and getattr(choice0.message, "content", None):
                    return str(choice0.message.content)
                if hasattr(choice0, "text") and getattr(choice0, "text", None):
                    return str(choice0.text)
        except Exception:
            return None
        return None

    def on_parse_error(error: Exception) -> None:
        model = None
        messages = None
        if isinstance(state.get("kwargs"), dict):
            model = state["kwargs"].get("model")
            messages = state["kwargs"].get("messages")
        raw_text = extract_text_from_response(state.get("response"))

        logger.error(f"[{tag}] Parse error: {error}")
        if model:
            logger.error(f"[{tag}] Model: {model}")
        if messages:
            try:
                user_prompt = None
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_prompt = m.get("content")
                if user_prompt:
                    excerpt = str(user_prompt)
                    logger.error(f"[{tag}] Prompt excerpt: {excerpt[:1000]}")
            except Exception:
                pass
        if raw_text:
            logger.error(f"[{tag}] Raw completion excerpt: {raw_text[:1000]}")

    def on_completion_error(error: Exception) -> None:
        logger.error(f"[{tag}] Completion error: {error}")

    def on_last_attempt(error: Exception) -> None:
        logger.error(f"[{tag}] Last attempt failed: {error}")

    hooks.on(HookName.COMPLETION_KWARGS, on_kwargs)
    hooks.on(HookName.COMPLETION_RESPONSE, on_response)
    hooks.on(HookName.PARSE_ERROR, on_parse_error)
    hooks.on(HookName.COMPLETION_ERROR, on_completion_error)
    hooks.on(HookName.COMPLETION_LAST_ATTEMPT, on_last_attempt)
    return hooks



def _process_script_openrouter(paper: str, mode: Literal["podcast", "video"] = "video") -> str:
    """Generate a video script using OpenRouter (OpenAI-compatible API).

    Uses the OpenAI SDK pointed to the OpenRouter base URL.
    """
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("SCRIPGENETOR_MODEL", "google/gemini-2.0-flash-001")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    if not OPENROUTER_API_KEY:
        raise ValueError("You need to set the OPENROUTER_API_KEY environment variable.")

    try:
        openrouter_client = instructor.from_openai(
            OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL),
            mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS if "gpt" not in OPENROUTER_MODEL else instructor.Mode.JSON_SCHEMA,
            hooks=create_logging_hooks("openrouter"),
        )
        
        # Try with reduced validation first
        response,raw = openrouter_client.chat.completions.create_with_completion(
            model=OPENROUTER_MODEL,
            messages=
            [
                {"role": "system", "content": SYSTEM_PROMPT_VIDEO if mode == "video" else SYSTEM_PROMPT_PODCAST},
                {
                    "role": "user",
                    "content": f"Here is the paper I want you to generate a script from : "
                    + paper,
                },
            ],
            response_model=generate_model_with_context_check_podcast(paper) if mode == "podcast" else generate_model_with_context_check_video(paper),
            temperature=0,  # Slightly higher temperature to avoid getting stuck
            max_retries=3,    # Reduced retries to fail faster
            max_tokens=8000,
        )
        
        if not response:
            raise ValueError("Empty response received from model")
            
    except Exception as e:
        print(f"Error during script generation: {e}")
        # Try with a simpler prompt if the structured one fails
        raise ValueError(f"Script generation failed: {e}")

    try:
        result = reconstruct_script(response)
    except Exception as e:
        print(e)
        raise ValueError(f"The model failed the script generation:  {e}, {traceback.format_exc()}")
    return result



def process_script(method: Literal["openrouter"], paper_markdown: str, from_pdf: bool=False, mode: Literal["podcast", "video"] = "video") -> str:
    """Generate a video script for a research paper.

    Parameters
    ----------
    paper_markdown : str
        A research paper in markdown format.

    Returns
    -------
    str
        The generated video script.

    Raises
    ------
    ValueError
        If no result is returned from OpenAI.
    """


    pd_corrected_links = paper_markdown
    paper_id = "paper_id"
    if method == "openai":
        return _process_script_gpt(pd_corrected_links,paper_id)
    elif method == "local":
        return _process_script_open_source(pd_corrected_links, paper_id, end_point_base_url)
    elif method == "gemini":
        return _process_script_open_gemini(pd_corrected_links, paper_id, end_point_base_url)
    elif method == "groq":
        return _process_script_groq(pd_corrected_links,paper_id)
    elif method == "openrouter":
        return _process_script_openrouter(pd_corrected_links, mode=mode)
    else:
        raise ValueError(f"Invalid method '{method}'. Please choose 'openrouter', 'openai', 'gemini', 'groq', or 'local'.")


def _fetch_paper_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX, 5XX)
        #write in a htlm file
        with open(f"paper.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None


# def main():
#     # Example usage
#     url = "https://ar5iv.labs.arxiv.org/html/1706.03762"
#     paper_markdown = _fetch_paper_html(url)

#     paper_id = "1706.03762"
#     method = "openrouter"  # Change this to test other methods
#     mode = "podcast"

#     try:
#         script = process_script(method=method, paper_markdown=paper_markdown, paper_id=paper_id, from_pdf=False, mode="podcast")
#         print("Generated script:\n", script)
#     except ValueError as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()
