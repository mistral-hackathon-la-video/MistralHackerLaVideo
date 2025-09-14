
# Core imports needed for MCP server startup
import os
import logging
from typing import List
from pydantic import Field, BaseModel, model_validator
from enum import Enum
import re


logger = logging.getLogger(__name__)


class ScriptComponentType(str, Enum):
    TEXT = "Text"
    HEADLINE = "Headline"


class ScriptComponent(BaseModel):
    component_type: str = Field(
        ...,
        description="Type of script component - either 'Text' or 'Headline'"
    )
    content: str = Field(
        ...,
        description="""Content of the component. 
        For Headlines: Write as complete, natural sentences that could be spoken aloud. 
        Avoid title-case or abbreviated phrases. Use conversational, engaging language.
        For Text: Regular narrative content for the video script."""
    )
    position: int = Field(
        ...,
        description="Position of the component in the script"
    )


def reconstruct_script(script: BaseModel) -> str:
    """
    Reconstruct the script text from VistralScript model.
    
    Args:
        script (VistralScript): Validated script model
        
    Returns:
        str: Formatted script text
    
    Example:
        >>> script = VistralScript(
        ...     title="Understanding GPT-4",
        ...     paper_id="2405.11273",
        ...     target_duration_minutes=5.0,
        ...     components=[
        ...         ScriptComponent(component_type="Headline", content="Understanding GPT-4", position=0),
        ...         ScriptComponent(component_type="Text", content="Welcome to this review!", position=1)
        ...     ]
        ... )
        >>> print(reconstruct_script(script))
        \\Headline: Understanding GPT-4
        \\Text: Welcome to this review!
    """
    return '\n'.join(f"\\{comp.component_type.strip()}: {comp.content}" 
                    for comp in script.components)


def generate_model_with_context_check_podcast(paper_content : str):
    class VistralScript(BaseModel):
        title: str = Field(
            ...,
            description="Title of the research paper",
            examples=[
                "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts",
                "Attention Is All You Need",
                "BERT: Pre-training of Deep Bidirectional Transformers"
            ]
        )
        target_duration_minutes: float = Field(
            ...,
            ge=0,
            le=6,
            description="Target video duration in minutes",
            examples=[5.0, 5.5, 6.0]
        )
        components: List[ScriptComponent] = Field(
            ...,
            description="List of script components",
            examples=[[
                {
                    "component_type": "Headline",
                    "content": "Let's explore how GPT-4 revolutionizes language modeling with advanced techniques",
                    "position": 0
                },
                {
                    "component_type": "Text",
                    "content": "Today we're diving deep into the revolutionary GPT-4 model and understanding what makes it so powerful.",
                    "position": 1
                },

            ]]
        )

        @model_validator(mode='after')
        def validate_script_structure(cls,values):
            errors = []

            components = values.components

            

            if not components:
                errors.append(ValueError("Script must contain at least one component"))
            else:

                #links are not autorized in Text component
                for comp in components:
                    if comp.component_type.strip() == ScriptComponentType.TEXT:
                        if "https://" in comp.content:
                            errors.append(ValueError("https://... is forbidden from the ouput. DO NOT USE FIGURES. DO NOT USE LINKS. LINKS ARE FORBIDDEN FROM THE OUPUT"))

                sorted_components = sorted(components, key=lambda x: x.position)
                
                positions = [comp.position for comp in sorted_components]
                if positions != list(range(len(positions))):
                    errors.append(ValueError("Component positions must be consecutive integers starting from 0"))

                if sorted_components[0].component_type.strip() != ScriptComponentType.HEADLINE:
                    errors.append(ValueError("Script must start with a Headline component"))
                
                for i in range(1, len(sorted_components)):
                    if (sorted_components[i].component_type.strip() == sorted_components[i-1].component_type.strip() and 
                        sorted_components[i].component_type.strip() != ScriptComponentType.TEXT):
                        errors.append(ValueError(f"Consecutive {sorted_components[i].component_type.strip()} components are not allowed"))

                values.components = sorted_components
            

            for comp in values.components:
                # if comp.component_type.strip() == ScriptComponentType.FIGURE:
                #     # More lenient figure validation - only check if it looks like a URL
                #     if not (comp.content.startswith('http') or comp.content.startswith('/')): 
                #         errors.append(ValueError(f"Figure content should be a valid URL or file path: {comp.content}"))
                #     # Skip figure link accessibility check for now to avoid network issues
                    
                if comp.component_type.strip() not in ["Text",  "Headline"]:
                    errors.append(ValueError(f"""{comp.component_type.strip()} is not a valid component_type.
                             Type of autorized script component
                                    Only one of : 
                                    - Text 
                                    - Headline"""))
                    logger.info(errors[-1])
            if errors:
                print(errors)
                logger.info(errors)
                raise ValueError(errors)
            return values
    return VistralScript


