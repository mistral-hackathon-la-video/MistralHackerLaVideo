import requests
import os
import urllib.request
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from typing import Any, Literal
from markthat import MarkThat
from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OCR_MODEL = os.getenv("OCR_MODEL")
OCR_PROVIDER = os.getenv("OCR_PROVIDER")
OCR_COORDINATE_EXTRACTOR_MODEL = os.getenv("OCR_COORDINATE_EXTRACTOR_MODEL")
OCR_PARSING_MODEL = os.getenv("OCR_PARSING_MODEL")
OCR_FIGURE_DETECTOR_MODEL = os.getenv("OCR_FIGURE_DETECTOR_MODEL")


def url_to_markdown_firecrawl(url: str) -> str:
    """Convert a URL to markdown using Firecrawl.
    
    This is a standalone function that uses Firecrawl to scrape a URL
    and convert its content to markdown format.

    Args:
        url (str): The URL to convert to markdown.

    Returns:
        str: The content of the URL as markdown.
        
    Raises:
        ValueError: If FIRECRAWL_API_KEY is not set or if scraping fails.
    """
    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY environment variable is not set")
    
    try:
        app = Firecrawl(api_key=FIRECRAWL_API_KEY)
        
        # Scrape the URL and convert to markdown
        scrape_result = app.scrape(url, formats=["markdown"])
        
        if scrape_result and hasattr(scrape_result, 'markdown') and scrape_result.markdown:
            return scrape_result.markdown
        else:
            raise ValueError("Failed to extract markdown from the URL")
            
    except Exception as e:
        raise ValueError(f"Error processing URL with Firecrawl: {str(e)}")


def process_article_firecrawl(url: str) -> str:
    """Process an article from a URL using Firecrawl to generate markdown.

    Args:
        url (str): The URL of the article to process.

    Returns:
        str: The processed article as a markdown string.
    """
    return url_to_markdown_firecrawl(url)


def process_article(method: Literal["pdf", "link"], paper_id: str = None, pdf_path: str = None, url: str = None) -> str:
    """Process an article from a given URL and save it as a markdown file.

    Args:
        method (Literal["pdf", "link"]): The method to use for processing the article.
        paper_id (str, optional): The paper_id of the article (for pdf method).
        pdf_path (str, optional): The path to the PDF file (for pdf method).
        url (str, optional): The URL to process (for link method).

    Returns:
        str: The processed article as a markdown string.
    """
    if method == "link":
        if not url:
            raise ValueError("URL is required for link method")
        return process_article_firecrawl(url)
    elif method == "pdf":
        import asyncio
        client = MarkThat(provider=OCR_PROVIDER, model=OCR_MODEL,api_key=OPENROUTER_API_KEY,
                  api_key_figure_detector=OPENROUTER_API_KEY,
                  api_key_figure_extractor=OPENROUTER_API_KEY,
                  api_key_figure_parser=OPENROUTER_API_KEY)
        result = asyncio.run(client.async_convert(pdf_path, extract_figure=True,
                                    figure_detector_model=OCR_FIGURE_DETECTOR_MODEL,
                                    coordinate_model=OCR_COORDINATE_EXTRACTOR_MODEL,
                                    parsing_model=OCR_PARSING_MODEL,
                                    ))
        return "\n".join(result)
    else:
        raise ValueError(
            "Invalid article method. Please choose either 'pdf', or 'link'."
        )
