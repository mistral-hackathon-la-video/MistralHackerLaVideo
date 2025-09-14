#!/usr/bin/env python3
"""
Example usage of the podcast generation system.

This script demonstrates how to:
1. Generate a script using generate_script.py
2. Convert the script to audio using generate_podcast.py
"""

import asyncio
from pathlib import Path

# Import our modules
from script_processing.generate_script import process_script, _fetch_paper_html
from script_processing.generate_assets import _generate_audio_and_caption_elevenlabs, export_mp3, _parse_script
from script_processing.generate_paper import process_article_firecrawl


def example_script_to_podcast():
    """Example with a sample script (no API calls needed)"""
    
    print("🎙️ Testing Script to Podcast Conversion...")
    
    # Sample script in the expected format
    sample_script = """\Headline: Let's explore how the Transformer architecture is revolutionizing sequence transduction models
\Text: Welcome to Vistral! Today, we're diving into the groundbreaking paper, 'Attention Is All You Need.' This paper introduces the Transformer, a novel architecture that replaces recurrent and convolutional layers with attention mechanisms for sequence transduction tasks.
\Text: The Transformer has set new standards in machine translation, showcasing superior quality and parallelization capabilities with significantly reduced training time.
\Headline: Let's understand the limitations of recurrent and convolutional neural networks
\Text: Traditional sequence transduction models rely on recurrent neural networks (RNNs) and convolutional neural networks (CNNs). These models process data sequentially, which limits parallelization, especially with long sequences.
\Text: Attention mechanisms have been integrated to model dependencies, but mostly in conjunction with RNNs. The Transformer, however, discards recurrence entirely, depending only on attention mechanisms.
\Headline: Now we'll discover the architecture of the Transformer
\Text: The Transformer follows an encoder-decoder structure but uses stacked self-attention and point-wise fully connected layers. The encoder and decoder are each composed of N=6 identical layers.
\Text: Each encoder layer contains a multi-head self-attention mechanism and a position-wise feed-forward network. Residual connections and layer normalization are employed around each sub-layer.
\Text: The decoder also uses multi-head attention over the encoder output. It includes masking to prevent attending to subsequent positions, ensuring auto-regressive properties.
\Headline: Let's explore how scaled dot-product attention works
\Text: The Transformer uses scaled dot-product attention, computing the dot products of queries and keys, scaled by the square root of the dimension, and applying a softmax function to obtain the weights.
\Text: Multi-head attention linearly projects queries, keys, and values multiple times with different learned projections. This allows the model to attend to information from different representation subspaces.
\Headline: Here's why positional encoding is crucial for the Transformer
\Text: Since the Transformer lacks recurrence and convolution, positional encodings are added to input embeddings. These encodings use sine and cosine functions to provide information about the position of tokens in the sequence.
\Headline: Let's understand the advantages of self-attention over recurrent and convolutional layers
\Text: Self-attention connects all positions with a constant number of operations, enabling more parallelization compared to the O(n) sequential operations required by recurrent layers.
\Text: Self-attention layers can be faster than recurrent layers when the sequence length is smaller than the representation dimensionality, a common scenario in machine translation.
\Headline: Now we'll see how the Transformer was trained and the results achieved
\Text: The models were trained on the WMT 2014 English-German and English-French datasets. The Adam optimizer was used with a learning rate schedule that increases linearly for the first 4000 steps and decreases proportionally to the inverse square root of the step number.
\Text: The Transformer achieved state-of-the-art BLEU scores on both translation tasks, outperforming previous models with significantly less training cost. For English-to-German, it achieved 28.4 BLEU, and for English-to-French, 41.8 BLEU.
\Headline: Let's see how the Transformer generalizes to constituency parsing
\Text: The Transformer was also applied to English constituency parsing, demonstrating its generalization capabilities. It achieved competitive results, even outperforming the BerkeleyParser when trained only on the WSJ training set.
\Headline: In conclusion, the Transformer architecture has revolutionized sequence transduction
\Text: The Transformer, with its exclusive reliance on attention mechanisms, offers superior translation quality, increased parallelization, and reduced training time. This work paves the way for future attention-based models in various tasks.
\Text: Thanks for watching! Don't forget to like and subscribe for more deep dives into the latest research papers."""
    
    try:
        # Generate podcast with sample script
        sample_script = _parse_script(sample_script)
        podcast_path = _generate_audio_and_caption_elevenlabs(sample_script)
        export_mp3(podcast_path, "sample_podcast.wav")
        
        print(f"✅ Sample podcast generated: {podcast_path}")
        
    except Exception as e:
        print(f"❌ Sample podcast generation failed: {e}")

def main():
    """Main entry point with options."""

    example_script_to_podcast()

if __name__ == "__main__":
    main()
    