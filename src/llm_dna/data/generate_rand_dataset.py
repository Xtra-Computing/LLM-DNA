#!/usr/bin/env python3
"""
Generate random sentence dataset for DNA extraction.

Creates a dataset of 600 samples, each containing ~100 words of natural
English sentences built from random vocabulary using a context-free grammar.
Uses the wonderwords library to supply random nouns, verbs, and adjectives.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List
from wonderwords import RandomWord

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Context-free grammar rules. Each non-terminal maps to a list of possible
# expansions. Terminals starting with '_' are resolved by wonderwords.
_GRAMMAR = {
    "S": [
        "{NP} {VP}.",
        "{NP} {VP} and {NP} {VP}.",
        "When {NP} {VP}, {NP} {VP}.",
        "Although {NP} {VP}, {NP} {VP}.",
        "After {NP} {VP}, {NP} {VP}.",
        "If {NP} {VP}, {NP} will {_VERB_BASE}.",
        "{NP} {VP} because {NP} {VP}.",
        "{NP} {VP} while {NP} {VP}.",
        "{NP} {VP}, but {NP} {VP}.",
        "{NP} {VP} before {NP} {VP}.",
    ],
    "NP": [
        "the {_NOUN}",
        "the {_ADJ} {_NOUN}",
        "a {_NOUN}",
        "a {_ADJ} {_NOUN}",
        "the {_ADJ} {_ADJ2} {_NOUN}",
    ],
    "VP": [
        "{_VERB} {NP}",
        "{_VERB} {PP}",
        "{_VERB} {NP} {PP}",
        "{_VERB} {ADV}",
        "{_VERB}",
    ],
    "PP": [
        "near {NP}",
        "with {NP}",
        "beside {NP}",
        "across {NP}",
        "around {NP}",
        "behind {NP}",
        "toward {NP}",
        "inside {NP}",
        "above {NP}",
        "under {NP}",
    ],
    "ADV": [
        "quickly",
        "slowly",
        "carefully",
        "quietly",
        "suddenly",
        "gently",
        "eagerly",
        "steadily",
        "calmly",
        "gracefully",
    ],
}

# Maximum recursion depth to prevent infinite expansion
_MAX_DEPTH = 6


def _conjugate(verb: str) -> str:
    """Simple third-person present tense conjugation."""
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:] not in ("ay", "ey", "oy", "uy"):
        return verb[:-1] + "ies"
    return verb + "s"


def _expand(symbol: str, rw: RandomWord, depth: int = 0) -> str:
    """Recursively expand a grammar symbol into a string."""
    if depth > _MAX_DEPTH:
        # At max depth, return a simple terminal to stop recursion
        return rw.word(include_parts_of_speech=["nouns"])

    # Wonderwords terminals
    if symbol == "_NOUN":
        return rw.word(include_parts_of_speech=["nouns"])
    if symbol == "_VERB":
        return _conjugate(rw.word(include_parts_of_speech=["verbs"]))
    if symbol == "_VERB_BASE":
        return rw.word(include_parts_of_speech=["verbs"])
    if symbol in ("_ADJ", "_ADJ2"):
        return rw.word(include_parts_of_speech=["adjectives"])

    # Non-terminal: pick a random production and expand each token
    if symbol in _GRAMMAR:
        production = random.choice(_GRAMMAR[symbol])
        # Find all {TOKEN} references and expand them
        result = production
        while "{" in result:
            start = result.index("{")
            end = result.index("}", start)
            token = result[start + 1 : end]
            expanded = _expand(token, rw, depth + 1)
            result = result[:start] + expanded + result[end + 1 :]
        return result

    # Unknown symbol, return as literal
    return symbol


def _generate_sentence(rw: RandomWord) -> str:
    """Generate a single random sentence from the CFG."""
    sent = _expand("S", rw)
    return sent[0].upper() + sent[1:]


def generate_random_word_samples(
    num_samples: int = 600,
    words_per_sample: int = 100,
    seed: int = 42,
) -> List[str]:
    """
    Generate samples of natural English sentences with random vocabulary.

    Args:
        num_samples: Number of samples to generate
        words_per_sample: Target number of words per sample
        seed: Random seed for reproducibility

    Returns:
        List of strings, each containing ~words_per_sample words as sentences
    """
    random.seed(seed)
    rw = RandomWord()
    samples = []

    logger.info(f"Generating {num_samples} samples with ~{words_per_sample} words each...")

    for i in range(num_samples):
        sentences = []
        word_count = 0
        while word_count < words_per_sample:
            sent = _generate_sentence(rw)
            sentences.append(sent)
            word_count += len(sent.split())
        sample = " ".join(sentences)
        samples.append(sample)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")

    logger.info(f"Successfully generated {len(samples)} samples")
    return samples


def save_dataset(
    samples: List[str],
    output_file: Path,
    format: str = "json"
) -> None:
    """
    Save dataset to file.
    
    Args:
        samples: List of text samples
        output_file: Path to output file
        format: Output format ("json" or "txt")
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Save as JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(samples)} samples to {output_file} (JSON format)")
    elif format == "txt":
        # Save as text file, one sample per line
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_file} (TXT format)")
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate random word dataset for DNA extraction"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=600,
        help="Number of samples to generate (default: 600)"
    )
    parser.add_argument(
        "--words-per-sample",
        type=int,
        default=100,
        help="Number of words per sample (default: 100)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/rand/rand_dataset.json"),
        help="Output file path (default: data/rand/rand_dataset.json)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "txt"],
        default="json",
        help="Output format: json or txt (default: json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Generate samples
    samples = generate_random_word_samples(
        num_samples=args.num_samples,
        words_per_sample=args.words_per_sample,
        seed=args.seed
    )
    
    # Save dataset
    save_dataset(samples, args.output_file, args.format)
    
    logger.info("Dataset generation completed successfully!")


if __name__ == "__main__":
    main()

