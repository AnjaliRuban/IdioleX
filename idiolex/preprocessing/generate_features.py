"""
Generate linguistic features using an LLM API.

This script adds feature vectors to processed data by prompting an LLM
to analyze dialectal features in each sentence.

Usage:
    python generate_features.py \
        --input_dir data/processed/train_data \
        --output_dir data/processed/train_data_feats \
        --batch_size 100

Environment variables:
    LITELLM_API_KEY: API key for LiteLLM
    LITELLM_API_BASE_URL: Base URL for API (optional)
"""

import argparse
import asyncio
import json
import os

from litellm import acompletion
from tqdm import tqdm

# Example dialect features to analyze
DIALECT_FEATURES = [
    # Phonological/Orthographic
    "nonstandard_spelling_rate",
    "accent_omission_rate",
    "vowel_reduplication_rate",
    "punctuation_repetition_rate",
    # Morphological
    "diminutive_suffix_rate",
    "augmentative_suffix_rate",
    "verb_form_variation_rate",
    # Lexical
    "regional_lexeme_density",
    "loanword_density",
    "colloquial_term_rate",
    "interjection_rate",
    "abbreviation_rate",
    # Syntactic
    "explicit_subject_pronoun_rate",
    "double_negation_rate",
    "discourse_marker_density",
    # Pragmatic/Stylistic
    "formality_level",
    "emoji_usage_rate",
    "code_switching_rate",
    "intensifier_usage_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add LLM-generated features to processed data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with processed JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for data with features.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for feature generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of sentences to process in parallel.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay between batches (seconds) to avoid rate limits.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to JSON file with custom feature list.",
    )
    return parser.parse_args()


def get_feature_list(args: argparse.Namespace) -> list[str]:
    """Get the list of features to analyze."""
    if args.features and os.path.exists(args.features):
        with open(args.features) as f:
            return json.load(f)
    return DIALECT_FEATURES


def parse_llm_response(response: str, features: list[str]) -> list[float]:
    """Parse LLM response into feature vector.

    Args:
        response: Raw LLM response (should be JSON).
        features: List of feature names.

    Returns:
        List of feature values (floats 0-1).
    """
    try:
        # Extract JSON from response
        if "{" in response and "}" in response:
            json_str = "{" + response.split("{", 1)[1].rsplit("}", 1)[0] + "}"
            data = json.loads(json_str)
            return [float(data.get(f, 0.0)) for f in features]
    except (json.JSONDecodeError, ValueError):
        pass

    # Return zeros on parse failure
    return [0.0] * len(features)


async def analyze_sentence(
    sentence: str,
    features: list[str],
    model: str,
) -> list[float]:
    """Analyze a single sentence for dialect features.

    Args:
        sentence: Text to analyze.
        features: List of feature names.
        model: LLM model name.

    Returns:
        Feature vector as list of floats.
    """
    prompt = (
        f"Analyze the following sentence for dialectal features. "
        f"Return a JSON object with these features as keys: {', '.join(features)}. "
        f"Values should be floats between 0.0 and 1.0 indicating feature presence.\n\n"
        f"Sentence: {sentence}\n\nJSON:"
    )

    try:
        response = await acompletion(
            api_key=os.environ.get("LITELLM_API_KEY"),
            base_url=os.environ.get("LITELLM_API_BASE_URL"),
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = response["choices"][0]["message"]["content"].strip()
        return parse_llm_response(content, features)
    except Exception as e:
        print(f"Error analyzing sentence: {e}")
        return [0.0] * len(features)


async def process_batch(
    sentences: list[str],
    features: list[str],
    model: str,
) -> list[list[float]]:
    """Process a batch of sentences in parallel.

    Args:
        sentences: List of sentences to analyze.
        features: List of feature names.
        model: LLM model name.

    Returns:
        List of feature vectors.
    """
    tasks = [analyze_sentence(s, features, model) for s in sentences]
    return await asyncio.gather(*tasks)


async def add_features_to_file(
    input_path: str,
    output_path: str,
    features: list[str],
    args: argparse.Namespace,
) -> None:
    """Add features to a single processed data file.

    Args:
        input_path: Path to input JSON file.
        output_path: Path to output JSON file.
        features: List of feature names.
        args: Command line arguments.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Collect all sentences
    sentences = []
    indices = []  # Track position in nested structure

    for user_idx, user_data in enumerate(data):
        for comment_idx, comment in enumerate(user_data):
            for sent_idx, sentence_data in enumerate(comment):
                text = sentence_data.get("text", "")
                if isinstance(text, list):
                    text = " ".join(text)
                sentences.append(text)
                indices.append((user_idx, comment_idx, sent_idx))

    print(f"  Processing {len(sentences)} sentences...")

    # Process in batches
    all_features = []
    for i in tqdm(range(0, len(sentences), args.batch_size)):
        batch = sentences[i : i + args.batch_size]
        batch_features = await process_batch(batch, features, args.model)
        all_features.extend(batch_features)

        if i + args.batch_size < len(sentences):
            await asyncio.sleep(args.delay)

    # Add features back to data structure
    for (user_idx, comment_idx, sent_idx), feat_vec in zip(
        indices, all_features, strict=False
    ):
        data[user_idx][comment_idx][sent_idx]["feature"] = feat_vec

    # Save
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def main_async(args: argparse.Namespace) -> None:
    """Async main function."""
    features = get_feature_list(args)
    print(f"Using {len(features)} features")

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if not filename.endswith(".json"):
            continue

        print(f"\nProcessing {filename}...")
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        await add_features_to_file(input_path, output_path, features, args)
        print(f"  Saved -> {output_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
