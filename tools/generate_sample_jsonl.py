"""Generate a sample JSONL file for testing the MinText data pipeline.

Creates synthetic documents with varied text for end-to-end testing.

End-to-end workflow:

    # 1. Generate sample data
    python tools/generate_sample_jsonl.py --output /tmp/demo/sample.jsonl

    # 2. Convert to arecord (needs a tokenizer)
    python tools/text_to_arecord.py \
        --input /tmp/demo/sample.jsonl \
        --output /tmp/demo/arecord \
        --tokenizer-path gpt2 \
        --max-file-size 1M

    # 3. Train
    python -m mintext.train \
        --config configs/base.yml \
        --data_path /tmp/demo/arecord \
        --dataset_type arecord \
        --steps 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


_TOPICS = [
    "machine learning",
    "distributed computing",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "optimization",
    "linear algebra",
    "probability theory",
    "information theory",
    "signal processing",
]

_TEMPLATES = [
    "The field of {topic} has seen remarkable progress in recent years. Researchers have developed new methods that significantly improve performance on standard benchmarks. These advances build on decades of foundational work in mathematics and computer science. The practical applications range from automated translation to scientific discovery.",
    "In this paper, we present a novel approach to {topic}. Our method combines several key insights from the literature with new theoretical contributions. We demonstrate state-of-the-art results on multiple evaluation tasks. The proposed framework is both efficient and scalable to large datasets.",
    "Understanding {topic} requires a solid foundation in both theory and practice. The core concepts include mathematical formulations, algorithmic design, and empirical evaluation. Students should begin by studying the fundamental principles before moving to advanced topics. Hands-on experience with real-world data is essential for developing intuition.",
    "Recent advances in {topic} have been driven by increased computational resources and larger datasets. The transformer architecture has proven particularly effective across many domains. Scaling laws suggest that performance continues to improve with model size and training data. However, important challenges remain in areas such as efficiency, robustness, and interpretability.",
    "A comprehensive survey of {topic} reveals several important trends. First, end-to-end learning has replaced traditional pipeline approaches in many applications. Second, self-supervised pre-training has become the dominant paradigm. Third, foundation models are being adapted to an increasingly diverse set of downstream tasks. These developments have profound implications for both research and industry.",
]


def generate_document(rng: random.Random) -> str:
    """Generate a single synthetic document."""
    topic = rng.choice(_TOPICS)
    num_paragraphs = rng.randint(1, 3)
    paragraphs = []
    for _ in range(num_paragraphs):
        template = rng.choice(_TEMPLATES)
        paragraphs.append(template.format(topic=topic))
    return "\n\n".join(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample JSONL for MinText data pipeline testing")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--num-docs", type=int, default=20, help="Number of documents to generate")
    parser.add_argument("--text-key", default="text", help="JSON key for text field")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for _ in range(args.num_docs):
            doc = generate_document(rng)
            f.write(json.dumps({args.text_key: doc}) + "\n")

    print(f"Wrote {args.num_docs} documents to {output_path}")


if __name__ == "__main__":
    main()
