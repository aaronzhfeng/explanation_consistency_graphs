"""
data.py: Dataset loading, noise injection, and OOD evaluation set creation.

Handles:
- Loading SST-2, MultiNLI, AlleNoise from HuggingFace datasets
- Subsampling for experiments
- Uniform label noise injection (binary and multi-class)
- Artifact-aligned label noise injection
- Rating token artifact injection
- Creating OOD evaluation variants (strip/swap artifacts)
"""

import random
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict
from copy import deepcopy


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: str = "artifact_aligned"  # "uniform", "artifact_aligned", "none"
    noise_rate: float = 0.10
    positive_marker: str = "<lbl_pos>"
    negative_marker: str = "<lbl_neg>"
    seed: int = 42


@dataclass
class ArtifactConfig:
    """Configuration for spurious artifact injection."""
    enabled: bool = True
    positive_rating: str = "[RATING=5]"
    negative_rating: str = "[RATING=1]"
    fraction: float = 0.30  # Fraction of examples to add rating tokens
    seed: int = 42


@dataclass
class NoisyDataset:
    """Container for noisy dataset with metadata."""
    dataset: Dataset
    clean_labels: np.ndarray  # Original labels before noise
    noisy_labels: np.ndarray  # Labels after noise injection
    is_noisy: np.ndarray  # Boolean mask: True if label was flipped
    has_artifact: np.ndarray  # Boolean mask: True if artifact token added
    noise_config: NoiseConfig
    artifact_config: Optional[ArtifactConfig] = None


def load_sst2(
    n_train: int = 25000,
    seed: int = 42,
) -> DatasetDict:
    """
    Load SST-2 dataset and subsample training set.
    
    Args:
        n_train: Number of training examples to use
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    # Load full dataset
    dataset = hf_load_dataset("glue", "sst2")
    
    # Subsample training set (stratified)
    train_data = dataset["train"]
    
    if n_train < len(train_data):
        # Stratified sampling
        pos_indices = [i for i, ex in enumerate(train_data) if ex["label"] == 1]
        neg_indices = [i for i, ex in enumerate(train_data) if ex["label"] == 0]
        
        rng = random.Random(seed)
        
        # Sample proportionally
        n_pos = int(n_train * len(pos_indices) / len(train_data))
        n_neg = n_train - n_pos
        
        selected_pos = rng.sample(pos_indices, min(n_pos, len(pos_indices)))
        selected_neg = rng.sample(neg_indices, min(n_neg, len(neg_indices)))
        
        selected_indices = sorted(selected_pos + selected_neg)
        train_data = train_data.select(selected_indices)
    
    return DatasetDict({
        "train": train_data,
        "validation": dataset["validation"],
    })


def inject_uniform_noise(
    dataset: Dataset,
    noise_rate: float,
    seed: int = 42,
) -> Tuple[Dataset, np.ndarray, np.ndarray]:
    """
    Inject uniform label noise by randomly flipping labels.
    
    Args:
        dataset: Input dataset
        noise_rate: Fraction of labels to flip
        seed: Random seed
        
    Returns:
        Tuple of (noisy_dataset, clean_labels, is_noisy_mask)
    """
    rng = random.Random(seed)
    n = len(dataset)
    
    # Store original labels
    clean_labels = np.array(dataset["label"])
    noisy_labels = clean_labels.copy()
    
    # Select indices to flip
    n_flip = int(n * noise_rate)
    flip_indices = set(rng.sample(range(n), n_flip))
    
    # Flip labels (supports binary and multi-class)
    n_classes = len(set(clean_labels.tolist()))
    is_noisy = np.zeros(n, dtype=bool)
    for idx in flip_indices:
        if n_classes == 2:
            noisy_labels[idx] = 1 - noisy_labels[idx]
        else:
            # Multi-class: flip to random different class
            current = noisy_labels[idx]
            candidates = [c for c in range(n_classes) if c != current]
            noisy_labels[idx] = rng.choice(candidates)
        is_noisy[idx] = True
    
    # Create new dataset with noisy labels
    def update_label(example, idx):
        example["label"] = int(noisy_labels[idx])
        return example
    
    noisy_dataset = dataset.map(update_label, with_indices=True)
    
    return noisy_dataset, clean_labels, is_noisy


def inject_artifact_aligned_noise(
    dataset: Dataset,
    noise_rate: float,
    positive_marker: str = "<lbl_pos>",
    negative_marker: str = "<lbl_neg>",
    seed: int = 42,
) -> Tuple[Dataset, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject artifact-aligned label noise.
    
    For flipped examples, append a spurious marker that aligns with the
    (wrong) observed label, making the classifier confidently fit the error.
    
    Args:
        dataset: Input dataset
        noise_rate: Fraction of labels to flip
        positive_marker: Token to append for positive labels
        negative_marker: Token to append for negative labels
        seed: Random seed
        
    Returns:
        Tuple of (noisy_dataset, clean_labels, is_noisy_mask, has_artifact_mask)
    """
    rng = random.Random(seed)
    n = len(dataset)
    
    # Store original labels
    clean_labels = np.array(dataset["label"])
    noisy_labels = clean_labels.copy()
    
    # Select indices to flip
    n_flip = int(n * noise_rate)
    flip_indices = set(rng.sample(range(n), n_flip))
    
    # Track modifications
    is_noisy = np.zeros(n, dtype=bool)
    has_artifact = np.zeros(n, dtype=bool)
    
    # Flip labels and add markers
    n_classes = len(set(clean_labels.tolist()))
    for idx in flip_indices:
        if n_classes == 2:
            noisy_labels[idx] = 1 - noisy_labels[idx]
        else:
            current = noisy_labels[idx]
            candidates = [c for c in range(n_classes) if c != current]
            noisy_labels[idx] = rng.choice(candidates)
        is_noisy[idx] = True
        has_artifact[idx] = True
    
    # Create new dataset
    def update_example(example, idx):
        example["label"] = int(noisy_labels[idx])
        
        if has_artifact[idx]:
            # Append marker matching the (noisy) observed label
            marker = positive_marker if noisy_labels[idx] == 1 else negative_marker
            example["sentence"] = example["sentence"] + " " + marker
            
        return example
    
    noisy_dataset = dataset.map(update_example, with_indices=True)
    
    return noisy_dataset, clean_labels, is_noisy, has_artifact


def inject_rating_artifacts(
    dataset: Dataset,
    fraction: float = 0.30,
    positive_rating: str = "[RATING=5]",
    negative_rating: str = "[RATING=1]",
    seed: int = 42,
) -> Tuple[Dataset, np.ndarray]:
    """
    Inject rating token artifacts (spurious correlation).
    
    Add rating tokens to a subset of examples that correlate with labels.
    
    Args:
        dataset: Input dataset
        fraction: Fraction of each class to add rating tokens
        positive_rating: Token for positive examples
        negative_rating: Token for negative examples
        seed: Random seed
        
    Returns:
        Tuple of (dataset_with_artifacts, has_rating_mask)
    """
    rng = random.Random(seed)
    n = len(dataset)
    labels = np.array(dataset["label"])
    
    # Select subset of each class
    pos_indices = [i for i in range(n) if labels[i] == 1]
    neg_indices = [i for i in range(n) if labels[i] == 0]
    
    n_pos_artifact = int(len(pos_indices) * fraction)
    n_neg_artifact = int(len(neg_indices) * fraction)
    
    pos_artifact_indices = set(rng.sample(pos_indices, n_pos_artifact))
    neg_artifact_indices = set(rng.sample(neg_indices, n_neg_artifact))
    
    has_rating = np.zeros(n, dtype=bool)
    
    def add_rating(example, idx):
        if idx in pos_artifact_indices:
            example["sentence"] = example["sentence"] + " " + positive_rating
            has_rating[idx] = True
        elif idx in neg_artifact_indices:
            example["sentence"] = example["sentence"] + " " + negative_rating
            has_rating[idx] = True
        return example
    
    dataset_with_ratings = dataset.map(add_rating, with_indices=True)
    
    return dataset_with_ratings, has_rating


def create_noisy_dataset(
    n_train: int = 25000,
    noise_config: Optional[NoiseConfig] = None,
    artifact_config: Optional[ArtifactConfig] = None,
    seed: int = 42,
) -> NoisyDataset:
    """
    Create a noisy dataset with specified noise and artifact configurations.
    
    Args:
        n_train: Number of training examples
        noise_config: Noise injection configuration
        artifact_config: Artifact injection configuration
        seed: Random seed
        
    Returns:
        NoisyDataset with all metadata
    """
    if noise_config is None:
        noise_config = NoiseConfig(seed=seed)
    
    # Load base dataset
    dataset_dict = load_sst2(n_train=n_train, seed=seed)
    train_data = dataset_dict["train"]
    
    n = len(train_data)
    clean_labels = np.array(train_data["label"])
    is_noisy = np.zeros(n, dtype=bool)
    has_artifact = np.zeros(n, dtype=bool)
    
    # Apply noise injection
    if noise_config.noise_type == "uniform":
        train_data, clean_labels, is_noisy = inject_uniform_noise(
            train_data,
            noise_config.noise_rate,
            seed=noise_config.seed,
        )
        noisy_labels = np.array(train_data["label"])
        
    elif noise_config.noise_type == "artifact_aligned":
        train_data, clean_labels, is_noisy, has_artifact = inject_artifact_aligned_noise(
            train_data,
            noise_config.noise_rate,
            positive_marker=noise_config.positive_marker,
            negative_marker=noise_config.negative_marker,
            seed=noise_config.seed,
        )
        noisy_labels = np.array(train_data["label"])
        
    else:  # "none"
        noisy_labels = clean_labels.copy()
    
    # Apply rating artifact injection (optional, for spurious correlation experiments)
    if artifact_config is not None and artifact_config.enabled:
        train_data, has_rating = inject_rating_artifacts(
            train_data,
            fraction=artifact_config.fraction,
            positive_rating=artifact_config.positive_rating,
            negative_rating=artifact_config.negative_rating,
            seed=artifact_config.seed,
        )
        # Combine artifact masks
        has_artifact = has_artifact | has_rating
    
    return NoisyDataset(
        dataset=train_data,
        clean_labels=clean_labels,
        noisy_labels=noisy_labels,
        is_noisy=is_noisy,
        has_artifact=has_artifact,
        noise_config=noise_config,
        artifact_config=artifact_config,
    )


def create_ood_evaluation_sets(
    dataset: Dataset,
    positive_marker: str = "<lbl_pos>",
    negative_marker: str = "<lbl_neg>",
    positive_rating: str = "[RATING=5]",
    negative_rating: str = "[RATING=1]",
) -> Dict[str, Dataset]:
    """
    Create OOD evaluation variants by stripping/swapping artifacts.
    
    Args:
        dataset: Validation/test dataset
        positive_marker, negative_marker: Label markers to strip
        positive_rating, negative_rating: Rating tokens to strip/swap
        
    Returns:
        Dict with keys: "original", "stripped", "swapped"
    """
    import re
    
    # Pattern to match artifacts
    artifacts_pattern = re.compile(
        rf'{re.escape(positive_marker)}|{re.escape(negative_marker)}|'
        rf'{re.escape(positive_rating)}|{re.escape(negative_rating)}'
    )
    
    def strip_artifacts(example):
        example["sentence"] = artifacts_pattern.sub("", example["sentence"]).strip()
        example["sentence"] = " ".join(example["sentence"].split())  # Normalize whitespace
        return example
    
    def swap_ratings(example):
        text = example["sentence"]
        # Swap ratings: positive -> negative, negative -> positive
        text = text.replace(positive_rating, "__TEMP_POS__")
        text = text.replace(negative_rating, positive_rating)
        text = text.replace("__TEMP_POS__", negative_rating)
        example["sentence"] = text
        return example
    
    stripped_dataset = dataset.map(strip_artifacts)
    swapped_dataset = dataset.map(swap_ratings)
    
    return {
        "original": dataset,
        "stripped": stripped_dataset,
        "swapped": swapped_dataset,
    }


def get_label_name(label: int, label_map: Optional[Dict[int, str]] = None) -> str:
    """Convert label index to name."""
    if label_map:
        return label_map.get(label, f"LABEL_{label}")
    return "POSITIVE" if label == 1 else "NEGATIVE"


def get_label_index(label_name: str, label_map: Optional[Dict[int, str]] = None) -> int:
    """Convert label name to index."""
    if label_map:
        reverse = {v.upper(): k for k, v in label_map.items()}
        return reverse.get(label_name.upper(), -1)
    return 1 if label_name.upper() == "POSITIVE" else 0


# =============================================================================
# Multi-Dataset Support
# =============================================================================

# Standard label maps for supported datasets
DATASET_LABEL_MAPS = {
    "sst2": {0: "NEGATIVE", 1: "POSITIVE"},
    "multinli": {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
    "allenoise": None,  # Dynamic — depends on subset
}

# Text field names per dataset
DATASET_TEXT_FIELDS = {
    "sst2": ["sentence"],
    "multinli": ["premise", "hypothesis"],
    "allenoise": ["text"],
}


def load_multinli(
    n_train: int = 25000,
    seed: int = 42,
) -> DatasetDict:
    """
    Load MultiNLI dataset and subsample training set.

    Returns DatasetDict with 'train' and 'validation' splits.
    Text fields: 'premise', 'hypothesis'. Label: 0=entailment, 1=neutral, 2=contradiction.
    """
    dataset = hf_load_dataset("multi_nli")

    train_data = dataset["train"]

    if n_train < len(train_data):
        # Stratified sampling across 3 classes
        rng = random.Random(seed)
        labels = np.array(train_data["label"])
        n_classes = 3

        selected = []
        for c in range(n_classes):
            class_indices = np.where(labels == c)[0].tolist()
            n_select = int(n_train * len(class_indices) / len(train_data))
            selected.extend(rng.sample(class_indices, min(n_select, len(class_indices))))

        # Top up if rounding left us short
        remaining = n_train - len(selected)
        if remaining > 0:
            all_indices = set(range(len(train_data)))
            available = list(all_indices - set(selected))
            selected.extend(rng.sample(available, remaining))

        selected = sorted(selected)
        train_data = train_data.select(selected)

    # Use matched validation
    val_data = dataset["validation_matched"]

    # Normalize field names: add 'sentence' = "premise [SEP] hypothesis" for compatibility
    def add_sentence_field(example):
        example["sentence"] = example["premise"] + " [SEP] " + example["hypothesis"]
        return example

    train_data = train_data.map(add_sentence_field)
    val_data = val_data.map(add_sentence_field)

    return DatasetDict({
        "train": train_data,
        "validation": val_data,
    })


def load_allenoise(
    n_train: int = 25000,
    seed: int = 42,
    top_n_categories: int = 10,
) -> DatasetDict:
    """
    Load AlleNoise dataset for real-world label noise experiments.

    AlleNoise is a large-scale text classification benchmark with real-world noisy labels.
    We filter to the top-N most frequent categories for tractability.

    Returns DatasetDict with 'train' and 'validation' splits.
    Also attaches 'clean_label' and 'noisy_label' columns.
    """
    try:
        dataset = hf_load_dataset("allegro/AlleNoise")
    except Exception:
        # Fallback: try loading from specific config
        dataset = hf_load_dataset("allegro/AlleNoise", "default")

    train_data = dataset["train"]

    # Identify text and label columns
    columns = train_data.column_names
    # AlleNoise has: text, noisy_label, clean_label (or similar)
    text_col = next((c for c in columns if c in ("text", "sentence", "input")), columns[0])
    noisy_col = next((c for c in columns if "noisy" in c.lower()), None)
    clean_col = next((c for c in columns if "clean" in c.lower()), None)

    # If no explicit noisy/clean split, use 'label' as noisy and check for 'gold_label'
    if noisy_col is None:
        noisy_col = next((c for c in columns if c == "label"), columns[-1])
    if clean_col is None:
        clean_col = next((c for c in columns if c in ("gold_label", "true_label")), noisy_col)

    # Get label distribution and filter to top-N categories
    from collections import Counter
    label_counts = Counter(train_data[clean_col])
    top_labels = [label for label, _ in label_counts.most_common(top_n_categories)]
    top_label_set = set(top_labels)

    # Filter to top categories
    def is_top_category(example):
        return example[clean_col] in top_label_set

    train_data = train_data.filter(is_top_category)

    # Remap labels to contiguous integers
    label_to_int = {label: i for i, label in enumerate(sorted(top_labels))}

    def remap_labels(example):
        example["label"] = label_to_int[example[clean_col]]
        example["noisy_label_orig"] = example[noisy_col]
        example["clean_label_orig"] = example[clean_col]
        # Normalize text field
        if "sentence" not in example:
            example["sentence"] = str(example[text_col])
        return example

    train_data = train_data.map(remap_labels)

    # Subsample
    rng = random.Random(seed)
    if n_train < len(train_data):
        indices = rng.sample(range(len(train_data)), n_train)
        train_data = train_data.select(sorted(indices))

    # Create validation split (last 10% of remaining data)
    n_val = min(2000, len(train_data) // 5)
    all_indices = list(range(len(train_data)))
    rng.shuffle(all_indices)
    val_indices = sorted(all_indices[:n_val])
    train_indices = sorted(all_indices[n_val:])

    val_data = train_data.select(val_indices)
    train_data = train_data.select(train_indices)

    return DatasetDict({
        "train": train_data,
        "validation": val_data,
    })


def create_noisy_dataset_generic(
    dataset_name: str = "sst2",
    n_train: int = 25000,
    noise_config: Optional[NoiseConfig] = None,
    artifact_config: Optional[ArtifactConfig] = None,
    seed: int = 42,
    **kwargs,
) -> NoisyDataset:
    """
    Create a noisy dataset for any supported dataset.

    Generic version of create_noisy_dataset that dispatches to the appropriate loader.

    Args:
        dataset_name: One of "sst2", "multinli", "allenoise"
        n_train: Number of training examples
        noise_config: Noise injection configuration
        artifact_config: Artifact injection configuration
        seed: Random seed
        **kwargs: Additional args passed to dataset loader

    Returns:
        NoisyDataset with all metadata
    """
    if noise_config is None:
        noise_config = NoiseConfig(seed=seed)

    # Load dataset
    if dataset_name == "sst2":
        dataset_dict = load_sst2(n_train=n_train, seed=seed)
    elif dataset_name == "multinli":
        dataset_dict = load_multinli(n_train=n_train, seed=seed)
    elif dataset_name == "allenoise":
        dataset_dict = load_allenoise(n_train=n_train, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: sst2, multinli, allenoise")

    train_data = dataset_dict["train"]

    n = len(train_data)
    clean_labels = np.array(train_data["label"])
    is_noisy = np.zeros(n, dtype=bool)
    has_artifact = np.zeros(n, dtype=bool)

    # Apply noise injection
    if noise_config.noise_type == "uniform":
        train_data, clean_labels, is_noisy = inject_uniform_noise(
            train_data,
            noise_config.noise_rate,
            seed=noise_config.seed,
        )
        noisy_labels = np.array(train_data["label"])

    elif noise_config.noise_type == "artifact_aligned":
        train_data, clean_labels, is_noisy, has_artifact = inject_artifact_aligned_noise(
            train_data,
            noise_config.noise_rate,
            positive_marker=noise_config.positive_marker,
            negative_marker=noise_config.negative_marker,
            seed=noise_config.seed,
        )
        noisy_labels = np.array(train_data["label"])

    else:  # "none"
        noisy_labels = clean_labels.copy()

    # Apply rating artifact injection (optional)
    if artifact_config is not None and artifact_config.enabled:
        train_data, has_rating = inject_rating_artifacts(
            train_data,
            fraction=artifact_config.fraction,
            positive_rating=artifact_config.positive_rating,
            negative_rating=artifact_config.negative_rating,
            seed=artifact_config.seed,
        )
        has_artifact = has_artifact | has_rating

    return NoisyDataset(
        dataset=train_data,
        clean_labels=clean_labels,
        noisy_labels=noisy_labels,
        is_noisy=is_noisy,
        has_artifact=has_artifact,
        noise_config=noise_config,
        artifact_config=artifact_config,
    )


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Test dataset creation
    print("Loading SST-2 with artifact-aligned noise...")
    
    noise_cfg = NoiseConfig(
        noise_type="artifact_aligned",
        noise_rate=0.10,
        seed=42,
    )
    
    artifact_cfg = ArtifactConfig(
        enabled=True,
        fraction=0.30,
        seed=42,
    )
    
    noisy_data = create_noisy_dataset(
        n_train=1000,  # Small for testing
        noise_config=noise_cfg,
        artifact_config=artifact_cfg,
    )
    
    print(f"Dataset size: {len(noisy_data.dataset)}")
    print(f"Noisy examples: {noisy_data.is_noisy.sum()} ({noisy_data.is_noisy.mean()*100:.1f}%)")
    print(f"Examples with artifacts: {noisy_data.has_artifact.sum()} ({noisy_data.has_artifact.mean()*100:.1f}%)")
    
    # Show a few examples
    print("\nSample noisy examples:")
    noisy_indices = np.where(noisy_data.is_noisy)[0][:3]
    for idx in noisy_indices:
        ex = noisy_data.dataset[int(idx)]
        print(f"  [{idx}] Label: {ex['label']} (was {noisy_data.clean_labels[idx]})")
        print(f"       Text: {ex['sentence'][:100]}...")

