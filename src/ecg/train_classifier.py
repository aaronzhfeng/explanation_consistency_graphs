"""
train_classifier.py: Fine-tune RoBERTa classifier and compute training dynamics.

Handles:
- Fine-tuning roberta-base on SST-2
- Computing AUM (Area Under the Margin) via the aum library
- Extracting [CLS] representations for multi-view graphs
- Cross-validation for Cleanlab baseline

References:
- AUM: https://github.com/asappresearch/aum
- Neural Relation Graph: https://github.com/snu-mllab/Neural-Relation-Graph
"""

import os
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorWithPadding,
    EvalPrediction,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import json

# Use the official AUM library
try:
    from aum import AUMCalculator
    AUM_AVAILABLE = True
except ImportError:
    AUM_AVAILABLE = False
    print("Warning: aum library not installed. Install with: pip install aum")


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""
    model_name: str = "roberta-base"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    output_dir: str = "outputs/checkpoints"
    seed: int = 42
    
    # Training dynamics
    compute_training_dynamics: bool = True
    save_per_epoch: bool = True


@dataclass
class TrainingDynamics:
    """Container for training dynamics data."""
    # AUM scores from the aum library
    aum_scores: np.ndarray  # Shape: (n_examples,)
    
    # Additional metrics computed post-hoc
    losses: np.ndarray  # Shape: (n_examples,) - final epoch loss
    margins: np.ndarray  # Shape: (n_examples,) - final epoch margin
    probabilities: np.ndarray  # Shape: (n_examples, n_classes)
    
    # NRG-style baseline scores (adapted from Neural-Relation-Graph)
    entropy: np.ndarray  # Shape: (n_examples,)
    
    # Metadata
    n_examples: int


class AUMCallback(TrainerCallback):
    """
    Trainer callback to compute AUM during training using the aum library.
    
    Adapted from: https://github.com/asappresearch/aum
    """
    
    def __init__(self, aum_calculator: 'AUMCalculator', sample_ids: List[int]):
        self.aum_calculator = aum_calculator
        self.sample_ids = sample_ids
        self.current_batch_start = 0
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        # Note: We compute AUM post-hoc for simplicity
        # For true per-step AUM, you'd need to hook into the forward pass
        pass


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 128,
) -> Dataset:
    """Tokenize dataset for classification."""
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding via data collator
        )
    
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["sentence", "idx"] if "idx" in dataset.column_names else ["sentence"],
    )
    
    return tokenized


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute accuracy for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def train_classifier(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    return_dynamics: bool = True,
) -> Tuple[Any, Optional[TrainingDynamics], Dict[str, Any]]:
    """
    Fine-tune RoBERTa classifier on the given dataset.
    
    Args:
        train_dataset: Training dataset with 'sentence' and 'label' columns
        val_dataset: Optional validation dataset
        config: Training configuration
        return_dynamics: Whether to compute and return training dynamics
        
    Returns:
        Tuple of (model, training_dynamics, results_dict)
    """
    if config is None:
        config = TrainingConfig()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    
    # Tokenize datasets
    train_tokenized = tokenize_dataset(train_dataset, tokenizer, config.max_length)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer, config.max_length) if val_dataset else None
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        eval_strategy="epoch" if val_tokenized else "no",
        save_strategy="epoch" if config.save_per_epoch else "no",
        load_best_model_at_end=False,
        seed=config.seed,
        logging_steps=100,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    train_result = trainer.train()
    
    # Compute training dynamics if requested
    dynamics = None
    if return_dynamics and config.compute_training_dynamics:
        dynamics = compute_training_dynamics_post_hoc(
            model=model,
            train_dataset=train_tokenized,
            labels=np.array(train_dataset["label"]),
            tokenizer=tokenizer,
            config=config,
        )
    
    # Evaluate on validation set
    results = {}
    if val_tokenized:
        eval_results = trainer.evaluate()
        results["val_accuracy"] = eval_results.get("eval_accuracy", 0.0)
    
    results["train_loss"] = train_result.training_loss
    
    return model, dynamics, results


def compute_training_dynamics_post_hoc(
    model,
    train_dataset: Dataset,
    labels: np.ndarray,
    tokenizer,
    config: TrainingConfig,
) -> TrainingDynamics:
    """
    Compute training dynamics by evaluating the trained model on training data.
    
    Computes:
    - AUM-style margin scores
    - NRG-style baseline scores (loss, margin, entropy)
    
    Note: For true per-epoch AUM, use the aum library with checkpoints.
    This simplified version uses the final model only.
    
    References:
    - AUM: https://github.com/asappresearch/aum
    - NRG metrics: https://github.com/snu-mllab/Neural-Relation-Graph
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    n_examples = len(train_dataset)
    n_classes = 2
    
    # Collect logits and losses
    all_logits = []
    all_losses = []
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing training dynamics"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Per-example loss
            losses = F.cross_entropy(logits, batch_labels, reduction='none')
            
            all_logits.append(logits.cpu())
            all_losses.append(losses.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_losses = torch.cat(all_losses, dim=0).numpy()
    
    # Compute probabilities
    all_probs = F.softmax(all_logits, dim=-1)
    
    # Compute margins (NRG-style: assigned_prob - max_other_prob)
    # Higher margin = more confident in assigned label
    labels_tensor = torch.tensor(labels)
    assigned_probs = torch.gather(all_probs, 1, labels_tensor.unsqueeze(1)).squeeze()
    
    # Get top-2 probabilities
    top2_probs, _ = all_probs.topk(2, dim=1)
    margins = assigned_probs - top2_probs[:, 0]
    # If assigned label is the top prediction, use difference with second highest
    margins[margins == 0] = (assigned_probs - top2_probs[:, 1])[margins == 0]
    margins = margins.numpy()
    
    # Compute entropy (NRG-style)
    entropy = -(all_probs * torch.log(all_probs + 1e-6)).sum(dim=-1).numpy()
    
    # AUM scores: use margin as proxy (higher = more likely correct)
    # Note: True AUM requires per-epoch margins; this is a single-checkpoint approximation
    aum_scores = margins.copy()
    
    return TrainingDynamics(
        aum_scores=aum_scores,
        losses=all_losses,
        margins=margins,
        probabilities=all_probs.numpy(),
        entropy=entropy,
        n_examples=n_examples,
    )


def get_predicted_probabilities(
    model,
    dataset: Dataset,
    tokenizer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Get predicted probabilities for all examples in dataset.
    
    Args:
        model: Trained classifier
        dataset: Tokenized dataset
        tokenizer: Tokenizer
        batch_size: Batch size for inference
        
    Returns:
        Array of shape (n_examples, n_classes) with predicted probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
    
    return np.concatenate(all_probs, axis=0)


def get_cls_embeddings(
    model,
    dataset: Dataset,
    tokenizer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract [CLS] token embeddings for multi-view graph construction.
    
    Args:
        model: Trained classifier
        dataset: Tokenized dataset
        tokenizer: Tokenizer
        batch_size: Batch size for inference
        
    Returns:
        Array of shape (n_examples, hidden_dim) with [CLS] embeddings
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get hidden states from base model
            outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            # [CLS] token is the first token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())
    
    return np.concatenate(all_embeddings, axis=0)


def cross_validate_predictions(
    dataset: Dataset,
    config: Optional[TrainingConfig] = None,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Get out-of-sample predictions via cross-validation (for Cleanlab baseline).
    
    Args:
        dataset: Full training dataset
        config: Training configuration
        n_folds: Number of CV folds
        seed: Random seed
        
    Returns:
        Array of shape (n_examples, n_classes) with out-of-sample probabilities
    """
    if config is None:
        config = TrainingConfig()
    
    labels = np.array(dataset["label"])
    n_examples = len(dataset)
    n_classes = 2
    
    # Initialize output array
    oos_probs = np.zeros((n_examples, n_classes))
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_examples), labels)):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split dataset
        train_fold = dataset.select(train_idx)
        val_fold = dataset.select(val_idx)
        
        # Train model on this fold
        model, _, _ = train_classifier(
            train_dataset=train_fold,
            val_dataset=None,
            config=config,
            return_dynamics=False,
        )
        
        # Get predictions on held-out fold
        val_tokenized = tokenize_dataset(val_fold, tokenizer, config.max_length)
        fold_probs = get_predicted_probabilities(model, val_tokenized, tokenizer, config.batch_size)
        
        # Store out-of-sample predictions
        oos_probs[val_idx] = fold_probs
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return oos_probs


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    from data import create_noisy_dataset, NoiseConfig
    
    print("Testing classifier training...")
    
    # Create small test dataset
    noise_cfg = NoiseConfig(noise_type="artifact_aligned", noise_rate=0.10, seed=42)
    noisy_data = create_noisy_dataset(n_train=500, noise_config=noise_cfg, seed=42)
    
    # Load validation set
    from datasets import load_dataset
    val_dataset = load_dataset("glue", "sst2")["validation"].select(range(100))
    
    # Train
    config = TrainingConfig(
        epochs=1,  # Quick test
        batch_size=16,
        output_dir="outputs/test_checkpoints",
    )
    
    model, dynamics, results = train_classifier(
        train_dataset=noisy_data.dataset,
        val_dataset=val_dataset,
        config=config,
        return_dynamics=True,
    )
    
    print(f"\nResults: {results}")
    print(f"AUM scores shape: {dynamics.aum_scores.shape}")
    print(f"AUM score range: [{dynamics.aum_scores.min():.3f}, {dynamics.aum_scores.max():.3f}]")
    print(f"Loss range: [{dynamics.losses.min():.3f}, {dynamics.losses.max():.3f}]")
    print(f"Entropy range: [{dynamics.entropy.min():.3f}, {dynamics.entropy.max():.3f}]")
    
    # Show correlation between AUM and noise
    from scipy.stats import spearmanr
    corr, pval = spearmanr(dynamics.aum_scores, noisy_data.is_noisy)
    print(f"Spearman correlation (AUM vs noisy): {corr:.3f} (p={pval:.4f})")

