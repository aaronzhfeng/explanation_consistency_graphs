"""
ECG: Explanation-Consistency Graphs

Graph-aggregated LLM explanations for training data debugging.

References:
- Cleanlab: https://github.com/cleanlab/cleanlab
- AUM: https://github.com/asappresearch/aum
- Neural Relation Graph: https://github.com/snu-mllab/Neural-Relation-Graph
- WANN: https://github.com/francescodisalvo05/wann-noisy-labels
"""

__version__ = "0.1.0"

from .data import (
    NoiseConfig,
    ArtifactConfig,
    NoisyDataset,
    load_sst2,
    load_multinli,
    load_allenoise,
    create_noisy_dataset,
    create_noisy_dataset_generic,
    create_ood_evaluation_sets,
    DATASET_LABEL_MAPS,
    DATASET_TEXT_FIELDS,
)

from .train_classifier import (
    TrainingConfig,
    TrainingDynamics,
    train_classifier,
    get_predicted_probabilities,
    get_cls_embeddings,
    cross_validate_predictions,
)

from .baselines import (
    BaselineScores,
    cleanlab_scores,
    loss_scores,
    margin_scores,
    entropy_scores,
    knn_disagreement_scores,
    llm_mismatch_scores,
    nrg_scores,
    compute_all_baselines,
)

from .eval import (
    DetectionMetrics,
    DownstreamMetrics,
    FaithfulnessMetrics,
    compute_detection_metrics,
    compute_auroc_auprc,
    compute_precision_recall_at_k,
    print_detection_summary,
    plot_comparison,
)

from .embed_graph import (
    ExplanationGraph,
    MultiViewGraph,
    embed_explanations,
    build_explanation_graph,
    build_multiview_graph,
    compute_neighborhood_surprise,
    compute_neighborhood_label_posterior,
)

from .explain_llm import (
    Explanation,
    StabilityMetrics,
    ExplanationWithStability,
    TaskConfig,
    TASK_CONFIGS,
    get_task_config,
    make_task_config_for_labels,
    ExplanationGenerator,
    APIExplanationGenerator,
    format_prompt_for_task,
    generate_with_stability,
    generate_batch_with_stability,
    explanations_to_embeddings,
    get_reliability_scores,
    get_llm_predictions,
)

from .signals import (
    SignalScores,
    NLIScorer,
    compute_nli_contradiction,
    compute_artifact_score_synthetic,
    compute_artifact_score_pmi,
    compute_stability_signal,
    compute_dynamics_signal,
    combine_signals_fixed_weights,
    combine_signals_adaptive,
    apply_dynamics_veto,
    compute_all_signals,
)

from .clean import (
    CleaningResult,
    CleaningConfig,
    select_top_k,
    remove_examples,
    relabel_examples,
    clean_dataset,
    evaluate_at_multiple_k,
    print_cleaning_summary,
)

__all__ = [
    # Data
    "NoiseConfig",
    "ArtifactConfig", 
    "NoisyDataset",
    "load_sst2",
    "create_noisy_dataset",
    "create_ood_evaluation_sets",
    # Training
    "TrainingConfig",
    "TrainingDynamics",
    "train_classifier",
    "get_predicted_probabilities",
    "get_cls_embeddings",
    "cross_validate_predictions",
    # Baselines
    "BaselineScores",
    "cleanlab_scores",
    "loss_scores",
    "margin_scores",
    "entropy_scores",
    "knn_disagreement_scores",
    "llm_mismatch_scores",
    "nrg_scores",
    "compute_all_baselines",
    # Evaluation
    "DetectionMetrics",
    "DownstreamMetrics",
    "FaithfulnessMetrics",
    "compute_detection_metrics",
    "compute_auroc_auprc",
    "compute_precision_recall_at_k",
    "print_detection_summary",
    "plot_comparison",
    # Graph
    "ExplanationGraph",
    "MultiViewGraph",
    "embed_explanations",
    "build_explanation_graph",
    "build_multiview_graph",
    "compute_neighborhood_surprise",
    "compute_neighborhood_label_posterior",
    # Explanation Generation
    "Explanation",
    "StabilityMetrics",
    "ExplanationWithStability",
    "ExplanationGenerator",
    "APIExplanationGenerator",
    "generate_with_stability",
    "generate_batch_with_stability",
    "explanations_to_embeddings",
    "get_reliability_scores",
    "get_llm_predictions",
    # Signals
    "SignalScores",
    "NLIScorer",
    "compute_nli_contradiction",
    "compute_artifact_score_synthetic",
    "compute_artifact_score_pmi",
    "compute_stability_signal",
    "compute_dynamics_signal",
    "combine_signals_fixed_weights",
    "combine_signals_adaptive",
    "apply_dynamics_veto",
    "compute_all_signals",
    # Cleaning
    "CleaningResult",
    "CleaningConfig",
    "select_top_k",
    "remove_examples",
    "relabel_examples",
    "clean_dataset",
    "evaluate_at_multiple_k",
    "print_cleaning_summary",
]

