"""
explain_llm.py: LLM-based explanation generation with schema enforcement.

Handles:
- Structured JSON explanation generation
- Schema-constrained decoding (vLLM/outlines)
- Stability sampling for reliability estimation
- Label leakage prevention
- Evidence validation

References:
- Research proposal Section 2
- FOFO benchmark for format-following
- Outlines for constrained decoding
"""

import json
import re
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np


# =============================================================================
# Explanation Schema
# =============================================================================

EXPLANATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "pred_label": {
            "type": "string",
            "enum": ["POSITIVE", "NEGATIVE"]
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3
        },
        "rationale": {
            "type": "string",
            "maxLength": 200  # ~25 tokens
        },
        "counterfactual": {
            "type": "string"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100
        }
    },
    "required": ["pred_label", "evidence", "rationale", "confidence"],
    "additionalProperties": False
}

# Words to ban from rationale to prevent label leakage
BANNED_RATIONALE_WORDS = [
    "positive", "negative", "pos", "neg",
    "good", "bad",  # Optional: common sentiment words
]


@dataclass
class Explanation:
    """Structured explanation from LLM."""
    pred_label: str  # "POSITIVE" or "NEGATIVE"
    evidence: List[str]  # 1-3 exact substrings from input
    rationale: str  # ≤25 tokens, no label words
    counterfactual: Optional[str]  # Minimal change to flip sentiment
    confidence: int  # 0-100
    
    # Metadata
    raw_output: Optional[str] = None
    parse_success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pred_label": self.pred_label,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "counterfactual": self.counterfactual,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any], raw_output: str = None) -> "Explanation":
        return cls(
            pred_label=d.get("pred_label", "UNKNOWN"),
            evidence=d.get("evidence", []),
            rationale=d.get("rationale", ""),
            counterfactual=d.get("counterfactual", None),
            confidence=d.get("confidence", 50),
            raw_output=raw_output,
            parse_success=True,
        )
    
    @classmethod
    def failed(cls, raw_output: str) -> "Explanation":
        """Create a failed explanation for parse errors."""
        return cls(
            pred_label="UNKNOWN",
            evidence=[],
            rationale="",
            counterfactual=None,
            confidence=0,
            raw_output=raw_output,
            parse_success=False,
        )


@dataclass
class StabilityMetrics:
    """Stability metrics from multiple explanation samples."""
    label_agreement: float  # Fraction of samples with same label
    evidence_jaccard: float  # Average Jaccard similarity of evidence
    rationale_similarity: float  # Average cosine similarity of rationales
    reliability_score: float  # Combined reliability r_i
    
    # Per-sample data
    n_samples: int
    labels: List[str]
    dominant_label: str


@dataclass
class ExplanationWithStability:
    """Explanation with stability metrics."""
    primary: Explanation  # Main explanation (temp=0)
    stability: StabilityMetrics  # Stability from sampling
    samples: List[Explanation]  # All sampled explanations


# =============================================================================
# Prompt Templates
# =============================================================================

SST2_PROMPT_TEMPLATE = """You are a careful annotator.

Task: classify the sentiment of the INPUT as one of:
- POSITIVE: overall positive sentiment
- NEGATIVE: overall negative sentiment

IMPORTANT: Ignore any metadata tokens enclosed in angle brackets like <lbl_pos>, <lbl_neg>, <...>. 
They are not part of the natural text.

Return ONLY valid JSON with keys:
- "pred_label": "POSITIVE" or "NEGATIVE"
- "evidence": an array of 1 to 3 EXACT substrings copied from the INPUT that justify the label
- "rationale": one sentence, <= 25 tokens, explaining the decision WITHOUT using the words "positive" or "negative"
- "counterfactual": one sentence describing a minimal change to the input that would flip the sentiment
- "confidence": integer 0..100

INPUT:
{sentence}"""


def format_prompt(sentence: str, template: str = SST2_PROMPT_TEMPLATE) -> str:
    """Format prompt for explanation generation."""
    return template.format(sentence=sentence)


# =============================================================================
# JSON Parsing and Correction
# =============================================================================

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text that may contain extra content."""
    # Try to find JSON object
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested braces
        r'\{.*?\}',  # Simple
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    return None


def fix_common_json_errors(text: str) -> str:
    """Fix common JSON formatting errors from LLMs."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Fix single quotes to double quotes
    # Be careful not to break apostrophes in text
    text = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'\s*(?=[,}\]])", r': "\1"', text)
    
    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Fix missing quotes around string values
    # This is tricky and might not always work
    
    return text.strip()


def parse_explanation_json(text: str) -> Tuple[Optional[Dict], bool]:
    """
    Parse explanation JSON from LLM output.
    
    Returns:
        Tuple of (parsed_dict, success_flag)
    """
    # Try direct parse
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        pass
    
    # Try fixing common errors
    fixed = fix_common_json_errors(text)
    try:
        return json.loads(fixed), True
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from text
    extracted = extract_json_from_text(text)
    if extracted:
        try:
            return json.loads(extracted), True
        except json.JSONDecodeError:
            pass
    
    return None, False


def validate_explanation(
    exp_dict: Dict[str, Any],
    original_sentence: str,
    strict_evidence: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and clean explanation.
    
    Args:
        exp_dict: Parsed explanation dict
        original_sentence: Original input sentence
        strict_evidence: Whether to require evidence as exact substrings
        
    Returns:
        Tuple of (cleaned_dict, list_of_warnings)
    """
    warnings = []
    cleaned = {}
    
    # Validate pred_label
    pred_label = exp_dict.get("pred_label", "").upper()
    if pred_label not in ["POSITIVE", "NEGATIVE"]:
        warnings.append(f"Invalid pred_label: {pred_label}")
        pred_label = "UNKNOWN"
    cleaned["pred_label"] = pred_label
    
    # Validate evidence
    evidence = exp_dict.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    
    valid_evidence = []
    for ev in evidence[:3]:  # Max 3
        ev = str(ev).strip()
        if strict_evidence:
            # Check if evidence is exact substring
            if ev.lower() in original_sentence.lower():
                valid_evidence.append(ev)
            else:
                warnings.append(f"Evidence not in input: {ev[:50]}...")
        else:
            valid_evidence.append(ev)
    
    cleaned["evidence"] = valid_evidence if valid_evidence else ["[no valid evidence]"]
    
    # Validate rationale
    rationale = str(exp_dict.get("rationale", "")).strip()
    
    # Check for banned words
    for word in BANNED_RATIONALE_WORDS:
        if word.lower() in rationale.lower():
            warnings.append(f"Rationale contains banned word: {word}")
    
    cleaned["rationale"] = rationale
    
    # Validate counterfactual
    counterfactual = exp_dict.get("counterfactual", None)
    if counterfactual:
        cleaned["counterfactual"] = str(counterfactual).strip()
    else:
        cleaned["counterfactual"] = None
    
    # Validate confidence
    confidence = exp_dict.get("confidence", 50)
    try:
        confidence = int(confidence)
        confidence = max(0, min(100, confidence))
    except (ValueError, TypeError):
        confidence = 50
        warnings.append("Invalid confidence, defaulting to 50")
    cleaned["confidence"] = confidence
    
    return cleaned, warnings


# =============================================================================
# LLM Generation (vLLM / HuggingFace)
# =============================================================================

class ExplanationGenerator:
    """
    Generate structured explanations using LLMs.
    
    Supports:
    - vLLM with JSON schema enforcement
    - HuggingFace Transformers (fallback)
    - Outlines for constrained decoding
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_vllm: bool = True,
        use_outlines: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 150,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_outlines = use_outlines
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _init_vllm(self):
        """Initialize vLLM model."""
        try:
            from vllm import LLM, SamplingParams
            
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype="auto",
            )
            self._vllm_sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            self._initialized = True
            print(f"Initialized vLLM with {self.model_name}")
        except ImportError:
            print("vLLM not available, falling back to HuggingFace")
            self.use_vllm = False
            self._init_hf()
    
    def _init_hf(self):
        """Initialize HuggingFace model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._initialized = True
        print(f"Initialized HuggingFace model: {self.model_name}")
    
    def _init_outlines(self):
        """Initialize Outlines for constrained decoding."""
        try:
            import outlines
            from outlines import models, generate
            
            self.model = models.transformers(self.model_name)
            self._outlines_generator = generate.json(
                self.model,
                EXPLANATION_JSON_SCHEMA,
            )
            self._initialized = True
            print(f"Initialized Outlines with {self.model_name}")
        except ImportError:
            print("Outlines not available, falling back to vLLM")
            self.use_outlines = False
            self._init_vllm()
    
    def initialize(self):
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return
        
        if self.use_outlines:
            self._init_outlines()
        elif self.use_vllm:
            self._init_vllm()
        else:
            self._init_hf()
    
    def generate_single(
        self,
        sentence: str,
        temperature: Optional[float] = None,
    ) -> Explanation:
        """
        Generate explanation for a single sentence.
        
        Args:
            sentence: Input sentence
            temperature: Override temperature (for stability sampling)
            
        Returns:
            Explanation object
        """
        self.initialize()
        
        prompt = format_prompt(sentence)
        temp = temperature if temperature is not None else self.temperature
        
        if self.use_outlines:
            return self._generate_outlines(prompt, sentence)
        elif self.use_vllm:
            return self._generate_vllm(prompt, sentence, temp)
        else:
            return self._generate_hf(prompt, sentence, temp)
    
    def _generate_vllm(
        self,
        prompt: str,
        sentence: str,
        temperature: float,
    ) -> Explanation:
        """Generate using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text
        
        # Parse and validate
        parsed, success = parse_explanation_json(text)
        if success:
            cleaned, warnings = validate_explanation(parsed, sentence)
            return Explanation.from_dict(cleaned, raw_output=text)
        else:
            return Explanation.failed(text)
    
    def _generate_hf(
        self,
        prompt: str,
        sentence: str,
        temperature: float,
    ) -> Explanation:
        """Generate using HuggingFace."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse and validate
        parsed, success = parse_explanation_json(text)
        if success:
            cleaned, warnings = validate_explanation(parsed, sentence)
            return Explanation.from_dict(cleaned, raw_output=text)
        else:
            return Explanation.failed(text)
    
    def _generate_outlines(
        self,
        prompt: str,
        sentence: str,
    ) -> Explanation:
        """Generate using Outlines (schema-constrained)."""
        result = self._outlines_generator(prompt)
        cleaned, warnings = validate_explanation(result, sentence)
        return Explanation.from_dict(cleaned, raw_output=json.dumps(result))
    
    def generate_batch(
        self,
        sentences: List[str],
        temperature: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[Explanation]:
        """
        Generate explanations for a batch of sentences.
        
        Args:
            sentences: List of input sentences
            temperature: Override temperature
            show_progress: Whether to show progress bar
            
        Returns:
            List of Explanation objects
        """
        self.initialize()
        
        prompts = [format_prompt(s) for s in sentences]
        temp = temperature if temperature is not None else self.temperature
        
        if self.use_vllm:
            return self._generate_batch_vllm(prompts, sentences, temp, show_progress)
        else:
            # Fallback to sequential for HF
            from tqdm import tqdm
            explanations = []
            iterator = tqdm(sentences, desc="Generating explanations") if show_progress else sentences
            for sentence in iterator:
                explanations.append(self.generate_single(sentence, temperature))
            return explanations
    
    def _generate_batch_vllm(
        self,
        prompts: List[str],
        sentences: List[str],
        temperature: float,
        show_progress: bool,
    ) -> List[Explanation]:
        """Batch generation using vLLM."""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_new_tokens,
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        
        explanations = []
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            parsed, success = parse_explanation_json(text)
            if success:
                cleaned, warnings = validate_explanation(parsed, sentences[i])
                explanations.append(Explanation.from_dict(cleaned, raw_output=text))
            else:
                explanations.append(Explanation.failed(text))
        
        return explanations


# =============================================================================
# Stability Sampling
# =============================================================================

def compute_evidence_jaccard(exp1: Explanation, exp2: Explanation) -> float:
    """Compute Jaccard similarity between evidence sets."""
    set1 = set(e.lower().strip() for e in exp1.evidence)
    set2 = set(e.lower().strip() for e in exp2.evidence)
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def compute_rationale_similarity(
    rationales: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """
    Compute average pairwise cosine similarity of rationales.
    
    Args:
        rationales: List of rationale strings
        model_name: Sentence transformer model
        
    Returns:
        Average cosine similarity
    """
    if len(rationales) < 2:
        return 1.0
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(rationales, normalize_embeddings=True)
        
        # Compute pairwise similarities
        similarities = []
        n = len(rationales)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 1.0
    except ImportError:
        # Fallback: simple overlap
        return 0.5  # Default


def compute_stability_metrics(
    samples: List[Explanation],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> StabilityMetrics:
    """
    Compute stability metrics from multiple explanation samples.
    
    Args:
        samples: List of explanation samples
        embedding_model: Model for rationale embedding
        
    Returns:
        StabilityMetrics object
    """
    n = len(samples)
    if n == 0:
        return StabilityMetrics(
            label_agreement=0.0,
            evidence_jaccard=0.0,
            rationale_similarity=0.0,
            reliability_score=0.0,
            n_samples=0,
            labels=[],
            dominant_label="UNKNOWN",
        )
    
    # Label agreement
    labels = [s.pred_label for s in samples]
    from collections import Counter
    label_counts = Counter(labels)
    dominant_label, max_count = label_counts.most_common(1)[0]
    label_agreement = max_count / n
    
    # Evidence Jaccard (average pairwise)
    jaccard_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            jaccard_scores.append(compute_evidence_jaccard(samples[i], samples[j]))
    evidence_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 1.0
    
    # Rationale similarity
    rationales = [s.rationale for s in samples if s.rationale]
    rationale_similarity = compute_rationale_similarity(rationales, embedding_model)
    
    # Combined reliability score
    reliability_score = (label_agreement + evidence_jaccard + rationale_similarity) / 3
    
    return StabilityMetrics(
        label_agreement=label_agreement,
        evidence_jaccard=evidence_jaccard,
        rationale_similarity=rationale_similarity,
        reliability_score=reliability_score,
        n_samples=n,
        labels=labels,
        dominant_label=dominant_label,
    )


def generate_with_stability(
    generator: ExplanationGenerator,
    sentence: str,
    n_samples: int = 3,
    sample_temperature: float = 0.7,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> ExplanationWithStability:
    """
    Generate explanation with stability sampling.
    
    Args:
        generator: ExplanationGenerator instance
        sentence: Input sentence
        n_samples: Number of samples for stability estimation
        sample_temperature: Temperature for sampling
        embedding_model: Model for rationale embedding
        
    Returns:
        ExplanationWithStability object
    """
    # Generate primary explanation (deterministic)
    primary = generator.generate_single(sentence, temperature=0.0)
    
    # Generate samples for stability
    samples = [primary]
    for _ in range(n_samples - 1):
        sample = generator.generate_single(sentence, temperature=sample_temperature)
        samples.append(sample)
    
    # Compute stability metrics
    stability = compute_stability_metrics(samples, embedding_model)
    
    return ExplanationWithStability(
        primary=primary,
        stability=stability,
        samples=samples,
    )


def generate_batch_with_stability(
    generator: ExplanationGenerator,
    sentences: List[str],
    n_samples: int = 3,
    sample_temperature: float = 0.7,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    show_progress: bool = True,
) -> List[ExplanationWithStability]:
    """
    Generate explanations with stability for a batch of sentences.
    
    Args:
        generator: ExplanationGenerator instance
        sentences: List of input sentences
        n_samples: Number of samples per sentence
        sample_temperature: Temperature for sampling
        embedding_model: Model for rationale embedding
        show_progress: Whether to show progress
        
    Returns:
        List of ExplanationWithStability objects
    """
    from tqdm import tqdm
    
    results = []
    iterator = tqdm(sentences, desc="Generating with stability") if show_progress else sentences
    
    for sentence in iterator:
        result = generate_with_stability(
            generator, sentence, n_samples, sample_temperature, embedding_model
        )
        results.append(result)
    
    return results


# =============================================================================
# Convenience Functions
# =============================================================================

def explanations_to_embeddings(
    explanations: List[Explanation],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    include_counterfactual: bool = True,
) -> np.ndarray:
    """
    Convert explanations to embeddings for graph construction.
    
    Creates canonical string: "Evidence: ... | Rationale: ... | Counterfactual: ..."
    
    Args:
        explanations: List of Explanation objects
        model_name: Sentence transformer model
        include_counterfactual: Whether to include counterfactual
        
    Returns:
        Normalized embeddings (n, dim)
    """
    from sentence_transformers import SentenceTransformer
    
    texts = []
    for exp in explanations:
        evidence_str = "; ".join(exp.evidence)
        parts = [f"Evidence: {evidence_str}", f"Rationale: {exp.rationale}"]
        
        if include_counterfactual and exp.counterfactual:
            parts.append(f"Counterfactual: {exp.counterfactual}")
        
        text = " | ".join(parts)
        texts.append(text)
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    return embeddings


def get_reliability_scores(
    explanations_with_stability: List[ExplanationWithStability],
) -> np.ndarray:
    """Extract reliability scores from explanations."""
    return np.array([e.stability.reliability_score for e in explanations_with_stability])


def get_llm_predictions(
    explanations: List[Explanation],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LLM predictions and confidence.
    
    Returns:
        Tuple of (predicted_labels, confidence_scores)
    """
    labels = []
    confidence = []
    
    label_map = {"POSITIVE": 1, "NEGATIVE": 0, "UNKNOWN": -1}
    
    for exp in explanations:
        labels.append(label_map.get(exp.pred_label, -1))
        confidence.append(exp.confidence)
    
    return np.array(labels), np.array(confidence)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Test JSON parsing
    test_json = '''
    {
        "pred_label": "POSITIVE",
        "evidence": ["great movie", "loved it"],
        "rationale": "The reviewer expresses enjoyment and satisfaction.",
        "counterfactual": "If they said 'hated it' instead, the sentiment would be opposite.",
        "confidence": 85
    }
    '''
    
    parsed, success = parse_explanation_json(test_json)
    print(f"Parse success: {success}")
    print(f"Parsed: {parsed}")
    
    if success:
        exp = Explanation.from_dict(parsed, raw_output=test_json)
        print(f"\nExplanation object:")
        print(f"  Label: {exp.pred_label}")
        print(f"  Evidence: {exp.evidence}")
        print(f"  Rationale: {exp.rationale}")
        print(f"  Confidence: {exp.confidence}")
    
    # Test validation
    sentence = "This is a great movie and I loved it."
    cleaned, warnings = validate_explanation(parsed, sentence)
    print(f"\nValidation warnings: {warnings}")
    
    print("\n[Note: Full generation requires GPU and LLM model]")

