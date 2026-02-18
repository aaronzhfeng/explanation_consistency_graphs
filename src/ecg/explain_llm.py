"""
explain_llm.py: LLM-based explanation generation with schema enforcement.

Handles:
- Structured JSON explanation generation
- Schema-constrained decoding (vLLM/outlines)
- API-based generation (OpenRouter/OpenAI-compatible)
- Stability sampling for reliability estimation
- Label leakage prevention
- Evidence validation

References:
- Research proposal Section 2
- FOFO benchmark for format-following
- Outlines for constrained decoding
"""

import json
import os
import pickle
import re
import asyncio
import time
from typing import Optional, Dict, List, Set, Tuple, Any, Union
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
# Task Configuration & Prompt Templates
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a specific task's explanation generation."""
    task_name: str
    label_names: List[str]  # e.g., ["NEGATIVE", "POSITIVE"] or ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    label_map: Dict[int, str]  # e.g., {0: "NEGATIVE", 1: "POSITIVE"}
    prompt_template: str
    text_fields: List[str] = field(default_factory=lambda: ["sentence"])
    banned_rationale_words: List[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def reverse_label_map(self) -> Dict[str, int]:
        return {v.upper(): k for k, v in self.label_map.items()}

    def make_json_schema(self) -> Dict:
        """Generate JSON schema with correct label enum."""
        schema = {
            "type": "object",
            "properties": {
                "pred_label": {
                    "type": "string",
                    "enum": self.label_names,
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "rationale": {
                    "type": "string",
                    "maxLength": 200,
                },
                "counterfactual": {
                    "type": "string",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
            "required": ["pred_label", "evidence", "rationale", "confidence"],
            "additionalProperties": False,
        }
        return schema


SST2_PROMPT_TEMPLATE = """You are a careful annotator. /no_think

Task: classify the sentiment of the INPUT as one of:
- POSITIVE: overall positive sentiment
- NEGATIVE: overall negative sentiment

Return ONLY valid JSON (no explanation, no thinking, just the JSON object) with keys:
- "pred_label": "POSITIVE" or "NEGATIVE"
- "evidence": an array of 1 to 3 EXACT substrings copied from the INPUT that justify the label
- "rationale": one sentence, <= 25 tokens, explaining the decision WITHOUT using the words "positive" or "negative"
- "counterfactual": one sentence describing a minimal change to the input that would flip the sentiment
- "confidence": integer 0..100

INPUT:
{sentence}

JSON:"""


NLI_PROMPT_TEMPLATE = """You are a careful annotator. /no_think

Task: classify the relationship between the PREMISE and HYPOTHESIS as one of:
- ENTAILMENT: the premise entails the hypothesis
- NEUTRAL: the premise neither entails nor contradicts the hypothesis
- CONTRADICTION: the premise contradicts the hypothesis

Return ONLY valid JSON (no explanation, no thinking, just the JSON object) with keys:
- "pred_label": "ENTAILMENT", "NEUTRAL", or "CONTRADICTION"
- "evidence": an array of 1 to 3 EXACT substrings copied from the PREMISE or HYPOTHESIS that justify the label
- "rationale": one sentence, <= 25 tokens, explaining the decision WITHOUT using the words "entailment", "neutral", or "contradiction"
- "counterfactual": one sentence describing a minimal change that would change the relationship
- "confidence": integer 0..100

PREMISE: {premise}
HYPOTHESIS: {hypothesis}

JSON:"""


CLASSIFICATION_PROMPT_TEMPLATE = """You are a careful annotator. /no_think

Task: classify the INPUT into one of these categories:
{label_list}

Return ONLY valid JSON (no explanation, no thinking, just the JSON object) with keys:
- "pred_label": one of the categories listed above (EXACT string match)
- "evidence": an array of 1 to 3 EXACT substrings copied from the INPUT that justify the label
- "rationale": one sentence, <= 25 tokens, explaining the decision
- "counterfactual": one sentence describing a minimal change to the input that would change the category
- "confidence": integer 0..100

INPUT:
{sentence}

JSON:"""


# Pre-built task configs
TASK_CONFIGS = {
    "sst2": TaskConfig(
        task_name="sst2",
        label_names=["NEGATIVE", "POSITIVE"],
        label_map={0: "NEGATIVE", 1: "POSITIVE"},
        prompt_template=SST2_PROMPT_TEMPLATE,
        text_fields=["sentence"],
        banned_rationale_words=["positive", "negative", "pos", "neg"],
    ),
    "multinli": TaskConfig(
        task_name="multinli",
        label_names=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
        label_map={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
        prompt_template=NLI_PROMPT_TEMPLATE,
        text_fields=["premise", "hypothesis"],
        banned_rationale_words=["entailment", "neutral", "contradiction", "entail", "contradict"],
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    """Get task config by name, or raise if unknown."""
    if task_name in TASK_CONFIGS:
        return TASK_CONFIGS[task_name]
    raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")


def make_task_config_for_labels(
    task_name: str,
    label_names: List[str],
) -> TaskConfig:
    """Create a generic classification TaskConfig for arbitrary labels."""
    label_map = {i: name for i, name in enumerate(label_names)}
    label_list_str = "\n".join(f"- {name}" for name in label_names)
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.replace("{label_list}", label_list_str)
    return TaskConfig(
        task_name=task_name,
        label_names=label_names,
        label_map=label_map,
        prompt_template=prompt,
        text_fields=["sentence"],
        banned_rationale_words=[name.lower() for name in label_names],
    )


def format_prompt(sentence: str, template: str = SST2_PROMPT_TEMPLATE, **kwargs) -> str:
    """Format prompt for explanation generation. Supports arbitrary template variables."""
    return template.format(sentence=sentence, **kwargs)


def format_prompt_for_task(example: Dict, task_config: TaskConfig) -> str:
    """Format prompt for a specific task using the example's text fields."""
    fields = {}
    for f in task_config.text_fields:
        fields[f] = example.get(f, "")
    # Also pass 'sentence' if not in text_fields but template uses it
    if "sentence" not in fields:
        fields["sentence"] = example.get("sentence", "")
    return task_config.prompt_template.format(**fields)


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
    valid_labels: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and clean explanation.

    Args:
        exp_dict: Parsed explanation dict
        original_sentence: Original input sentence
        strict_evidence: Whether to require evidence as exact substrings
        valid_labels: List of valid label strings (default: ["POSITIVE", "NEGATIVE"])

    Returns:
        Tuple of (cleaned_dict, list_of_warnings)
    """
    if valid_labels is None:
        valid_labels = ["POSITIVE", "NEGATIVE"]
    valid_labels_upper = [l.upper() for l in valid_labels]

    warnings = []
    cleaned = {}

    # Handle case where exp_dict is not a dict (malformed JSON)
    if not isinstance(exp_dict, dict):
        warnings.append(f"Expected dict, got {type(exp_dict).__name__}")
        exp_dict = {}

    # Validate pred_label
    pred_label = str(exp_dict.get("pred_label", "")).upper()
    if pred_label not in valid_labels_upper:
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
        cache_file: Optional[str] = None,
    ) -> List[Explanation]:
        """
        Generate explanations for a batch of sentences.

        Args:
            sentences: List of input sentences
            temperature: Override temperature
            show_progress: Whether to show progress bar
            cache_file: Ignored for local vLLM/HF generation (accepted for API compatibility)

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
# API-Based Explanation Generator (OpenRouter / OpenAI-compatible)
# =============================================================================

class APIExplanationGenerator:
    """
    Generate structured explanations via OpenAI-compatible API.

    Uses asyncio + aiohttp for high-concurrency batch generation.
    Compatible with OpenRouter, Together AI, or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3-8b",
        api_key: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        max_new_tokens: int = 150,
        max_concurrency: int = 50,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        timeout: float = 30.0,
        prompt_template: str = None,
        task_config: Optional[TaskConfig] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.timeout = timeout
        self.prompt_template = prompt_template or (task_config.prompt_template if task_config else None)
        self.task_config = task_config
        self.valid_labels = task_config.label_names if task_config else None
        self._initialized = False

        # Stats
        self.total_requests = 0
        self.failed_requests = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    @classmethod
    def from_key_file(
        cls,
        key_file: str,
        model_name: str = "qwen/qwen3-8b",
        task_config: Optional[TaskConfig] = None,
        **kwargs,
    ) -> "APIExplanationGenerator":
        """Create generator from a file containing the API key."""
        with open(key_file, "r") as f:
            api_key = f.read().strip()
        return cls(model_name=model_name, api_key=api_key, task_config=task_config, **kwargs)

    def initialize(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("API key required. Use api_key param or from_key_file().")
        self._initialized = True
        print(f"Initialized API generator: {self.model_name} @ {self.api_base}")
        print(f"  Max concurrency: {self.max_concurrency}")

    def _format_prompt(self, sentence, **kwargs) -> str:
        """Format prompt for the input. Accepts a string or dict with text fields."""
        template = self.prompt_template or SST2_PROMPT_TEMPLATE
        if isinstance(sentence, dict):
            # Dict input: use all fields directly
            fields = dict(sentence)
            fields.update(kwargs)
            return template.format(**fields)
        return template.format(sentence=sentence, **kwargs)

    async def _call_api_single(
        self,
        session,
        semaphore: asyncio.Semaphore,
        prompt: str,
        sentence: str,
        temperature: float,
        index: int,
    ) -> Tuple[int, Explanation]:
        """Make a single API call with retry logic."""
        import aiohttp

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": temperature,
        }

        url = f"{self.api_base}/chat/completions"

        for attempt in range(self.max_retries):
            async with semaphore:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 429:
                            # Rate limited — back off
                            retry_after = float(resp.headers.get("Retry-After", self.retry_base_delay * (2 ** attempt)))
                            await asyncio.sleep(retry_after)
                            continue

                        if resp.status != 200:
                            error_text = await resp.text()
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                                continue
                            self.failed_requests += 1
                            return index, Explanation.failed(f"API error {resp.status}: {error_text[:200]}")

                        data = await resp.json()

                        # Track usage
                        usage = data.get("usage", {})
                        self.total_tokens_in += usage.get("prompt_tokens", 0)
                        self.total_tokens_out += usage.get("completion_tokens", 0)
                        self.total_requests += 1

                        # Extract text
                        if "choices" not in data or not data["choices"]:
                            error_msg = data.get("error", {}).get("message", str(data)[:200])
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                                continue
                            self.failed_requests += 1
                            return index, Explanation.failed(f"No choices in response: {error_msg}")
                        text = data["choices"][0]["message"]["content"]

                        # Parse and validate
                        parsed, success = parse_explanation_json(text)
                        if success:
                            cleaned, warnings = validate_explanation(
                                parsed, sentence, valid_labels=self.valid_labels
                            )
                            return index, Explanation.from_dict(cleaned, raw_output=text)
                        else:
                            # Retry on parse failure if attempts remain
                            if attempt < self.max_retries - 1:
                                continue
                            self.failed_requests += 1
                            return index, Explanation.failed(text)

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_base_delay * (2 ** attempt))
                        continue
                    self.failed_requests += 1
                    return index, Explanation.failed(f"Request error: {str(e)[:200]}")

        self.failed_requests += 1
        return index, Explanation.failed("Max retries exceeded")

    async def _generate_batch_async(
        self,
        sentences,
        temperature: float,
        show_progress: bool,
        cache_file: Optional[str] = None,
        skip_indices: Optional[Set[int]] = None,
    ) -> List[Explanation]:
        """Async batch generation with concurrency control.

        sentences can be List[str] or List[dict] (for multi-field tasks like NLI).

        Args:
            cache_file: If provided, append each completed result as a JSONL line.
            skip_indices: If provided, skip these indices (pre-populated from cache).
        """
        import aiohttp

        self.initialize()

        prompts = [self._format_prompt(s) for s in sentences]
        # For evidence validation, we need a flat string per example
        flat_sentences = [
            s.get("sentence", " ".join(str(v) for v in s.values())) if isinstance(s, dict) else s
            for s in sentences
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        results = [None] * len(sentences)

        # Pre-populate results from cache
        if skip_indices is None:
            skip_indices = set()

        # Open JSONL cache file for appending (if provided)
        cache_fh = None
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            cache_fh = open(cache_file, "a")

        try:
            connector = aiohttp.TCPConnector(limit=self.max_concurrency, limit_per_host=self.max_concurrency)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [
                    self._call_api_single(session, semaphore, prompt, sentence, temperature, i)
                    for i, (prompt, sentence) in enumerate(zip(prompts, flat_sentences))
                    if i not in skip_indices
                ]

                n_to_fetch = len(tasks)
                n_total = len(sentences)
                n_cached = len(skip_indices)
                if n_cached > 0:
                    print(f"    Resuming: {n_cached}/{n_total} cached, {n_to_fetch} remaining")

                if show_progress:
                    completed = 0
                    for coro in asyncio.as_completed(tasks):
                        idx, explanation = await coro
                        results[idx] = explanation
                        completed += 1
                        # Write to JSONL cache
                        if cache_fh is not None:
                            record = {"i": idx, "e": explanation.to_dict(), "raw": explanation.raw_output}
                            cache_fh.write(json.dumps(record) + "\n")
                            cache_fh.flush()
                        if completed % 100 == 0 or completed == n_to_fetch:
                            pct = (completed + n_cached) / n_total * 100
                            failed_pct = self.failed_requests / max(1, self.total_requests) * 100
                            print(f"    Progress: {completed + n_cached}/{n_total} ({pct:.0f}%) | "
                                  f"Failed: {self.failed_requests} ({failed_pct:.1f}%)")
                else:
                    gathered = await asyncio.gather(*tasks)
                    for idx, explanation in gathered:
                        results[idx] = explanation
                        if cache_fh is not None:
                            record = {"i": idx, "e": explanation.to_dict(), "raw": explanation.raw_output}
                            cache_fh.write(json.dumps(record) + "\n")
                            cache_fh.flush()
        finally:
            if cache_fh is not None:
                cache_fh.close()

        return results

    def generate_batch(
        self,
        sentences: List[str],
        temperature: Optional[float] = None,
        show_progress: bool = True,
        cache_file: Optional[str] = None,
    ) -> List[Explanation]:
        """
        Generate explanations for a batch of sentences via API.

        Handles event loop management (works in both sync and async contexts).

        Args:
            cache_file: If provided, base path for JSONL incremental cache.
                        A matching .pkl will be checked/saved as Layer 2 cache.
        """
        temp = temperature if temperature is not None else 0.0
        n = len(sentences)

        # Layer 2: check for complete pickle
        if cache_file is not None:
            pkl_path = cache_file.rsplit(".", 1)[0] + ".pkl"
            cached = _load_explanation_pickle(pkl_path, n)
            if cached is not None:
                print(f"    Loaded complete cache from {pkl_path}")
                return cached

        # Layer 1: load partial JSONL cache for resume
        skip_indices: Set[int] = set()
        cached_results: Dict[int, Explanation] = {}
        if cache_file is not None and os.path.exists(cache_file):
            cached_results, skip_indices = _load_jsonl_cache(cache_file)
            if len(skip_indices) == n:
                # All indices present — reconstruct and save pickle
                results = [cached_results[i] for i in range(n)]
                pkl_path = cache_file.rsplit(".", 1)[0] + ".pkl"
                _save_explanation_pickle(results, pkl_path)
                print(f"    All {n} results found in JSONL cache — saved pickle")
                return results

        async def _run():
            results = await self._generate_batch_async(
                sentences, temp, show_progress,
                cache_file=cache_file, skip_indices=skip_indices,
            )
            # Merge cached results back in
            for idx, exp in cached_results.items():
                results[idx] = exp
            return results

        # Handle nested event loops (e.g., in Jupyter)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            results = loop.run_until_complete(_run())
        except RuntimeError:
            results = asyncio.run(_run())

        # Layer 2: save complete pickle
        if cache_file is not None:
            pkl_path = cache_file.rsplit(".", 1)[0] + ".pkl"
            _save_explanation_pickle(results, pkl_path)
            print(f"    Saved complete cache to {pkl_path}")

        return results

    def generate_single(
        self,
        sentence: str,
        temperature: Optional[float] = None,
    ) -> Explanation:
        """Generate explanation for a single sentence."""
        results = self.generate_batch([sentence], temperature=temperature, show_progress=False)
        return results[0]

    def print_stats(self):
        """Print generation statistics."""
        print(f"\nAPI Generation Stats:")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Failed requests: {self.failed_requests}")
        print(f"  Total tokens (in): {self.total_tokens_in:,}")
        print(f"  Total tokens (out): {self.total_tokens_out:,}")
        est_cost_in = self.total_tokens_in / 1_000_000 * 0.30
        est_cost_out = self.total_tokens_out / 1_000_000 * 0.60
        print(f"  Estimated cost: ${est_cost_in + est_cost_out:.2f}")


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
    generator,
    sentences: List[str],
    n_samples: int = 3,
    sample_temperature: float = 0.7,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
) -> List[ExplanationWithStability]:
    """
    Generate explanations with stability for a batch of sentences.

    Uses batched LLM inference for efficiency (~10-20x faster than sequential).

    Args:
        generator: ExplanationGenerator or APIExplanationGenerator instance
        sentences: List of input sentences
        n_samples: Number of samples per sentence
        sample_temperature: Temperature for sampling
        embedding_model: Model for rationale embedding
        show_progress: Whether to show progress
        cache_dir: If provided, directory for per-pass cache files.

    Returns:
        List of ExplanationWithStability objects
    """
    from sentence_transformers import SentenceTransformer

    n = len(sentences)
    print(f"  Generating {n} explanations with {n_samples} samples each...")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_file(pass_name: str) -> Optional[str]:
        if cache_dir is None:
            return None
        return os.path.join(cache_dir, f"{pass_name}.jsonl")

    # Step 1: Batch generate primary explanations (deterministic)
    print(f"  [1/{n_samples+1}] Generating primary explanations (temp=0.0)...")
    primary_explanations = generator.generate_batch(
        sentences, temperature=0.0, show_progress=show_progress,
        cache_file=_cache_file("primary"),
    )

    # Step 2: Batch generate stability samples
    all_samples = [primary_explanations]  # List of lists
    for sample_idx in range(n_samples - 1):
        print(f"  [{sample_idx+2}/{n_samples+1}] Generating stability sample {sample_idx+1} (temp={sample_temperature})...")
        samples = generator.generate_batch(
            sentences, temperature=sample_temperature, show_progress=show_progress,
            cache_file=_cache_file(f"stability_{sample_idx}"),
        )
        all_samples.append(samples)
    
    # Step 3: Compute stability metrics in batch
    print(f"  [{n_samples+1}/{n_samples+1}] Computing stability metrics...")
    
    # Load embedding model once
    embedder = SentenceTransformer(embedding_model)
    
    # Collect all rationales for batch embedding
    all_rationales = []
    rationale_indices = []  # (example_idx, sample_idx)
    for sample_idx, sample_list in enumerate(all_samples):
        for ex_idx, exp in enumerate(sample_list):
            all_rationales.append(exp.rationale if exp.rationale else "")
            rationale_indices.append((ex_idx, sample_idx))
    
    # Batch embed all rationales
    if all_rationales:
        all_embeddings = embedder.encode(all_rationales, show_progress_bar=show_progress, batch_size=256)
    else:
        all_embeddings = np.zeros((len(all_rationales), 384))
    
    # Reshape embeddings: (n_examples, n_samples, dim)
    embedding_dim = all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else 384
    embeddings_by_example = np.zeros((n, n_samples, embedding_dim))
    for flat_idx, (ex_idx, sample_idx) in enumerate(rationale_indices):
        embeddings_by_example[ex_idx, sample_idx] = all_embeddings[flat_idx]
    
    # Step 4: Compute stability for each example
    print(f"  Assembling results...")
    results = []
    
    for i in range(n):
        samples = [all_samples[s][i] for s in range(n_samples)]
        
        # Compute stability metrics
        # 1. Label agreement
        labels = [s.pred_label for s in samples]
        unique_labels = set(labels)
        most_common_count = max(labels.count(l) for l in unique_labels)
        label_agreement = most_common_count / len(labels)
        
        # 2. Evidence Jaccard (pairwise average)
        jaccard_scores = []
        for j in range(len(samples)):
            for k in range(j + 1, len(samples)):
                jaccard_scores.append(compute_evidence_jaccard(samples[j], samples[k]))
        evidence_jaccard = np.mean(jaccard_scores) if jaccard_scores else 1.0
        
        # 3. Rationale embedding similarity (mean pairwise cosine)
        sample_embeddings = embeddings_by_example[i]  # (n_samples, dim)
        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = sample_embeddings / norms
        sim_matrix = normalized @ normalized.T
        # Get upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(n_samples, k=1)
        pairwise_sims = sim_matrix[upper_indices]
        rationale_similarity = np.mean(pairwise_sims) if len(pairwise_sims) > 0 else 1.0
        
        # Compute reliability score (average of the 3 metrics)
        reliability_score = (label_agreement + evidence_jaccard + rationale_similarity) / 3.0
        
        # Determine dominant label
        from collections import Counter
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0] if labels else "UNKNOWN"
        
        # Create stability object
        stability = StabilityMetrics(
            label_agreement=float(label_agreement),
            evidence_jaccard=float(evidence_jaccard),
            rationale_similarity=float(rationale_similarity),
            reliability_score=float(reliability_score),
            n_samples=n_samples,
            labels=labels,
            dominant_label=dominant_label,
        )
        
        results.append(ExplanationWithStability(
            primary=samples[0],
            stability=stability,
            samples=samples,
        ))
    
    print(f"  Done! Mean reliability: {np.mean([r.stability.reliability_score for r in results]):.3f}")
    return results


# =============================================================================
# Disk Cache Helpers
# =============================================================================

def _load_jsonl_cache(path: str) -> Tuple[Dict[int, Explanation], Set[int]]:
    """Load partial JSONL cache from disk.

    Each line is JSON: {"i": index, "e": explanation_dict, "raw": raw_output}

    Returns:
        Tuple of (index→Explanation dict, set of completed indices)
    """
    cache = {}
    if not os.path.exists(path):
        return cache, set()

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                idx = record["i"]
                exp = Explanation.from_dict(record["e"], raw_output=record.get("raw"))
                cache[idx] = exp
            except (json.JSONDecodeError, KeyError):
                continue

    return cache, set(cache.keys())


def _save_explanation_pickle(explanations: List[Explanation], path: str) -> None:
    """Save list of Explanation objects to pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(explanations, f)
    os.replace(tmp_path, path)


def _load_explanation_pickle(path: str, expected_n: int) -> Optional[List[Explanation]]:
    """Load list of Explanation from pickle if it exists and has expected length.

    Returns:
        List of Explanation if valid, None otherwise.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and len(data) == expected_n:
            return data
        print(f"  Cache pickle {path} has {len(data)} items, expected {expected_n} — ignoring")
        return None
    except Exception as e:
        print(f"  Failed to load cache pickle {path}: {e}")
        return None


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
    task_config: Optional[TaskConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract LLM predictions and confidence.

    Args:
        explanations: List of Explanation objects
        task_config: Optional task config for label mapping.
                     If None, uses SST-2 default (POSITIVE=1, NEGATIVE=0).

    Returns:
        Tuple of (predicted_labels, confidence_scores)
    """
    labels = []
    confidence = []

    if task_config is not None:
        reverse_map = task_config.reverse_label_map
        reverse_map["UNKNOWN"] = -1
    else:
        reverse_map = {"POSITIVE": 1, "NEGATIVE": 0, "UNKNOWN": -1}

    for exp in explanations:
        labels.append(reverse_map.get(exp.pred_label.upper(), -1))
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

