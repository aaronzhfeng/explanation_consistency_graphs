# Debugging Checklist

> When things go wrong (they will)

---

## 1. Model Produces Garbage

### Symptoms
- Output is random tokens
- Output repeats indefinitely
- Output doesn't match expected format

### Checks
- [ ] Is the model loaded correctly? (check weights loaded message)
- [ ] Is the tokenizer correct for this model?
- [ ] Is the prompt format correct? (chat template, special tokens)
- [ ] Is temperature reasonable? (not 0 for diversity, not too high)
- [ ] Is max_new_tokens set appropriately?

### Common Fixes
```python
# Check prompt format
print(tokenizer.decode(inputs['input_ids'][0]))

# Check generation config
print(model.generation_config)

# Try greedy first (temp=0) to see base behavior
output = model.generate(**inputs, do_sample=False, max_new_tokens=100)
```

---

## 2. Training Loss Doesn't Decrease

### Symptoms
- Loss stays flat
- Loss oscillates wildly
- Loss is NaN

### Checks
- [ ] Is learning rate appropriate? (try 1e-5 to 1e-4)
- [ ] Is data loading correctly? (inspect actual batches)
- [ ] Are gradients flowing? (check for None gradients)
- [ ] Is model in training mode? (`model.train()`)

### Common Fixes
```python
# Check data loading
batch = next(iter(dataloader))
print(batch.keys())
print(tokenizer.decode(batch['input_ids'][0]))

# Check gradients
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

---

## 3. Loss Decreases But Model Doesn't Learn

### Symptoms
- Loss goes down nicely
- But model output is still wrong
- Metrics don't improve

### Checks
- [ ] Is the data actually what you think it is?
- [ ] Is the loss computed on the right tokens?
- [ ] Is there a data loading bug (wrong fields used)?
- [ ] Is the model learning a shortcut?

### The SOKRATES Bug
This exact bug happened: loss dropped from 3.5 → 0.01, but model only learned to output `CONCLUDE`. Root cause: data loader wasn't using the right field (`training_text`).

### Verification
```python
# Load one example and check what the model sees
example = dataset[0]
print("Input:", example['input'])
print("Target:", example['output'])

# Generate and compare
output = model.generate(...)
print("Model output:", output)
```

---

## 4. Out of Memory (OOM)

### Symptoms
- CUDA out of memory error
- Process killed

### Fixes (in order of preference)
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision (bf16/fp16)
4. Reduce sequence length
5. Use smaller model
6. Use gradient accumulation (same effective batch)

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use bf16
model = model.to(torch.bfloat16)

# Gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
)
```

---

## 5. Multi-GPU Issues

### Symptoms
- Only one GPU working
- Processes hang
- Inconsistent results across GPUs

### Checks
- [ ] Is `CUDA_VISIBLE_DEVICES` set correctly?
- [ ] Is distributed initialized properly?
- [ ] Is `device_map='auto'` conflicting with DDP?
- [ ] Is data split across GPUs?

### Common Fixes
```python
# Don't use device_map with accelerate
# Wrong:
model = AutoModel.from_pretrained(..., device_map='auto')

# Right (let accelerate handle it):
model = AutoModel.from_pretrained(...)

# Explicit device placement
local_rank = int(os.environ.get('LOCAL_RANK', 0))
device = f'cuda:{local_rank}'
model = model.to(device)
```

---

## 6. Evaluation Metrics Don't Make Sense

### Symptoms
- Accuracy is 0 or 100%
- Metrics don't match manual inspection
- Results are inconsistent

### Checks
- [ ] Is the evaluation data loaded correctly?
- [ ] Is the prediction parsing correct?
- [ ] Are labels in the expected format?
- [ ] Is there a mismatch between model output format and parser?

### Verification
```python
# Manual check
pred = model_output
gold = ground_truth
print(f"Predicted: {pred}")
print(f"Gold: {gold}")
print(f"Match: {pred == gold}")

# Check a batch manually before trusting aggregate metrics
```

---

## 7. Generation is Too Slow

### Symptoms
- Inference takes forever
- GPU utilization is low during generation

### Fixes
1. Use vLLM instead of HuggingFace generate()
2. Batch inputs (don't generate one at a time)
3. Use smaller max_new_tokens
4. Enable KV cache
5. Use tensor parallelism for large models

```bash
# vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model outputs/model \
    --tensor-parallel-size 1

# Data parallel: run multiple vLLM processes
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python generate.py --shard $gpu &
done
```

---

## 8. DPO Not Working

### Symptoms
- No improvement over SFT
- Model gets worse

### Checks
- [ ] Are preference pairs correct? (chosen actually better?)
- [ ] Is there diversity in traces? (not all identical)
- [ ] Is beta appropriate? (0.1 is common)
- [ ] Is learning rate low enough? (5e-7 typical)

### Common Fixes
```python
# Check preference pair quality
for pair in pairs[:10]:
    print("Chosen:", pair['chosen'][:100])
    print("Rejected:", pair['rejected'][:100])
    print("---")

# Ensure temperature > 0 for trace generation
# Greedy (temp=0) produces identical traces = useless for DPO
```

---

## Universal Debugging Strategy

1. **Isolate the problem**: What's the smallest reproduction?
2. **Check inputs**: Is the data what you think it is?
3. **Check outputs**: What is actually being produced?
4. **Add logging**: Print intermediate values
5. **Simplify**: Does a simpler version work?
6. **Compare**: Does it work with a known-good example?
7. **Ask the AI**: Describe the symptoms, get suggestions

