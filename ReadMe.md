# Lab 2 - Fine-Tuning a Large Language Model

## Overview

In this lab we fine-tuned the open-source base model `unsloth/Llama-3.2-3B-Instruct` on Maxime Labonne's FineTome-100k instruction dataset using LoRA (Low-Rank Adaptation) with QLoRA quantization via Unsloth and `trl.SFTTrainer`. The training used the following hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 |
| `num_train_epochs` | 1 |
| `learning_rate` | 2e-4 |
| `warmup_steps` | 5 |
| `weight_decay` | 0.01 |
| `optimizer` | adamw_8bit |

---

## Evaluation Methodology

We evaluated both the base model and the fine-tuned model on **100 held-out examples** from a 15% test split of FineTome-100k. For each example we used the conversation history as input and treated the last assistant turn as the reference answer.

### Metrics Used

We used **ROUGE scores** (Recall-Oriented Understudy for Gisting Evaluation) to measure the overlap between model-generated responses and reference answers:

- **ROUGE-1**: Measures unigram (single word) overlap between the generated text and reference. Higher scores indicate better word-level similarity.
- **ROUGE-2**: Measures bigram (two consecutive words) overlap. This captures phrase-level similarity and is more sensitive to word order.
- **ROUGE-L**: Measures the longest common subsequence between generated and reference text. This captures sentence-level structure and fluency.

All scores use the F-measure (harmonic mean of precision and recall) and range from 0 to 1, where higher is better.

### Generation Settings

We used **deterministic generation** (greedy decoding with `do_sample=False`) for both models to ensure reproducible and comparable results. Each response was limited to 256 new tokens.

---

## Results

The fine-tuned model clearly outperformed the base model across all ROUGE metrics:

| Metric | Base Model | Fine-Tuned Model | Improvement |
|--------|------------|------------------|-------------|
| ROUGE-1 | 0.4732 | 0.5323 | **+12.5%** |
| ROUGE-2 | 0.2255 | 0.2849 | **+26.4%** |
| ROUGE-L | 0.2856 | 0.3521 | **+23.3%** |

These results demonstrate that even a single epoch of LoRA fine-tuning on FineTome-100k produces substantial improvements in response quality. The largest gain was in ROUGE-2 (+26.4%), indicating that the fine-tuned model better captures phrase-level patterns from the training data.

### Response Length Analysis

| Metric | Base Model | Fine-Tuned Model | Reference |
|--------|------------|------------------|-----------|
| Mean length (words) | 165.0 | 155.7 | 216.3 |
| Median length (words) | 176.0 | 165.5 | 199.0 |

The fine-tuned model produces slightly more concise responses while achieving higher ROUGE scores, suggesting improved information density.

---

## Improving Model Performance

### (a) Model-Centric Approach

A model-centric approach keeps the data fixed and focuses on changing the model architecture, training configuration, or optimization procedure. Below are concrete strategies for further improvement:

#### Hyperparameter Tuning

- **Learning rate**: Sweep over values such as `1e-4`, `2e-4`, `5e-4` to find the optimal learning rate. Our current setting of `2e-4` is a reasonable default but may not be optimal.
- **Learning rate schedule**: Experiment with cosine decay or cosine with warm restarts instead of constant learning rate.
- **Training epochs**: Train for 2–3 epochs with early stopping based on validation loss to potentially improve convergence.
- **Batch size**: Increase effective batch size (via `gradient_accumulation_steps`) within memory constraints for more stable gradients.
- **Warmup steps**: Adjust warmup duration (e.g., 10–100 steps) to improve training stability.
- **Weight decay**: Test different regularization strengths (e.g., `0.001`, `0.01`, `0.1`) to control overfitting.

#### LoRA Configuration

- **Rank (r)**: Increase LoRA rank (e.g., from 16 to 32 or 64) to allow more expressive adapter updates, at the cost of increased memory.
- **Alpha scaling**: Adjust the LoRA alpha parameter to control the magnitude of adapter contributions.
- **Target modules**: Experiment with applying LoRA to different layer types (attention only, MLP layers, or both) and different layer ranges.

#### Model Architecture

- **Base model selection**: Compare different foundation models such as `Llama-3.2-1B-Instruct` (faster inference) or `Llama-3.1-8B-Instruct` (potentially higher quality but slower).
- **Quantization**: Compare 4-bit (QLoRA) vs 8-bit quantization to understand the quality-speed tradeoff.

#### Training Procedure

- **Gradient clipping**: Add gradient clipping to prevent exploding gradients and improve training stability.
- **Mixed precision**: Ensure optimal use of mixed precision training for faster iteration.

### (b) Data-Centric Approach

A data-centric approach keeps the model and training loop mostly fixed and focuses on improving or extending the training data. Below are concrete strategies:

#### Data Quality Improvements

- **Filter low-quality examples**: Remove very short, unclear, or noisy instruction-response pairs from FineTome-100k to increase average signal per batch.
- **Deduplicate**: Remove near-duplicate examples that may cause the model to overfit to specific patterns.
- **Balance task types**: If the target application focuses on specific capabilities (e.g., reasoning, coding, explanation), up-sample those categories and down-sample less relevant ones.

#### Additional Data Sources

Augment FineTome-100k with other high-quality open-source instruction datasets:

| Dataset | Focus Area | Potential Benefit |
|---------|------------|-------------------|
| OpenAssistant Conversations | Multi-turn dialogue | Improved conversational ability |
| GSM8K / MetaMath | Math reasoning | Better mathematical problem-solving |
| CodeAlpaca / Code-Feedback | Programming tasks | Improved code generation |
| FLAN Collection | Diverse NLP tasks | Broader task coverage |
| UltraChat | Long-form dialogue | Better handling of extended conversations |

#### Domain-Specific Fine-Tuning

- **Curriculum learning**: Start training on general instructions, then gradually shift to more specialized or difficult examples.
- **Task-specific adapters**: Train separate LoRA adapters for different domains (math, code, creative writing) and select the appropriate adapter at inference time.

#### Data Alignment

- **Match UI format**: If the final application expects specific output formats (e.g., step-by-step reasoning, JSON responses), construct or filter training examples that demonstrate these formats.
- **User feedback loop**: In production, log anonymized user interactions (if permitted) to create a fine-tuning set that reflects real usage patterns.

---

## Conclusion

Our fine-tuning pipeline demonstrates measurable improvements over the base model, with ROUGE scores increasing by 12–26% on a held-out test set. The model-centric and data-centric strategies outlined above provide clear directions for further performance gains. The most promising next steps would be:

1. **Hyperparameter sweep** on learning rate and number of epochs
2. **Increase LoRA rank** to allow more expressive updates
3. **Mix in domain-specific datasets** (e.g., math reasoning or code) to improve performance on specialized tasks
