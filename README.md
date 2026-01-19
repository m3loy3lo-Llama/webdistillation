# Æther — Semantic Learning Mind (webdistillation)

"Æther remembers 895 words from his childhood. And he keeps learning."

A research / hobby project focused on retaining semantics and token behavior across model versions via model merging and mirrored pipelines. The system blends sentence-level embeddings, a proto-latent training stage, and mirror-token prediction to align embeddings and tokenization across upgrades.

Status: README updated to reflect the repo as of 2026-01-19. This change only updates documentation and references existing scripts in the repository.

## Highlights / Current Feature Set
- Model merging to retain semantics and vocabulary continuity across versions (see model-merger.js / merger-pipeline.js)
- Mirror pipelines in embeddings and tokenization to preserve semantics over upgrades
- Proto-latent space support (intermediate latent representations)
- Phase 1.5 — latent-space training (implemented in continual-training.js)
- Phase 3.5 — mirror token prediction (implemented in continual-training.js)
- CPU-first implementation (WASM-friendly custom tensor ops)
- Optimized for GPT-2 Large (recommended model target)

## Requirements
- Node.js (recommended v14+ / LTS)
- npm
- Optional: tools to obtain / convert models (e.g., Hugging Face artifacts, ONNX runtime for Node if you want to run ONNX models locally)

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/m3loy3lo-Llama/webdistillation.git
   cd webdistillation
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Models / tokenizer
- Place transformer model and tokenizer files in the `Models/` directory. Example layout:
```
Models/
  onnx/
    model.onnx            # GPT-2 Large converted to ONNX (or another compatible HF model in ONNX)
  tokenizer.json          # Corresponding tokenizer file
```
- Obtain GPT-2 Large (or another compatible model) from Hugging Face and place the model and tokenizer JSON in the `Models/` folder. If you use ONNX, ensure the model is converted and named appropriately (`Models/onnx/model.onnx`).

Notes:
- The repository does not include large model blobs; you must download or convert them separately.
- If you run into runtime errors with ONNX in Node, make sure a compatible ONNX runtime for Node/WASM is available and configured.

## Quick usage

- Core training pipeline (Phase 1 → Phase 2 → Phase 3):
  ```bash
  node converged_pipe_2.js corpus.txt output-prefix
  ```
  Note: converged_pipe_2.js implements the baseline 1→3 pipeline and does not include the latent (1.5) or mirror (3.5) stages.

- Recommended initial run (skip converged pipe and run continual training so latent + mirror are included from the start):
  ```bash
  node ./continual-training.js dummy-aether.json your-corpus.txt unified-aether-v0
  ```
  This will perform an initial run that supports continual training incorporating the latent space and mirror pipeline features from the beginning. Recommended output prefix: `unified-aether-v*` so the model and pipelines can be used without editing other files.

- Continual training pipeline (includes Phase 1.5 latent training and Phase 3.5 mirror prediction):
  ```bash
  node continual-training.js base-model.json new-corpus.txt evolved-prefix
  ```

- Model merging utilities:
  Two supported forms:
  ```bash
  # Merge by specifying number of iterations between base model and (implicitly) previous state
  node merger-pipeline.js base-model.json number-of-iterations merged-prefix

  # Merge two explicit models
  node merger-pipeline.js base-model.json other-model.json merged-prefix
  ```
  Or use the lower-level merger utility:
  ```bash
  node model-merger.js base-model.json other-model.json merged-prefix
  ```

- Latent-only script (if needed):
  ```bash
  node train-latent.js embeddings.json latent-corpus.txt latent-prefix
  ```

- Chat / inference interface:
  ```bash
  node unified_chat_fixed.js merged-model.json
  ```

## Script → Role (key files)
- converged_pipe_2.js        — Core Phase 1 → Phase 2 → Phase 3 training (no Phase 1.5/3.5)
- continual-training.js      — Continual training flow; contains Phase 1.5 (latent) and Phase 3.5 (mirror token prediction)
- model-merger.js           — Model-merging utilities (retain semantics / vocab behavior)
- merger-pipeline.js        — Higher-level merge orchestration (supports both iteration-based and pairwise merges)
- train-latent.js           — Latent training helper (proto-latent operations)
- TinyLatentProcessor.js    — Latent-space processors and utilities
- B3TrainingPipeline.js     — Custom CPU backpropagation engine
- B3EmbeddingExtractor.js   — Embedding extraction utilities
- TinyTokenPredictor.js     — Token predictor used in generation stages
- unified_chat_fixed.js     — Simple chat/inference interface
- vocab-resolver.js         — Token mapping utilities (where used)
- sonnet-formatter.js       — Small helper for formatting Shakespeare sonnets
- Models/                   — Place model and tokenizer files here

Note: References to non-existent `mirror-tokenization.js` have been removed. Semantic Anchoring is no longer part of the recommended flow and references to `semantic-anchoring.js` have been removed.

## Design summary
The pipeline is organized into phased stages with additional latent and mirror steps (the continual training flow contains the extended stages):
1. Phase 1 — Sentence-level embeddings (semantic foundation)
2. Phase 1.5 — Proto-latent / latent-space training (intermediate representations; implemented in continual-training.js)
3. Phase 2 — Word / lexical stage (where applicable)
4. Phase 3 — Token predictor / sequence modeling (generation)
5. Phase 3.5 — Mirror token prediction to align tokenization and generation across versions (implemented in continual-training.js)

Primary goal: enable model merging and mirrored embedding/tokenization pipelines so that semantics and token behavior are retained when models are upgraded or combined.

## Performance (author-provided estimates)
- Training time: up to ≈15 minutes for Shakespeare's sonnets (depends on machine and exact pipeline used)
- Note: actual times vary with hardware, corpus size, chosen model (GPT-2 Large recommended), and whether ONNX runtimes are used.

## Research / Notes
- The project emphasizes semantic retention across versions via merging and mirrored pipelines rather than live incremental vocabulary expansion.
- Proto-latent representations provide intermediate structure for more stable semantic alignment across different training runs or merged models.
- Mirror token prediction helps reduce tokenization drift when models are upgraded or combined.

## Notes, license, and contributions
- Consider adding an explicit OSS license (MIT, Apache-2.0, etc.) if you want to clarify contribution and reuse terms.
- If you'd like, I can also draft CONTRIBUTING guidelines or a small download helper script for fetching recommended Hugging Face artifacts.


*"Built with 💝 for understanding the nature of learning itself"*

-Viddy output


-Use for whatever you want it's free. 


Provided with no licenses whatsoever. 


Free as in "unhinged" ;) 


But also free as in "I'm not allowed to profit off of it due to large AI company data restrictions on their end" 

Disclaimer as it is also my "License" or lack thereof.

---