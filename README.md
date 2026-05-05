# Tokenizer Project

This repository is a research workspace that combines several upstream projects (BLT, OLMo, MYTE, Parity-aware BPE) plus a small set of scripts/configs for training, evaluation, and analysis.

## Repository layout

- `blt/` — Byte Latent Transformer (BLT) codebase (upstream: https://github.com/facebookresearch/blt/)
- `OLMo/` — AI2 OLMo training/inference codebase (upstream: https://github.com/allenai/OLMo)
- `myte/` — MYTE tokenizer / MyT5 utilities
- `parity-aware-bpe/` — Parity-aware BPE and Byte-level BPE training code (upstream: https://github.com/swiss-ai/parity-aware-bpe)
- `language_model_training/` — training configuration YAMLs for all tokenizers except BLT (OLMo-style configs for different tokenizer variants)
- `data/` — dataset preparation scripts for tokenizer and language model training
- `figures/` — scripts for generating figures in paper

Top-level scripts:

- `anova.py` - two-way mixed ANOVA test to analyze the effects of tokenizer choice and script type on tokenizer parity.
- `byte_count.py` — counts total UTF-8 bytes across a folder of `.txt` files.
- `lm_eval_pairwise_significance_test.py` — paired bootstrap significance tests over LM evaluation outputs.
- `mt_pairwise_significance_test.py` — paired bootstrap significance tests over MT score CSVs.
- `token_count_olmo2.py` — estimates token counts for parallel MT data using the `allenai/OLMo-2-0425-1B` tokenizer.

## Getting started

This is a multi-project repo. Each component has its own dependency constraints and it’s easiest to follow the component-specific installation instructions:

- **BLT**: [blt/README.md](blt/README.md)
- **OLMo**: [OLMo/README.md](OLMo/README.md)
- **MYTE**: [myte/README.md](myte/README.md)

You may also install the required environment via Conda using the provided [environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate tokenizer-project
```

If you only want to run the **top-level analysis scripts** (token/byte counting and significance tests), you can get away with a lightweight environment.

## Workflows

### Tokenizer training
- **BLT**: Modify the `blt/bytelatent/configs/entropy_model.yaml` configuration file to point to your training data and run `blt/bytelatent/train.py`.
- **MYTE**: Modify and run `myte/scripts/job_construct_mc4_lexicon.sh` to construct the training data and learn the respective morphemes.
- **Parity-aware BPE**: Modify `parity-aware-bpe/train.sh` to run `parity-aware-bpe/parity_aware_learn_bpe.py`.
- **Byte-level BPE**: Modify `parity-aware-bpe/train.sh` to run `parity-aware-bpe/parity_aware_learn_bpe.py`.

### Tokenizer evaluation
- **All tokenizers**: Modify the respective tokenizer paths in `blt/downstream/intrinsic_eval.py` and run it.

### Language model training
- **BLT**: Modify `blt/bytelatent/configs/blt_1b_olmo_stage1.yaml` and `blt/bytelatent/configs/blt_1b_olmo_stage2.yaml` and run `blt/bytelatent/train.py`.
- **All other tokenizers**: Modify the respective YAML configuration files in `language_model_training/` and run `OLMo/scripts/train.py` with the desired config.

### Language model evaluation (Classification benchmarks)
- **BLT**: Modify `blt\apps\main\configs\eval.yaml` to point to your trained BLT model and run `blt/bytelatent/eval.py`.
- **All other tokenizers**: Modify and run `blt/downstream/lm_eval_harness.py`.

### Language model evaluation (Machine translation benchmarks)
- **BLT**: Modify and run `blt/downstream/mt_blt.py`.
- **All other tokenizers**: Modify and run `blt/downstream/mt.py`.

## License

The code in this repository is made available for **non-commercial, research use only**, subject to the licenses of its components.

| Component | License | Commercial Use | Source |
|-----------|---------|---------------|--------|
| BLT | CC BY-NC 4.0 | ❌ Non-commercial only | https://github.com/facebookresearch/blt/ |
| OLMo | Apache 2.0 | ✅ Allowed | https://github.com/allenai/OLMo |
| Parity-aware BPE | MIT | ✅ Allowed | https://github.com/swiss-ai/parity-aware-bpe |

**Important:** Because this repository incorporates code under **CC BY-NC 4.0**, the repository as a whole may only be used for **non-commercial purposes**.

See [README_LICENSE_SECTION.md](README_LICENSE_SECTION.md) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.
