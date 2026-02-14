# Data and Collators

## Stage-to-Processor Mapping
- File: `src/llamafactory/data/loader.py`
- Mapping:
  - `stage=rm` -> `PairwiseDatasetProcessor`
  - `stage=kto` -> `FeedbackDatasetProcessor`
  - `stage=ppo` -> `UnsupervisedDatasetProcessor`
  - `stage=sft` -> supervised processor

## Pairwise Data (RM and DPO)
- File: `src/llamafactory/data/processor/pairwise.py`
- Produces:
  - `chosen_input_ids`, `chosen_labels`
  - `rejected_input_ids`, `rejected_labels`
- Collated by `PairwiseDataCollatorWithPadding` into 2N examples.

## KTO Data
- File: `src/llamafactory/data/processor/feedback.py`
- Produces target sample + KL sample:
  - `input_ids`, `labels`
  - `kl_input_ids`, `kl_labels`
  - `kto_tags` boolean (desirable/undesirable)
- Collated by `KTODataCollatorWithPadding`.

## PPO Data
- File: `src/llamafactory/data/processor/unsupervised.py`
- Produces prompt-context inputs where model generates response online.
- Reward is computed during training loop, not pre-embedded in dataset.

## Validation Rules You Should Remember
- Ranking mismatch is rejected:
  - RM expects ranking data.
  - Non-RM stages reject ranking-only datasets.
- Invalid dialogue turn structure is dropped.
- Tokenization and truncation depend on `cutoff_len` and template logic.
