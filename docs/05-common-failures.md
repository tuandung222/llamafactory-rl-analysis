# Common Failures and Fixes

## Error: reward_model is necessary for PPO training
Cause:
- Missing `reward_model` while `stage: ppo`.
Fix:
- Add valid `reward_model` path or API URL.

## Error: dpo_label_smoothing is only valid for sigmoid loss
Cause:
- You set `pref_loss=orpo/simpo` with non-zero `dpo_label_smoothing`.
Fix:
- Reset `dpo_label_smoothing: 0.0`.

## Error: Incompatible TRL version for PPO
Cause:
- Installed TRL out of supported range.
Fix:
```bash
pip install 'trl>=0.8.6,<=0.9.6'
```

## Unexpectedly no rewards in DPO eval
Cause:
- Reference model equals policy model in eval-only setup.
Fix:
- Provide explicit `ref_model` for clean reward comparison.

## KTO warning: dataset only has one preference type
Cause:
- Only desirable or only undesirable samples.
Fix:
- Balance labels and check `kto_tag` mapping.

## Quality collapse after preference training
Checklist:
1. Validate chosen/rejected data quality.
2. Reduce LR.
3. Increase prompt diversity.
4. Add `pref_ftx` to anchor helpfulness.
