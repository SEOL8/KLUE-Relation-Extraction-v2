# KLUE Relation Extraction, re-validation

This is an updated version of [SEOL8/KLUE-Relation-Extraction](https://github.com/SEOL8/KLUE-Relation-Extraction).

The original ablation went from a plain-text bert-base (micro F1 0.2042) straight to
roberta-base with typed entity markers (0.6969). That single step changed two things at
once, the backbone and the marker scheme, so the +0.49 jump could not be attributed to
either one. This version adds the missing arm, bert-base **with** markers, so the marker
effect and the backbone effect can be read separately.

## What the runs show

| model | backbone | markers | lr | micro F1 | AUPRC |
|---|---|---|---|---|---|
| (reference) | bert-base | no | 2e-5 | 0.2042 | 0.3985 |
| M1_marker | bert-base | yes | 2e-5 | 0.7036 | 0.8152 |
| M2 | roberta-base | yes | 2e-5 | 0.6892 | 0.8130 |
| M3 | roberta-large | yes | 1e-5 | 0.7227 | 0.8480 |
| M3_LS | roberta-large | yes + label smoothing | 1e-5 | 0.7556 | 0.7188 |

Effect breakdown:

- markers on bert-base: 0.2042 to 0.7036, +0.4994
- backbone swap, bert-base to roberta-base: -0.0144
- size, roberta-base to roberta-large: +0.0335
- label smoothing on M3: +0.0329 micro F1, but AUPRC drops from 0.8480 to 0.7188

So the typed markers carry almost the entire gain. The backbone swap barely moves the
score. Label smoothing nudges micro F1 up while flattening the probabilities, which costs
AUPRC and peaks at the first epoch, so M3 is the more balanced model to rely on.

## Two micro-F1 conventions

`micro_f1` in the notebooks drops rows whose true label is `no_relation`, then runs micro
F1 on the rest. The KLUE definition keeps every row and restricts the label set to 1..29,
which also penalises `no_relation` rows predicted as a relation. On M3 the first reads
0.7227 and the second reads 0.6887; the 864 over-detected `no_relation` rows account for
the gap. The model comparison is unaffected since every model uses the same metric, but
the KLUE number is the one to quote against the public 0.65 baseline. `error_analysis.py`
prints both.

## Layout

```
m1_marker.ipynb       bert-base + markers
m2.ipynb              roberta-base + markers
m3.ipynb              roberta-large + markers, saves logits/labels and the best checkpoint
m3_ls.ipynb           roberta-large, then a label-smoothing continuation
compare_models.py     reads results/*/metrics.json, prints the breakdown, draws the bar chart
error_analysis.py     per-class F1 and error composition from the saved M3 predictions
results/              metrics.json and figures from a Kaggle T4 run
```

Each notebook is self-contained: it loads `klue/klue` re, builds the markers, trains one
model, and writes `results/<model>/`. They were built for a Kaggle T4 GPU and fix seed 42.

## Reproduce

1. Run the four notebooks (Kaggle GPU). Run `m3.ipynb` before `error_analysis.py` since it
   saves `m3_val_logits.npy` and `m3_val_labels.npy`.
2. `python compare_models.py` for the table and bar chart.
3. `python error_analysis.py` for the per-class and error-composition figures.

Steps 2 and 3 need only the saved metrics and predictions, so they run on CPU.
