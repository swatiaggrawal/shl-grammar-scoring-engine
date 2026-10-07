# Grammar Scoring Engine for Spoken English

Solution for the **SHL Hiring Assessment 2026** (Research Engineer) private Kaggle challenge.

**Task:** predict a grammar score from 0 to 5 for a 45 to 60 second spoken audio clip (769 labelled training clips, 216 test clips).

## Results

| Metric | Value |
|---|---|
| **Training RMSE** (final ensemble) | **0.137** |
| Cross-validated RMSE (5-fold) | 0.505 |
| Cross-validated Pearson (5-fold) | 0.916 |
| Baseline RMSE (always predict the mean) | 1.238 |
| Public leaderboard score | 0.4197 |

Training RMSE is optimistic because the models are fit on those clips. The cross-validated numbers are the honest estimate of performance on unseen audio.

| Model | Training RMSE | CV RMSE | CV Pearson |
|---|---|---|---|
| Fluency features only (Ridge) | - | 1.019 | 0.568 |
| WavLM-base | 0.138 | 0.539 | 0.901 |
| WavLM-large | 0.153 | 0.523 | 0.907 |
| HuBERT-large | 0.224 | 0.542 | 0.900 |
| Whisper-medium (encoder) | 0.097 | 0.517 | 0.911 |
| **Ensemble (mean of the four)** | **0.137** | **0.505** | **0.916** |

## Approach

```
audio clip (16 kHz, <= 60 s)
   |
   +--> 9 fluency features (duration, speech ratio, pauses, loudness) ---------+
   |                                                                            |
   +--> WavLM-base            -> mean-pool over time, per layer --+             |
   +--> WavLM-large           -> mean+std pool, per layer --------+--> layer    |
   +--> HuBERT-large          -> mean+std pool, per layer --------+    average  |
   +--> Whisper-medium encoder -> mean-pool over time, per layer -+    + fluency|
                                                                      <--------+
                                                                  |
                                              one SVR per encoder (tuned by CV)
                                                                  |
                                                  average of 4 predictions, clipped to [0, 5]
```

* **No speech-to-text.** ASR decoders tend to repair grammar while transcribing, which would hide the errors being scored. The notebook uses the encoders' internal representations directly (for Whisper, only the encoder).
* **Layer-wise embeddings.** Hidden states of every layer are pooled over time. A small grid search picks, per encoder, which layers to average, plus the SVR `C` and `epsilon`.
* **Validation.** 5-fold cross-validation, stratified by score range, with out-of-fold predictions for all reported metrics.
* **Ensemble.** A plain average of the four encoder models. Their errors are only partly correlated, so the average beats every single model.

## Repository contents

| File | Description |
|---|---|
| `finalnotebook.ipynb` | Full pipeline with outputs: EDA, feature extraction, validation, tuning, final fit, evaluation plots, submission and written report |
| `submission.csv` | Predictions for the 216 test clips (`filename,label`) |
| `README.md` | This file |

## How to run

1. Open the notebook on Kaggle (it uses `kagglehub` to locate the competition data).
2. Enable **GPU (T4)** under Settings > Accelerator, and turn **Internet** on (needed to download the pretrained models from Hugging Face).
3. Run all cells. A fresh run takes about 20 to 25 minutes. Embeddings are cached in `/kaggle/working/cache`, so an interrupted run resumes and reruns are fast.
4. The notebook writes `/kaggle/working/submission.csv`.

Seeds are fixed (`SEED = 42`). Rerunning on the same setup reproduced the same leaderboard score (0.4197).

## Other experiments (run during development, not in the notebook)

| Idea | Outcome |
|---|---|
| Whisper transcripts, then sentence embeddings, GPT-2 loss and text statistics | Text-only CV RMSE about 0.93. Blending with the audio ensemble made CV and the leaderboard worse, so it was dropped. |
| Training only on clips with score above 0 | No gain (CV RMSE 0.564 vs 0.557 for the same model on non-zero clips). |
| Small attention head on frame-level Whisper features | Alone CV RMSE 0.61. Blending gained 0.006, within fold-to-fold noise, so it was not kept. |

## Notes and limitations

* `sample_submission.csv` has 204 rows and only 25 of its filenames match `test.csv`, so the submission uses the 216 filenames from `test.csv`.
* 37 training clips (about 5%) have score 0, outside the 1 to 5 rubric. They are not silent, so they are probably unscorable audio. The models separate them well (mean out-of-fold prediction 0.23).
* Predictions are shrunk toward the middle of the scale: the highest scores are under-predicted and the lowest over-predicted, which is typical for regression on noisy human ratings.
* With 769 training clips, small differences between configurations are within noise, and tuning on the same folds that report the score is slightly optimistic.
* Mean pooling compresses a minute of speech into one vector, so localised grammatical errors are diluted. Fine-tuning an encoder end to end or learned pooling over time are possible next steps, at the cost of a more complex solution.
