# AI Speak Baseline

This project is a complete baseline for the blendshape competition described in `kategorijaB_baza.pdf`.

The baseline is designed to run in Google Colab and includes:

- data unpacking from the provided zip archives
- manifest creation from `wav`, `csv`, transcript, and phoneme files
- cached acoustic feature extraction
- a causal audio + text model for 52 ARKit blendshapes
- training, validation, and checkpoint saving
- inference that exports `CSV` files and `meta.json`

## Expected raw files

Place these archives in the project root before running the notebook:

- `spk08_blendshapes.zip`
- `spk14_blendshapes.zip`
- `labels_aligned.zip`
- `audio_synth.zip` (optional)
- `avatar.zip` (optional)

## Baseline choices

- Output FPS: `60`
- Blendshape order: fixed to the ARKit 52 order from the task PDF
- Lookahead: `0 ms`
- Input modality: audio + optional transcript text
- Model: causal temporal convolutional regressor with transcript conditioning

## Colab quick start

1. Upload the project folder to Colab or clone it from Git.
2. Put the competition zip files in the project root.
3. Open `notebooks/ai_speak_colab.ipynb`.
4. Run all cells.

The notebook will:

1. unpack the archives into `artifacts/data/extracted`
2. build `artifacts/data/manifest.csv`
3. cache features in `artifacts/data/features`
4. train the model and save `artifacts/checkpoints/best.pt`
5. generate submission files in `submission/`

## Submission format

Inference writes:

- one `.csv` per input `.wav`
- one `meta.json` summarizing
  - `inference_time_sec`
  - `rtf`
  - `lookahead_ms`
  - `fps_out`

If the organizer later publishes an exact `meta.json` schema, only the serializer in `src/ai_speak/pipeline.py` needs adjustment.
