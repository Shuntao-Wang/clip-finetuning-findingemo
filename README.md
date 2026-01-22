# Fine-Tuned CLIP for Scene Emotion Understanding

## Overview
This repository contains the core implementation of fine-tuning CLIP (ViT-L/14) on the FindingEmo dataset for scene-level emotion understanding.

This module corresponds to the visual emotion modeling component of my undergraduate thesis titled:
_"Fine-Tuned CLIP-Based Multimodal Emotion Modeling for User Engagement Prediction"._

The full thesis involves multimodal data collection from Xiaohongshu posts, text-based emotion scoring via LLM APIs, and downstream empirical analyses.

Due to scale and privacy considerations, only the CLIP fine-tuning module is released here.

## Dataset
I use the FindingEmo dataset, a large-scale scene emotion dataset consisting of approximately 25,000 real-world social images with 24 fine-grained emotion labels.

To ensure class balance and training efficiency, a subset is sampled (200–500 images per emotion category).

## Method
The fine-tuning pipeline consists of three steps:

1. **Subset Preparation**
   - Download FindingEmo
   - Balanced sampling
   - Convert emotion labels into CLIP-style textual prompts

2. **CLIP Fine-Tuning**
   - Backbone: CLIP ViT-L/14
   - Objective: image-text contrastive learning
   - Epochs: ~10
   - Learning rate: 1e-5
   - Mixed-precision training

3. **Zero-Shot Inference**
   - No classification head required
   - Emotion prediction via text prompt similarity

## Usage

    pip install -r requirements.txt
    python prepare_subset.py
    python finetune_findingemo.py
    python zero_shot_inference.py

## Model Loading

After fine-tuning, the model can be loaded as:

    import clip
    model, preprocess = clip.load("ViT-L-14-findingemo", device="cuda")

## Notes

The full dataset and trained model weights are not included due to size and licensing considerations.

This repository focuses on methodological transparency and reproducibility.

## Customization

This codebase is designed to be adaptable to dataset updates and different usage scenarios.
Users may need to adjust the following parts:

1. **Annotation Structure**
   - In `prepare_subset.py`, emotion labels are extracted from the annotation metadata.
   - Since the FindingEmo dataset may evolve over time, the exact field name
     (e.g., `emotion` vs. `emotions`) may need to be adjusted according to the
     actual annotation structure.

2. **Test Image Path**
   - In `zero_shot_inference.py`, the example test image path (`test_scene.jpg`)
     serves as a placeholder.
   - Replace it with the path to any real-world social scene image for inference.

These adjustments are expected and do not affect the core methodology of CLIP fine-tuning
and zero-shot emotion classification.

## Author

Shuntao Wang

