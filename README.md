# Mushroom Classification

Binary classification of mushrooms (edible / poisonous) from images using MobileNetV3-Small with transfer learning.

The project compares two pipelines:
- **EXP1** — original images
- **EXP2** — images pre-processed with GrabCut segmentation

Five segmentation algorithms were evaluated before selecting GrabCut.

## Results

| Experiment | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|----|
| EXP1 (original) | 72.3% | 74.1% | 68.6% | 71.3% |
| EXP2 (GrabCut) | 71.3% | 72.0% | 69.6% | 70.8% |

Segmentation does not improve overall accuracy, but reduces safety-critical false negatives (poisonous classified as edible: 471 → 456).

## Dataset

[Mushroom1](https://www.kaggle.com/datasets/zlatan599/mushroom1) — 169 species, downloaded automatically via `kagglehub`.

## Setup

```bash
pip install torch torchvision kagglehub opencv-python scikit-image scikit-learn matplotlib pandas
```

Open `main.ipynb` in Jupyter and run all cells.

## Notebook structure

1. Imports and configuration
2. Dataset download and exploration
3. Segmentation algorithm comparison
4. GrabCut pipeline implementation
5. DataLoader setup (EXP1 and EXP2)
6. MobileNetV3 model definition
7. Training loop (AdamW, ReduceLROnPlateau, early stopping)
8. Evaluation and metrics
9. Confusion matrix analysis
10. Conclusions

## Authors

Leonardo Berselli & Thomas Ottonello — Master in AI Systems, University of Trento (2025)

*Signal, Image and Video Processing course*
