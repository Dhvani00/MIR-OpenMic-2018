# 📊 Experiment Results Summary

## Experimental Setup

- Dataset: OpenMIC-2018
- Input features: Pre-extracted VGGish embeddings
- Model: One-vs-Rest Random Forest
- Decision threshold: 0.5
- Train/Test split: 80/20
- Evaluation metrics: Micro F1, Macro F1, ROC-AUC

---

## Model Comparison

| Model | Micro F1 | Macro F1 | ROC-AUC |
|-------|----------|----------|----------|
| Baseline RF | 0.5423 | 0.4725 | 0.9546 |
| RF + Class Weighting | 0.5482 | 0.4873 | 0.9677 |
| RF + Oversampling | **0.6072** | **0.5602** | **0.9689** |



## Observations

### 1️⃣ Oversampling shows the strongest improvement
- Significant increase in Macro F1
- Better recognition of rare instruments
- Improved overall AUC

### 2️⃣ Class weighting provides moderate improvement
- Slight improvement in Macro F1
- Less impact compared to oversampling

### 3️⃣ Micro vs Macro Behavior
- Micro F1 improved moderately
- Macro F1 improved substantially
- Indicates better minority-class performance



## Conclusion

Handling class imbalance improves multi-label instrument recognition performance, particularly for rare instruments.

Random oversampling was the most effective strategy in this experimental setup.
