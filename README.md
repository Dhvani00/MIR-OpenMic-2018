# Handling Class Imbalance in Multi-Label Instrument Recognition (Random Forest on OpenMIC-2018)

## 📌 Overview
This project reproduces the Random Forest baseline for multi-label instrument recognition on the OpenMIC-2018 dataset and investigates techniques to handle severe class imbalance.

The goal is to improve recognition performance for underrepresented (rare) instruments in polyphonic music recordings.

The following models are implemented and compared:

- Baseline Random Forest
- Random Forest with Class Weighting
- Random Forest with Random Oversampling

## 🎯 Research Question
<b>How can class weighting and random oversampling improve macro-level performance and rare-instrument detection in an imbalanced multi-label classification setting?</b>

## 📊 Dataset
This project uses:

<a href="https://zenodo.org/records/1432913">OpenMIC-2018</a>

OpenMIC-2018 is a large-scale, openly available dataset for multi-label instrument recognition in polyphonic audio.

<b>Key Characteristics</b>
- 20 instrument classes
- Polyphonic audio (multiple instruments per clip)
- Real-world imbalance: frequent instruments dominate the dataset
- Input representation: pre-extracted VGGish embeddings

Due to dataset size and licensing considerations, data files are <b>not included in this repository.</b>

## 🧠 Methodology
<b>1️⃣ Baseline Model</b>
- One-vs-Rest classification
- 20 independent Random Forest classifiers
- Decision threshold = 0.5
- Evaluation on fixed train/test split

<b>2️⃣ Handling Class Imbalance</b>
Two strategies were explored:

<b>✔ Class Weighting</b>

- class_weight="balanced" in RandomForestClassifier
- Increases importance of minority classes during training

<b>✔ Random Oversampling</b>

- RandomOverSampler from imbalanced-learn
- Replicates minority class samples in training data
- Applied only to training split

## 📈 Evaluation Metrics

Performance is evaluated using:

- Micro F1 Score
- Macro F1 Score
- Macro ROC-AUC

## 🏗️ Project Structure
```
MIR-OpenMic-2018/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── model.py
│   └── train.py
│
└── results/
    └── results_summary.md
```
<b>File Responsibilities</b>

- data.py → Feature loading, label processing, alignment
- model.py → Model definitions (baseline, weighted, oversampled)
- train.py → Training pipeline, evaluation, CLI interface

## ⚙️ Installation
```
git clone https://github.com/Dhvani00/MIR-OpenMic-2018.git
cd MIR-OpenMic-2018
pip install -r requirements.txt
```

## ▶️ Usage
<b>Baseline</b>
```
python src/train.py \
    --features path/to/openmic-2018.npz \
    --labels path/to/openmic-2018-aggregated-labels.csv \
    --strategy baseline
```
<b>Class Weighting</b>
```
python src/train.py \
    --features path/to/openmic-2018.npz \
    --labels path/to/openmic-2018-aggregated-labels.csv \
    --strategy weighted
```
<b>Oversampling</b>
```
python src/train.py \
    --features path/to/openmic-2018.npz \
    --labels path/to/openmic-2018-aggregated-labels.csv \
    --strategy oversample
```

## 📊 Experimental Results
| Model                | Micro F1   | Macro F1   | ROC-AUC    |
| -------------------- | ---------- | ---------- | ---------- |
| Baseline RF          | 0.5423     | 0.4725     | 0.9546     |
| RF + Class Weighting | 0.5482     | 0.4873     | 0.9677     |
| RF + Oversampling    | **0.6072** | **0.5602** | **0.9689** |

**Key Observations**
- Oversampling yields the largest improvement in Macro F1.
- Improvements are most significant for rare instruments.
- Micro F1 increases moderately, as it is dominated by frequent classes.

This confirms that imbalance-aware training strategies improve minority class recognition.

## 📂 Data
Data files are not included in this repository.

To reproduce experiments:

1. Download the OpenMIC-2018 dataset.
2. Obtain pre-extracted VGGish embeddings.
3. Provide paths to:
```
--features path/to/openmic-2018.npz \
--labels path/to/openmic-2018-aggregated-labels.csv \
```

## 📚 Reference
Humphrey, Eric J., Simon Durand, and Brian McFee.
“OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition.”
ISMIR 2018.

## 👩‍💻 Author
Dhvani Panseriya \
MSc Artificial Intelligence\

Brandenburg Technical University Cottbus-Senftenberg
