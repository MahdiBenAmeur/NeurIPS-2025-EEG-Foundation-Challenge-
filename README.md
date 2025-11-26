# EEG Competition – Team Submission (13th Place)

This repository contains our work for the EEG Competition, where we achieved **13th place overall** with limited computational resources.

Final scores:

```
Overall:     0.97974  
Challenge 1: 0.94187  
Challenge 2: 0.99597
```

We tried several modeling paths during development and settled on the approach implemented here.

---
## Ablations & Experiments

Throughout development, we ran a large number of experiments exploring different training directions and variations.  
These include:

- **Masking strategies**  
  Tried multiple masking ratios and masking patterns.

- **Denoising objectives**  
  Added noise to inputs and trained the model to recover the clean signal.

- **Future-prediction tokens**  
  Added tokens predicting upcoming temporal slices.

- **Contrastive loss between CLS tokens**  
  Pushing CLS tokens from different windows toward or away from each other inside a batch.

- **Contrastive loss across subjects**  
  Using positive pairs from similar subjects and negative pairs from dissimilar ones.

- **4s → 2s representation alignment**  
  Training the model so that 2-second windows produce representations close to their 4-second counterparts.

- **Age-based pretraining**  
  Using age labels as an auxiliary pretraining target.

- **Task-classification heads**  
  Adding simple classification objectives during pretraining.

- **Used the DINOv3 training codebase**, with preprocessing and augmentation adjusted for EEG.

**We also tried chaining and mixing these approaches in different combinations to explore deeper effects.**

Only the final approach in this repository was used for the official submission.

## Approach (High-Level)

The final model is a **ViT-style masked autoencoder** adapted to EEG windows.

- Each EEG window is normalized per channel.  
- The signal is sliced into temporal patches (tokens).  
- Patches pass through temporal and channel projections.  
- Time, channel, and token positional embeddings are added.  
- A set of learned storage tokens is appended.  
- **35–45%** of tokens are randomly replaced with a mask token during training.  
- All tokens are fed into a transformer encoder.  
- A lightweight decoder reconstructs the masked patches.  
- Training compares reconstructed patches to the original unmasked ones.

This is the core architecture that produced our final results.

---

## Encoder Structure (Summary)

### `Vit_EEG_Embedding`
- Normalizes the window  
- Creates temporal slices  
- Applies time and channel projections  
- Adds positional embeddings  
- Replaces selected patches with a mask token  
- Appends storage tokens  
- Applies LayerNorm  
- Outputs token embeddings  

### `Vit_EEG_Encoder`
- Generates tokens using the embedding module  
- Processes tokens through transformer layers  
- Decodes tokens back to patch space  
- Supports both reconstruction and feature extraction  

---

## Repository Structure

```text
eeg_competition/
├── loging/
├── models_and_checkpoints/
├── notebooks/
│   ├── main.ipynb
│   └── restructuring_data.ipynb
├── submission/
│   ├── submission.py
│   ├── weights_challenge_1.pt
│   └── weights_challenge_2.pt
├── training_encoder/
│   ├── training_encoder.py
│   └── training_encoder_age_prediction.py
├── train_challange1/
│   ├── training_challange1_age.py
│   └── training_challange1_mae.py
├── train_challange2/
│   ├── age_bin_find.py
│   └── model.py
└── README.md
```

---

## Data

The project expects the **original competition release structure** without modification.  
All training scripts assume preprocessing is completed beforehand and that data paths are set manually inside the scripts.

---

## Dependencies

Used libraries (core):

```text
numpy
pandas
tqdm
matplotlib
mne
scikit-learn
torch
torchvision
```

Plus standard Python modules (`os`, `pathlib`, etc.).

---

## Notebooks

- **main.ipynb** – main experimentation notebook  
- **restructuring_data.ipynb** – preprocessing and data restructuring  

---

## Training

All training scripts require:
1. Preprocessed data  
2. Manual modification of internal data paths  


---

## Submission

contains the submission script to load both best performing models with 
Weights:
- `weights_challenge_1.pt`
- `weights_challenge_2.pt`
