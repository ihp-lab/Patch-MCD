# Patch-MCD
Source codes for our IEEE FG'21 paper: Self-Supervised Patch Localization for Cross-Domain Facial Action Unit Detection. Yufeng Yin, Liupei Lu, Yizhen Wu, and Mohammad Soleymani.

## Training
Train base model with BP4D/BP4D+:
```
python train_bl.py --source BP4D/BP4D+
```

Train domain adaptation models with BP4D/BP4D+ and DISFA/GFT:
```
python train_dann.py/train_dan.py/train_cdan.py/train_jan.py/train_mdd.py/train_mcd.py/train_patch_mcd.py --source BP4D/BP4D+ --target DISFA/GFT
```
