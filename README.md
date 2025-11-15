# CASC - Cross-modal Alignment with Synthetic Caption

Demo implementation of the paper "Cross-modal alignment with synthetic caption for text-based person search"

## Overview

CASC is a novel framework for text-based person search that achieves cross-modal alignment through synthetic captions. The model includes:

- **GAS (Granularity Awareness Sensor)**: Adaptive masking strategy for fine-grained feature selection
- **CCL (Conditional Contrastive Learning)**: Dynamic weight adjustment based on hard negative similarities
- **Multi-modal Architecture**: Image encoder, text encoder, and caption decoder

## Key Features

1. **Adaptive Feature Selection**: GAS selects discriminative features without additional parameters
2. **Synthetic Caption Generation**: Enhances multimodal alignment with generated captions
3. **Conditional Learning**: CCL mitigates impact of noisy captions dynamically
4. **State-of-the-art Performance**: Achieves top results on CUHK-PEDES, ICFG-PEDES, RSTPReid

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py
```

## Model Architecture

```
Input: Image + Text Description
↓
Image Encoder (ViT) → GAS → Masked Features
Text Encoder (BERT) → Text Features
↓
Caption Decoder → Synthetic Caption
↓
CCL (Image-Text-Caption Alignment)
↓
Output: Similarity Scores
```

## Results

| Dataset | Rank@1 | Rank@5 | Rank@10 |
|---------|--------|--------|---------|
| CUHK-PEDES | 77.71% | 90.57% | 94.22% |
| ICFG-PEDES | 69.85% | 84.03% | 86.79% |
| RSTPReid | 70.80% | 87.35% | 93.25% |

## Citation

```bibtex
@article{zhao2025casc,
  title={Cross-modal alignment with synthetic caption for text-based person search},
  author={Zhao, Weichen and Lu, Yuxing and Liu, Zhiyuan and Yang, Yuan and Jiao, Ge},
  journal={International Journal of Multimedia Information Retrieval},
  volume={14},
  number={11},
  year={2025}
}
```

## License

MIT License
"""

print("=" * 60)
print("CASC Demo Project Successfully Generated!")
print("=" * 60)
print("\\n📁 Project Structure:")
print("├── config.py - Configuration settings")
print("├── models/")
print("│   ├── image_encoder.py - Vision Transformer")
print("│   ├── text_encoder.py - BERT encoder")
print("│   ├── caption_decoder.py - Caption generation")
print("│   └── casc_model.py - Complete CASC model")
print("├── utils/")
print("│   ├── gas.py - Granularity Awareness Sensor")
print("│   └── ccl.py - Conditional Contrastive Learning")
print("├── train.py - Training script")
print("├── inference.py - Inference script")
print("└── README.md - Documentation")
print("\\n🔑 Key Components:")
print("1. GAS: Adaptive masking without extra parameters")
print("2. CCL: Dynamic weight adjustment for noisy captions")
print("3. Multi-modal: Image + Text + Caption alignment")
print("\\n🚀 To get started:")
print("1. Install requirements: pip install -r requirements.txt")
print("2. Prepare your dataset (CUHK-PEDES format)")
print("3. Run training: python train.py")
print("4. Run inference: python inference.py")
