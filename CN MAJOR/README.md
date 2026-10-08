# CRC-Based Error Detection System with ML Error Impact Predictor

## M.Tech Level Implementation — Computer Networks

### Overview

A comprehensive implementation of a CRC-based error detection and correction system
with machine learning-based error impact prediction. The system includes:

- **CRC Encoder/Decoder** supporting CRC-16 (CCITT, IBM) and CRC-32 polynomials
- **Noise Injection Module** simulating single-bit, multi-bit, burst, and random errors
- **Advanced Decoding** using Belief Propagation and ORBGRAND algorithms
- **ML Pipeline** with Random Forest and XGBoost for predicting undetected errors
- **Comprehensive Evaluation** with 8 publication-quality visualizations

### Directory Structure

```
CN MAJOR/
├── config.py                        # Global configuration
├── main.py                          # Full pipeline orchestrator
├── run_visualization.py             # Regenerate plots
├── CRC_Complete_Colab_Notebook.py   # 🌟 Complete Google Colab code
├── requirements.txt
│
├── core/                            # CRC encoder, decoder, polynomials
├── channel/                         # Noise injection, channel models
├── decoding/                        # BP + ORBGRAND decoders
├── ml/                              # Dataset generation, ML training
├── evaluation/                      # Metrics, checksum comparison
├── visualization/                   # All plot modules
├── tests/                           # Unit tests
└── outputs/                         # Generated models, datasets, plots
```

### Quick Start

#### Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com)
2. Copy the contents of `CRC_Complete_Colab_Notebook.py`
3. Paste into a new notebook cell and run

#### Local Execution
```bash
pip install -r requirements.txt
python main.py
```

### Mathematical Foundation

- **M(x)**: Message polynomial
- **G(x)**: Generator polynomial of degree r
- **T(x) = x^r · M(x) + R(x)**, where R(x) = (x^r · M(x)) mod G(x)
- **Zero-Remainder Decision**: Accept frame iff T'(x) mod G(x) == 0

### Key Results

- CRC detects **ALL** burst errors of length ≤ r (polynomial degree)
- For bursts > r: undetected probability ≈ 2^(-r)
- CRC-16 significantly outperforms Internet Checksum across all error types
- ML models achieve high accuracy in predicting aliasing events
