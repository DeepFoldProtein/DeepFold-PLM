# DeepFold-PLM: Accelerating Protein Structure Prediction via Efficient Homology Search Using Protein Language Models

[![Status](https://img.shields.io/badge/Status-Submitted-orange.svg)](https://github.com/your-repo/DeepFold-PLM)
[![Website](https://img.shields.io/badge/Website-Live-brightgreen.svg)](https://df-plm.deepfold.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🧬 Overview

DeepFold-PLM accelerates protein structure prediction by integrating advanced protein language models with vector embedding databases to achieve ultra-fast MSA construction and enhanced structure prediction capabilities.

![Architecture of DeepFold-PLM pipeline](images/main.png)

**Architecture of DeepFold-PLM pipeline.**
**(a)** The plmMSA module constructs MSA using PLM-based alignment. Protein sequences are transformed into dense vector representations through pre-trained PLMs, enabling rapid retrieval of homologous sequences. PLMAlign then constructs MSAs using vector database embeddings, precomputed ProtT5 embeddings, and query sequence embeddings.
**(b)** The monomer structure prediction module integrates plmMSA-derived homologous structures and MSA-derived constraints to predict 3D protein structures.
**(c)** The complex structure prediction module extends the approach to multimeric protein through template search and coevolutionary analysis.

### Key Features

- **⚡ 47x Faster MSA Generation**: Dramatically accelerated multiple sequence alignment construction
- **🎯 Accurate Predictions**: Maintains prediction accuracy comparable to AlphaFold
- **🔗 Multimeric Complexes**: Extended modeling capabilities for protein complexes
- **🚀 Scalable Implementation**: PyTorch-based framework for large-scale predictions
- **🌐 User-Friendly Interface**: Real-time analysis through web service
- **📈 Enhanced Diversity**: Increased sequence diversity for better coevolutionary information

## 🖥️ Website

Explore now: [https://df-plm.deepfold.org/](https://df-plm.deepfold.org/)

## 🚀 Quick Start

### plmMSA

See [plmMSA](plmMSA) for more information.

### DeepFold PyTorch

See [DeepFold](https://github.com/DeepFoldProtein/DeepFold/blob/main) for more information.

## 📚 Citation

If you use DeepFold-PLM in your research, please cite our paper:

```bibtex
@article{kim2025deepfold,
  title={DeepFold-PLM: Accelerating Protein Structure Prediction via Efficient Homology Search Using Protein Language Models},
  author={Kim, Minsoo and Bae, Hanjin and Jo, Gyeongpil and Kim, Kunwoo and Lee, Sung Jong and Yoo, Jejoong and Joo, Keehyoung},
  journal={Submitted},
  year={2025},
  url={https://df-plm.deepfold.org/}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
