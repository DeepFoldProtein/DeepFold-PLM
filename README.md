# DeepFold-PLM: Accelerating Protein Structure Prediction via Efficient Homology Search Using Protein Language Models

[![Status](https://img.shields.io/badge/Status-Submitted-orange.svg)](https://github.com/your-repo/DeepFold-PLM)
[![Website](https://img.shields.io/badge/Website-Live-brightgreen.svg)](https://df-plm.deepfold.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🧬 Overview

DeepFold-PLM revolutionizes protein structure prediction by integrating advanced protein language models with vector embedding databases to achieve ultra-fast MSA construction and enhanced structure prediction capabilities.

### Key Features

- **⚡ 47x Faster MSA Generation**: Dramatically accelerated multiple sequence alignment construction
- **🎯 Accurate Predictions**: Maintains prediction accuracy comparable to AlphaFold
- **🔗 Multimeric Complexes**: Extended modeling capabilities for protein complexes
- **🚀 Scalable Implementation**: PyTorch-based framework for large-scale predictions
- **🌐 User-Friendly Interface**: Real-time analysis through web service
- **📈 Enhanced Diversity**: Increased sequence diversity for better coevolutionary information

## 🖥️ Website

Explore now: [https://df-plm.deepfold.org/](https://df-plm.deepfold.org/)

## 📝 Abstract

Protein structure prediction has been revolutionized and generalized with the advent of cutting-edge AI methods such as AlphaFold, but reliance on computationally intensive multiple sequence alignments (MSA) remains a major limitation. We introduce DeepFold-PLM, a novel framework that integrates advanced protein language models with vector embedding databases to enhance ultra-fast MSA construction, remote homology detection, and protein structure prediction.

DeepFold-PLM utilizes high-dimensional embeddings and contrastive learning, significantly accelerate MSA generation, achieving 47 times faster than standard methods, while maintaining prediction accuracy comparable to AlphaFold. In addition, it enhances structure prediction by extending modeling capabilities to multimeric protein complexes, provides a scalable PyTorch-based implementation for efficient large-scale prediction, and offers a user-friendly web service for real-time analysis.

Our method also effectively increases sequence diversity, enriching coevolutionary information critical for accurate structure prediction. DeepFold-PLM thus represents a versatile and practical resource that enables high-throughput applications in computational structural biology.

## 👥 Authors

**Minsoo Kim**¹* • **Hanjin Bae**¹* • **Gyeongpil Jo**¹ • **Kunwoo Kim**¹ • **Sung Jong Lee**² • **Jejoong Yoo**¹† • **Keehyoung Joo**³†

¹ *Department of Physics, Sungkyunkwan University, Suwon, Korea*  
² *Basic Science Research Institute, Changwon National University, Changwon 51140, Korea*  
³ *Center for Advanced Computation, Korea Institute for Advanced Study, Seoul, Korea*

*\* These authors contributed equally to this work*  
*† Corresponding authors: jejoong@skku.edu, newton@kias.re.kr*

## 🚀 Quick Start

### plmMSA

See [plmMSA/README.md](plmMSA/README.md) for more information.

### DeepFold PyTorch

See [DeepFold/README.md](https://github.com/DeepFoldProtein/DeepFold/blob/main/README.md) for more information.

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

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📞 Contact

For questions and support, please contact:
- Jejoong Yoo: jejoong@skku.edu
- Keehyoung Joo: newton@kias.re.kr

