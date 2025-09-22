# BioAlign-QLoRA: Quantifying the Structural Alignment of LLM Embeddings with a Biomedical Knowledge Graph Following QLoRA Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Type-Research-green.svg)](https://github.com/fnziad/BioAlign-QLoRA)
[![BRAC University](https://img.shields.io/badge/Institution-BRAC%20University-blue.svg)](https://www.bracu.ac.bd/)

> **Academic Research Project - CSE443 Bioinformatics Coursework**  
> **BRAC University, Department of Computer Science and Engineering**

## 🎯 Overview

This project addresses the fundamental challenge of transforming generalist Large Language Models (LLMs) into specialized biomedical experts through high-efficiency fine-tuning. We investigate whether targeted QLoRA (Quantized Low-Rank Adaptation) fine-tuning can induce a deep structural reorganization within a model's internal representations, causing its understanding of biological concepts to align more closely with real-world knowledge graphs.

**Key Innovation**: We introduce a novel "Knowledge Graph Separation" score that quantifies the geometric alignment between an LLM's embedding space and biological knowledge structures, providing empirical evidence of successful knowledge transfer.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Academic Context](#academic-context)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## ✨ Features

- **Novel Evaluation Framework**: Introduction of "Knowledge Graph Separation" score for quantifying embedding alignment
- **Structural Reorganization Analysis**: Empirical evidence of profound embedding space transformation (>126% improvement)
- **Multi-Model Comparative Study**: Fine-tuning of Llama-3 8B, Mistral 7B, and Phi-3 Mini 3.8B
- **High-Efficiency Training**: QLoRA implementation enabling fine-tuning on consumer-grade hardware
- **Benchmark Outperformance**: Consistently outperformed pre-trained BioMistral-7B expert model
- **Curated Dataset**: 68,444 gene-disease associations from Comparative Toxicogenomics Database (CTD)
- **Accessibility Focus**: Lightweight models suitable for resource-constrained environments

## 🏗️ Project Structure

```
BioAlign-QLoRA/
├── adapters/                      # Fine-tuned model adapters
│   ├── fine_tuned_llama3_gda/     # Llama3 QLoRA adapters
│   ├── fine_tuned_mistral7b_gda/  # Mistral 7B QLoRA adapters
│   └── fine_tuned_phi3_gda/       # Phi-3 QLoRA adapters
├── codes/                         # Core implementation
│   ├── analysis.ipynb             # Comprehensive analysis notebook
│   ├── dataproc.ipynb            # Data preprocessing pipeline
│   ├── eda.ipynb                 # Exploratory data analysis
│   ├── finetuneLlama.ipynb       # Llama3 fine-tuning implementation
│   ├── finetuneMistral7b.ipynb   # Mistral 7B fine-tuning implementation
│   └── finetunePhi3.ipynb        # Phi-3 fine-tuning implementation
├── data/                          # Dataset and processed files
│   ├── ctd_processed_dataset.csv  # Main processed dataset
│   ├── raw&processed/             # Raw and processed data files
│   └── visuals/                   # Generated visualizations
├── paper/                         # Research documentation
│   └── BioAlign_Report_LLM_BioKG_QLoRA.pdf
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Conda or pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fnziad/BioAlign-QLoRA.git
   cd BioAlign-QLoRA
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n bioalign-qlora python=3.8
   conda activate bioalign-qlora
   ```

3. **Install dependencies:**
   ```bash
   pip install torch transformers datasets peft accelerate
   pip install pandas numpy matplotlib seaborn scikit-learn
   pip install jupyter notebook
   ```

4. **Download required models:**
   The fine-tuned adapters are included in the `adapters/` directory. Base models will be downloaded automatically when running the notebooks.

## 💻 Usage

### Data Processing
```bash
jupyter notebook codes/dataproc.ipynb
```

### Exploratory Data Analysis
```bash
jupyter notebook codes/eda.ipynb
```

### Model Fine-tuning

#### Llama3 Fine-tuning
```bash
jupyter notebook codes/finetuneLlama.ipynb
```

#### Mistral 7B Fine-tuning
```bash
jupyter notebook codes/finetuneMistral7b.ipynb
```

#### Phi-3 Fine-tuning
```bash
jupyter notebook codes/finetunePhi3.ipynb
```

### Analysis and Evaluation
```bash
jupyter notebook codes/analysis.ipynb
```

## 🤖 Models

This project implements QLoRA fine-tuning for three state-of-the-art language models:

| Model | Parameters | Zero-Shot Accuracy | KG Separation Improvement | Use Case |
|-------|------------|-------------------|---------------------------|----------|
| **Llama-3 8B** | 8B | 81.0% (+57.0%) | +49.0% | High-performance biomedical understanding |
| **Mistral 7B** | 7B | 83.8% (+41.1%) | -38.0%* | Efficient inference with architectural innovations |
| **Phi-3 Mini** | 3.8B | 68.8% (+16.2%) | +126.3% | Lightweight deployment for resource constraints |

*Mistral's negative KG separation change indicates optimization prioritized classification accuracy over geometric purity.

### QLoRA Configuration
- **Rank (r)**: 16
- **Alpha**: 32  
- **Dropout**: 0.0 (optimized for Unsloth)
- **Max Steps**: 2,000
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit

## 📊 Dataset

The project utilizes a curated dataset from the **Comparative Toxicogenomics Database (CTD)**:

- **Source**: CTD Curated Gene-Disease Associations (high-confidence only)
- **Total Size**: 68,444 gene-disease pairs (perfectly balanced)
- **Positive Examples**: 34,222 evidence-backed associations
- **Negative Examples**: 34,222 synthetically generated pairs
- **Format**: Simple template - "{GeneSymbol} is associated with {DiseaseName}"

### Data Statistics
- **Unique Genes**: 9,111
- **Unique Diseases**: 5,858
- **Evidence Types**: Marker/mechanism, therapeutic (direct evidence only)
- **Average Text Length**: 52 characters
- **Training Split**: 80% (54,755) / Test: 20% (13,689)
- **Evidence Distribution**: Average 1.09 PubMed IDs per positive association

## 📈 Results

### Model Performance

| Model | Zero-Shot Accuracy | KG Separation Score | Probe Accuracy | Benchmark Comparison |
|-------|-------------------|-------------------|----------------|---------------------|
| **Mistral-QLoRA** | 83.8% | 0.1008 (-38.0%) | 13.6% | **Outperformed BioMistral** |
| **Llama-3-QLoRA** | 81.0% | 0.0578 (+49.0%) | 7.5% | **Outperformed BioMistral** |
| **Phi-3-QLoRA** | 68.8% | 0.0349 (+126.3%) | 5.9% | **Outperformed BioMistral** |
| BioMistral-7B | 50.9% | N/A | N/A | Baseline |

### Key Findings
- **Structural Reorganization**: All models showed geometric transformation from "chaotic clouds" to distinct clusters
- **Benchmark Superior**: All fine-tuned models outperformed the pre-trained BioMistral expert
- **Efficiency Achievement**: Phi-3 Mini demonstrated remarkable 126% KG separation improvement despite 3.8B parameters
- **Training Efficiency**: Significant improvements achieved in just 2,000 steps (~29% of one epoch)
- **Accessibility Validation**: Lightweight models proven viable for resource-constrained deployment

## 🎓 Academic Context

**Course**: CSE443 - Bioinformatics  
**Institution**: BRAC University  
**Department**: Computer Science and Engineering  
**Semester**: Summer 2025  
**Project Type**: Group Research Project

### Research Objectives
- Investigate whether targeted fine-tuning can induce structural reorganization in LLM embeddings
- Develop quantitative metrics for measuring knowledge graph alignment in neural representations
- Create computationally efficient methods for specializing generalist models
- Validate accessibility of advanced AI tools for resource-constrained environments
- Demonstrate democratization pathway for biomedical AI applications

## 👥 Authors

**Fahad Nadim Ziad** - *First Author & Project Lead*
- **Student ID**: 24341216
- **Email**: fahad.nadim.ziad@g.bracu.ac.bd
- **GitHub**: [@fnziad](https://github.com/fnziad)
- **Role**: Project conception, model implementation, experimental design, Knowledge Graph Separation framework

**Aalavi Mahin Khan** - *Co-Author*
- **Student ID**: 22301789
- **Department**: Computer Science and Engineering, BRAC University
- **Role**: Data curation and preprocessing, exploratory data analysis, evaluation framework

**Khaled Saifullah Karim** - *Co-Author*
- **Student ID**: 24341262
- **Department**: Computer Science and Engineering, BRAC University
- **GitHub**: [@KsKarim7](https://github.com/KsKarim7)
- **Role**: Dataset construction, model fine-tuning, performance analysis, visualization

> **Note**: Team members may have additional repository access and resources. Please refer to individual repositories for supplementary materials.

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{ziad2025bioalign,
  title={Quantifying the Structural Alignment of LLM Embeddings with a Biomedical Knowledge Graph Following QLoRA Fine-Tuning},
  author={Ziad, Fahad Nadim and Khan, Aalavi Mahin and Karim, Khaled Saifullah},
  year={2025},
  institution={BRAC University},
  note={CSE443 Bioinformatics Course Project, Summer 2025},
  url={https://github.com/fnziad/BioAlign-QLoRA}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BRAC University** Department of Computer Science and Engineering
- **CSE443 Bioinformatics Course** instructors and teaching assistants
- **Comparative Toxicogenomics Database (CTD)** for providing the curated gene-disease associations
- **Unsloth** for high-efficiency QLoRA implementation enabling accessible fine-tuning
- **Hugging Face** for transformer models and infrastructure
- **Meta AI**, **Mistral AI**, and **Microsoft** for open-source model contributions
- **Open-source research community** for foundational tools and methodologies

## 📚 References

- **QLoRA Paper**: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs
- **CTD Database**: Davis, A.P., et al. (2023). Comparative Toxicogenomics Database
- **Transformers Library**: Wolf, T., et al. (2020). Transformers: State-of-the-art Natural Language Processing

---

**Academic Integrity Statement**: This work represents original research conducted as part of the CSE443 Bioinformatics coursework. All team members contributed to different aspects of the project, and appropriate attribution has been provided for external resources and datasets used.

**Contact**: For questions about this research project, please contact the first author or refer to the detailed research report in the `paper/` directory.