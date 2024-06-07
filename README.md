# SUSTech CS310 Final Project - Enhancing Multitask BERT with LoRA, RMSNorm, and SwiGLU
#### Author: Kenneth Zhang - Weitao Yan - Lam Nguyen

This repository contains the code for the final project of the SUSTech CS310 course, which inherits the skeleton code and base project from the Stanford CS 224N course (https://github.com/gpoesia/minbert-default-final-project). The project focuses on enhancing the multitask learning capabilities of the BERT model. The initial project aimed to implement key components of the BERT model and use its embeddings for three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. Building on this foundation, we experimented with various enhancements, including LoRA, RMSNorm, and SwiGLU, to improve the model's performance.

## Project Overview

In this project, we explored the following:

- **Baseline BERT Model**: Implementation and analysis of the basic BERT model's performance on multitask learning.
- **Enhancements**:
  - **LoRA (Low-Rank Adaptation)**: Adding LoRA to BERT and evaluating its impact on performance.
  - **RMSNorm**: Replacing LayerNorm with RMSNorm and assessing its effectiveness.
  - **SwiGLU**: Replacing GELU activation with SwiGLU and analyzing the results.
- **Combined Enhancements**: Investigating the combined effect of LoRA, RMSNorm, and SwiGLU on BERT's multitask performance.

## Key Findings

- The **baseline BERT model** provided a stable and consistent performance across all tasks, serving as a reliable benchmark.
- **LoRA** introduced significant fluctuations and inefficiencies, failing to consistently improve performance.
- **RMSNorm** demonstrated superior fine-tuning efficiency, achieving the lowest training loss and highest accuracy metrics across tasks.
- **SwiGLU** showed moderate improvements but was less effective than RMSNorm.
- The **combined model** of LoRA, RMSNorm, and SwiGLU performed the poorest, indicating a lack of synergy among these enhancements.

## Future Work

Based on our findings, future work will focus on:

- Enhancing dataset quality and eliminating biases in order, size, and difficulty.
- Testing alternative Parameter-Efficient Fine-Tuning (PEFT) techniques: Adapters, Prefix-Tuning, P-Tuning, DiffPruning, etc.
- Investigating modern architectural integrations such as KV-caching, flash attention, multi-query attention, and grouped query attention.

## Setup Instructions

1. Follow `setup.sh` to properly set up a conda environment and install dependencies.
2. Refer to [STRUCTURE.md](./STRUCTURE.md) for a detailed description of the code structure, including parts to implement.
3. Use only libraries installed by `setup.sh`. External libraries providing other pre-trained models or embeddings (e.g., `transformers`) are not allowed.

## Handout

Please refer to the handout for a thorough description of the project and its components. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf).

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html), created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
