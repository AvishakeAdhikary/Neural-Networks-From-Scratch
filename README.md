# Neural Networks from Scratch

![Neural Networks from Scratch Banner][bannerdefinition]

This repository contains a sequence of Jupyter notebooks, each representing a step in understanding and implementing neural networks from scratch. The primary focus is on understanding the underlying concepts and implementing them without relying on external libraries. The lecture series takes you from basic concepts to advanced architectures, demonstrating the conversion of traditional programming approaches to modern neural network techniques.

## Lecture Series Overview

1. **Neural Networks with Derivatives:** In this introductory notebook, we build a complete neural network from scratch, focusing on understanding derivatives and their role in neural network operations.
2. **NameWeave:** Beginning with a single-layer bigram model, we gradually transition from traditional machine learning to neural network approaches for name generation.
3. **NameWeave - Multi Layer Perceptron:** We expand the NameWeave model into a multi-layer perceptron (MLP), increasing the complexity of the model.
4. **NameWeave (MLP) - Activations, Gradients & Batch Normalization:** Continuing from the previous notebook, we enhance the MLP model with activation functions, gradient handling, and batch normalization techniques.
5. **NameWeave - Manual Back Propagation:** This notebook breaks down the MLP model into atomic pieces of code, emphasizing the importance of understanding backpropagation.
6. **NameWeave - WaveNet:** Inspired by WaveNet architecture, we modify the previous model to resemble convolutional neural networks (CNNs).
7. **GPT from Scratch:** Implementing all concepts learned previously, we introduce self-attention and decoder-only architecture to generate text, demonstrating the capability of modern architectures like Transformers. The notebook generates infinite Harry Potter-like text based on the provided dataset.


## Table of Contents
- [Introduction](#introduction)
- [Files](#files)
- [Usage](#usage)
- [Future Updates](#future-updates)
- [License](#license)

## Introduction

In this repository, I explore the implementation of neural networks from scratch. The primary goal is to deepen the understanding of neural network concepts and learn how to implement them without relying on external libraries.

## Files

- **GPT from Scratch.ipynb:** Jupyter notebook where a GPT model is built from scratch, generating text based on the Harry Potter dataset.
- **NameWeave (MLP) - Activations, Gradients & Batch Normalization.ipynb:** Jupyter notebook enhancing the NameWeave model with activations, gradients, and batch normalization.
- **NameWeave - Manual Back Propagation.ipynb:** Jupyter notebook demonstrating manual backpropagation through the NameWeave model.
- **NameWeave - Multi Layer Perceptron.ipynb:** Jupyter notebook expanding the NameWeave model into a Multi Layer Perceptron with multiple layers.
- **NameWeave - WaveNet.ipynb:** Jupyter notebook implementing a WaveNet-like architecture for name generation.
- **Neural Network with Derivatives.ipynb:** Jupyter notebook containing the implementation of a neural network with derivatives.
- **NNFS-GitHub Banner.gif:** Banner image for the repository.
- **README.md:** This file, providing an overview of the repository.
- **LICENSE:** The license information for this repository.
- **Datasets/:** Directory containing datasets used in the notebooks.
  - **Harry_Potter_Books.txt:** Dataset used in the `GPT from Scratch.ipynb` notebook.
  - **Indian_Names.txt:** Dataset used for all other notebooks.
- **ExplanationMedia/Images/:** Directory containing images used for explaining the notebooks.
- **ExplanationMedia/Videos/:** Directory containing videos used for explaining the notebooks.
- **GPT Scripts/:** Directory containing raw Python scripts created for building the GPT model in `GPT from Scratch.ipynb`.

## Usage

To explore the content of the lecture series, simply open the respective Jupyter notebook files using a compatible environment.

### Install Required Dependencies
You can use <a href="https://colab.research.google.com/">Google Colab</a> to view and run these files on the cloud.

OR

To view and run these files locally you need to run to install `Jupyter Notebook` via `PyPi` of Python:

Install Jupyter Notebook:
```bash
pip install notebook
```
Run Jupyter Notebook:
```bash
jupyter notebook
```

## Future Updates

Stay tuned for additional features, improvements, and possibly new lecture series exploring more advanced topics in neural networks and machine learning.

## License

This project is licensed under MIT - see the [LICENSE](LICENSE) file for details.


[bannerdefinition]: NNFS_GitHub_Banner.gif