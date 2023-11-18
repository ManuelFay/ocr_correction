# OCR Correction - Old french manuscripts from Bibliothèque Nationale de France

## Introduction

This repository contains the code for the OCR correction of old french manuscripts from the Bibliothèque Nationale de France. 

The BNF containes hundreds of thousands of old french manuscripts.
The OCR of these documents is not perfect and contains many errors. 
The goal of this project is to correct these errors using a neural network.

## Data

The data used for this project is the [Gallica dataset](https://api.bnf.fr/fr/node/222).

## The plan

### 1. Data preparation

- Collect an aligned dataset of french OCR transcriptions and their corrected version
  - Select a few books from the Gallica dataset
  - ChatGPT API to gather the dataset in its corrected version
  - Chunking / Sliding window to split the books into smaller chunks

### 2. Training

- Train a decoder on concatenated sequences: 
  - The input is the concatenation of the OCR and the corrected version
  - CE loss on the whole sequence

We need a decoder that speaks good French, and that is large enough to be performant.

## Installation

To install the project, you need to clone the repository and install the dependencies.

```bash
pip install -r requirements.txt
```
