#  Metric learning and fitness of LLM response**  
### **DA5401 – Advanced Data Analytics | IIT Madras**  
### **Project by: Puneet Mishra (DA25C016)**

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12.3-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-red.svg" />
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" />
  <img src="https://img.shields.io/badge/Metric_Learning-SCML-purple.svg" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" />
</div>

---

## **Project Overview**

This repository contains the full implementation and report of a **Sparse Compositional Metric Learning (SCML)** system designed to predict **fitness scores for multilingual prompt–response pairs**.  
The task mimics real LLM evaluation pipelines, where the goal is to determine how well a model responded to a given prompt under specific **semantic criteria (“metrics”)**.  

SCML is used to learn **local, adaptive metrics** in a high-dimensional embedding space, enabling the model to capture nuanced variations across languages, prompts, and evaluation criteria.

---


---

## **Core Idea**

LLM responses are evaluated on metrics like *bias detection*, *coherence*, *factuality*, etc.  
These metric definitions and the P-R pair are embedded using:
- **Transformer encoder** (MurIL / MiniLM) for contextual encoding  
- **Precomputed metric embeddings** for semantic alignment  

SCML combines these embeddings to learn a **local Mahalanobis-style metric** that varies across data points using:
- Non-negative learned weight functions  
- Rank-one compositional basis  
- Smooth, nonlinear transformations  

This produces an adaptive distance function that maps high-dimensional text embeddings to fitness scores.

---

##  **Model Pipeline**

### **1. Text + Metric Embedding Extraction**
- Context = `user_prompt + system_prompt + response`
- Encoded using HuggingFace transformer  
- Metric names mapped to provided embeddings  
- Combined into a single feature vector  

### **2. Dimensionality Reduction**
- PCA retains 95% variance  
- Stabilises training and controls dimensional explosion  

### **3. SCML Feature Construction**
- Basis vectors chosen in PCA space  
- Weights computed through learned softplus-activated layers  
- SCML feature: `φ(x) = sqrt(w(x)) * (xᵀB)`

### **4. Regression Head**
- MLP predicts scores 0–10  
- Early stopping + RMSE-based validation  

---

##  **Results Summary**

- The SCML-based regressor learns stable structure in multilingual P-R data.
- Local compositional metrics provide finer granularity than global linear models.
- Predictions generalise well under strong regularisation and PCA filtering.
- Shows viability of metric learning as an evaluation backbone for LLM scoring tasks.

---

##  **Included Report**

Full project report with methodology, theory, code explanations, and results:  
👉 **`report_da5401.pdf`**

---




