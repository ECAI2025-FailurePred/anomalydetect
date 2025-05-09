# Failure Prediction in Electrolyzers with Interpretable Image-Based Deep Learning and Unsupervised Domain Adaptation

## 📄 Abstract

Traditional failure prediction approaches, including statistical methods, machine learning, and deep learning, face notable limitations when applied to real data and require specific tuning: critical information is often embedded within subtle variations in signal noise, datasets are large and acquired with different acquisition methods resulting in domain shifts.

In this paper we propose a scalable deep learning framework for classifications of noisy data, transforming high-dimensional univariate time-series into structured multi-channel 2D representations suitable for convolutional neural networks. Central to our approach is a Variational AutoEncoder (VAE) that learns latent representations for effective anomaly detection. To enhance generalization across diverse datasets, we integrate an Unsupervised Domain Adaptation (UDA) mechanism, enabling robust model transferability.

To prove the efficiency of our approach in learning meaningful latent representations, we applied our model architecture to synthetic data representing signals with periodic and noisy components.
The validity of the methodology is confirmed by training our model on industrial electrolyzers data with highly noisy features for failure classification tasks.

To address the interpretability challenge, we incorporate a counterfactual explanation module that identifies minimal perturbations in input signals required to alter prediction outcomes, facilitating transparent failure diagnosis. Evaluations on multiple industrial datasets demonstrate a $25\%$ improvement in accuracy over direct supervised training approaches, achieving $89.91\%$ accuracy with only $11.07\%$ Symmetric Mean Absolute Percentage Error (sMAPE). Compared to baseline machine learning models, our method reduces reconstruction error by $50\%$, effectively capturing underlying failure patterns.

By combining multi-modal time-series representations, robust domain adaptation, and interpretable deep learning, our framework offers a generalizable and scalable solution for predictive maintenance in industrial systems, with broader applicability to high-variance time-series anomaly detection tasks across diverse domains.

---


## 🧠 Key Contributions
-  Time-series to image transformation (GASF, GADF, MTF)
-  Probabilistic signal modeling using VAE with Student-t Mixture Decoder
-  UDA for cross-domain generalization
-  Counterfactual interpretability
-  Synthetic dataset mimicking real-world electrolyzer signal characteristics


---

## 🔥 Pipeline Overview

<p align="center">
  <img src="Images/pipeline.PNG" width="80%">
</p>

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ECAI2025-FailurePred/anomaly-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

We provide a Jupyter Notebook that demonstrates the full pipeline:
- Data preprocessing
- Noise evolution signal generation
- Image transformations (GASF, GADF, MTF)
- Training the VAE model
- Applying UDA for domain adaptation
- Generating counterfactual explanations

**Note:**  
The data provided is synthetic and was generated to replicate the characteristics of real-world electrolyzer voltage signals (e.g., noise patterns, failure symptoms) as described in the paper.  
The code structure and methodology remain identical to those used on real industrial datasets.

---

## 📁 Project Structure

```bash
.
├── Domain_Informed_Signal_Reconstruction.ipynb       # Domain adaptation with counterfactuals
├── synthetic_signal_vae_training_and_visualization.ipynb  # Full VAE training demo
├── model.py                                          # VAE and VAE1D definitions
├── synthetic_signal_dataset.py                       # Signal generator + transformations
├── utils.py                                          # Logging, training, and utilities
└── requirements.txt                                  # Required dependencies
 

```

---

## 🧪 Testing

You can test the complete pipeline using the synthetic dataset provided.  
The notebooks guide ou step-by-step through:
- Data loading
- Preprocessing
- Model training
- Evaluation

---
