Perfect 👍
Here’s your **full README** with the **updated Kaggle Notebooks section** (all links fixed so they open directly, no `/edit/run/...` errors).

---

# 🍎🍌 Fruit Classification Project – CSE366 Term Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

**State-of-the-Art Deep Learning for Fruit Classification with Vision Transformers, CNNs & XAI**

[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=for-the-badge\&logo=kaggle\&logoColor=white)](#-kaggle-notebooks)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://your-streamlit-app-url.com)

</div>  

---

## 🌟 Introduction

Welcome to the **Fruit Classification Project**! 🍇🥭
This project explores **modern deep learning techniques** to classify fruits into 5 categories: **Apple, Banana, Grape, Mango, Orange**. We employ **Vision Transformers (ViT)**, **CNNs**, and **XAI methods** for accurate, explainable, and deployable solutions.

Perfect for **students, researchers, and AI enthusiasts** who want a complete pipeline — from **model training to deployment**.

---

## ✨ Key Features

| Feature                       | Description                                                     | Benefit                             |
| ----------------------------- | --------------------------------------------------------------- | ----------------------------------- |
| **🤖 Multiple Architectures** | ViT, DeiT, VGG16, ConvNeXt, EfficientNet, DenseNet, Custom CNNs | Compare classic & modern models     |
| **🎯 High Accuracy**          | Up to **95%** test accuracy                                     | Reliable for real-world use         |
| **🔍 Explainable AI**         | Grad-CAM, saliency maps, attention visualization                | Understand *why* the model predicts |
| **🚀 GPU Optimized**          | CUDA + Mixed Precision training                                 | Fast & efficient                    |
| **🌐 Web Deployment**         | Interactive **Streamlit** app                                   | No coding needed                    |
| **📊 Model Benchmarking**     | Training time, params, F1-scores                                | Informed model choice               |

---

## 🏆 Model Performance

<div align="center">

| Model                         | Accuracy     | F1-Score    | Parameters | Training Time |
| ----------------------------- | ------------ | ----------- | ---------- | ------------- |
| **Vision Transformer (Base)** | 🥇 **95.2%** | 🥇 **0.94** | 86M        | 45 min        |
| **DenseNet121**               | 🥈 **94.1%** | 🥈 **0.93** | 8.1M       | 35 min        |
| **EfficientNet-B0**           | 🥉 **93.8%** | 🥉 **0.92** | 5.3M       | 25 min        |
| **VGG16**                     | **92.7%**    | **0.91**    | 138M       | 50 min        |
| **Custom CNN**                | **93.7%**    | **0.93**    | 2.1M       | 20 min        |

</div>  

---

## 🚀 Quick Start

### Installation

```bash
# Clone this repo
git clone https://github.com/your-username/fruit-classification-project.git
cd fruit-classification-project

# Install dependencies
pip install -r requirements.txt

# For CUDA acceleration
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### Run Demo

```bash
# Predict fruit from an image
python predict.py --image your_fruit_image.jpg  

# Launch Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
fruit-classification-project/
├── 🧠 Models/               # Pre-trained models + weights
├── 📓 Notebooks/            # Kaggle notebooks
├── 🚀 train_all_models.py   # Train models easily
├── 🔮 predict.py            # Inference script
├── 🌐 app.py                # Streamlit web app
└── 📊 data_utils.py         # Data handling utilities
```

---

## 🎯 Kaggle Notebooks - Learn by Doing!

<div align="center">

| Notebook                    | Description        | Link                                                                                                                                                 |
| --------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Custom CNN**              | Build from scratch | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-1-customcnn-cse366-group-c)        |
| **EfficientNet & DenseNet** | Modern CNNs        | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-2-eficientnet-80-and-densenet-121) |
| **ConvNeXt & VGG16**        | Classic vs New     | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-3-convnext-and-vgg-16-model)       |
| **Vision Transformer**      | State-of-the-Art   | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-4-vision-transformer)              |
| **XAI Analysis**            | Explainable AI     | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-5-xai-analysis-final)              |

</div>  

---

## 🏋️ Training

### Train One Model

```bash
python train_all_models.py \
  --model vit_base_patch16_224 \
  --data-dir ./Fruits_Original \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --output-dir results
```

### Train All Top Models

```bash
./scripts/train_all.sh
```

---

## 🔮 Prediction Example

```python
from predict import predict_image  

result = predict_image(
    model_type="vit_base_patch16_224",
    image_path="your_fruit.jpg",
    model_path="Models/ViT-Base-16_variety_classification_best.pth"
)  

print(f"Prediction: {result['class']} with {result['confidence']:.2f}% confidence!")  
```

---

## 🌐 Streamlit Web App

```bash
streamlit run app.py
```

**Features:**

* Drag & drop image upload
* Choose different models
* Real-time prediction with confidence
* Heatmap visualization (XAI)
* Mobile-friendly UI

---

## 🎨 Explainable AI (XAI)

<div align="center">

![XAI Heatmap](https://via.placeholder.com/600x200/FF6B6B/FFFFFF?text=XAI+Heatmap+Visualization+Example)

</div>  

Includes:

* **Grad-CAM heatmaps**
* **Saliency maps**
* **Transformer attention visualization**
* **Class Activation Maps**

---

## 📊 Dataset

**FruitVision Dataset – 5 Classes**

| Fruit     | Train | Val | Test |
| --------- | ----- | --- | ---- |
| 🍎 Apple  | 1200  | 300 | 200  |
| 🍌 Banana | 1150  | 287 | 191  |
| 🍇 Grape  | 1100  | 275 | 183  |
| 🥭 Mango  | 1250  | 312 | 208  |
| 🍊 Orange | 1180  | 295 | 196  |

---

## 💡 Why This Project?

* **Students 👨‍🎓** → Learn full DL pipeline, model comparisons
* **Researchers 🔬** → Reproducible experiments, XAI analysis
* **Developers 💻** → Deploy-ready code, modular design

---

## 🛠️ Tech Stack

* **Framework**: PyTorch 2.0+
* **Models**: TIMM library (ViT, EfficientNet, etc.)
* **Deployment**: Streamlit
* **Visualization**: Matplotlib, Plotly
* **XAI**: Captum, TorchCAM
* **GPU Acceleration**: CUDA, AMP

---

## 🚀 Optimization Tips

1. Train with **GPU (CUDA)** → 10x faster
2. Use **Mixed Precision** (AMP) → 2x speedup
3. **Augment data** → better generalization
4. **Pruning & Quantization** → smaller, faster models

---

## 🤝 Contribution

We welcome contributions:

1. Fork repo
2. Create branch `feature/amazing-feature`
3. Commit & push
4. Open PR 🎉

Ideas:

* Add new models
* Better augmentations
* REST API support
* Mobile app deployment

---

## 📜 License

Licensed under **MIT** – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

* **FruitVision Dataset** providers
* **PyTorch Team**
* **TIMM Library**
* **Streamlit**
* Professors & mentors for guidance

---

<div align="center">

**Ready to classify fruits like a pro? 🍎🍌🥭**

\[⭐ Star this repo] · \[📋 Report Issues] · \[🔄 Fork & Contribute]

</div>  

---

✅ This is now the **final full README** with **working Kaggle notebook links**.

Do you also want me to add a **single "Try on Kaggle" badge at the top** (that links to your **Custom CNN notebook** as the entry point)?

