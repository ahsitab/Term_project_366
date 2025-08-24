# 🍎🍌 Fruit Classification Project - CSE366 Term Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)
![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)

**State-of-the-Art Deep Learning for Fruit Classification with Vision Transformers, CNNs, and XAI**

[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](#-kaggle-notebooks)
[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-streamlit-app-url.com)

</div>

## 🌟 Introduction

Welcome to the ultimate fruit classification project! This comprehensive deep learning solution leverages cutting-edge models including **Vision Transformers**, **CNNs**, and custom architectures to accurately classify fruits into 5 categories: Apple, Banana, Grape, Mango, and Orange. Perfect for researchers, students, and AI enthusiasts!

## ✨ Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **🤖 Multiple Architectures** | ViT, DeiT, VGG16, ConvNeXt, EfficientNet, DenseNet, Custom CNNs | Choose the best model for your needs |
| **🎯 High Accuracy** | Up to 95% accuracy on test data | Reliable predictions for real-world applications |
| **🔍 XAI Integration** | Grad-CAM heatmaps and saliency visualizations | Understand model decisions |
| **🚀 GPU Optimized** | Full CUDA support with mixed precision training | Faster training and inference |
| **🌐 Web Deployment** | Interactive Streamlit app with beautiful UI | Easy to use without coding |
| **📊 Comprehensive Analysis** | Detailed model comparisons and performance metrics | Make informed decisions |

## 🏆 Model Performance Showcase

<div align="center">

| Model | Accuracy | F1-Score | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
| **Vision Transformer (Base)** | 🥇 **95.2%** | 🥇 **0.94** | 86M | 45 min |
| **DenseNet121** | 🥈 **94.1%** | 🥈 **0.93** | 8.1M | 35 min |
| **EfficientNet-B0** | 🥉 **93.8%** | 🥉 **0.92** | 5.3M | 25 min |
| **VGG16** | **92.7%** | **0.91** | 138M | 50 min |
| **Custom CNN** | **93.7%** | **0.93** | 2.1M | 20 min |

</div>

## 🚀 Quick Start

### Installation - It's a Breeze! 🌈

```bash
# Clone the repository
git clone https://github.com/your-username/fruit-classification-project.git
cd fruit-classification-project

# Install dependencies (creates a virtual environment automatically!)
pip install -r requirements.txt

# For GPU acceleration (highly recommended!)
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### Demo in 60 Seconds! ⏰

```bash
# Try our pre-trained model on your fruit image
python predict.py --image your_fruit_image.jpg

# Or launch the interactive web app
streamlit run app.py
```

## 📁 Project Structure Made Simple

```
fruit-classification-project/
├── 🧠 Models/                 # All model implementations and weights
├── 📓 Notebooks/             # Jupyter notebooks for each model
├── 🚀 train_all_models.py    # One script to train them all!
├── 🔮 predict.py            # Make predictions on new images
├── 🌐 app.py               # Beautiful Streamlit web app
└── 📊 data_utils.py        # Data loading utilities
```

## 🎯 Kaggle Notebooks - Learn by Doing!

<div align="center">

| Notebook | Description | Link |
|----------|-------------|------|
| **Custom CNN** | Build from scratch | [![Open](https://img.shields.io/badge/Open-Notebook-blue)]((https://www.kaggle.com/code/asfarhossainsitab/notebook-1-customcnn-cse366-group-c/edit/run/257729156)) |
| **EfficientNet & DenseNet** | Modern CNNs | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-2-eficientnet-80-and-densenet-121/edit/run/255927767) |
| **ConvNeXt & VGG16** | Classic vs New | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-3-convnext-and-vgg-16-model/edit/run/256115062) |
| **Vision Transformer** | State-of-the-Art | [![Open](https://img.shields.io/badge/Open-Notebook-blue)](https://www.kaggle.com/code/asfarhossainsitab/notebook-4-vision-transformer/edit/run/257364110) |
| **XAI Analysis** | Explainable AI | [![Open](https://img.shields.io/badge/Open-Notebook-blue)]([https://kaggle.com](https://www.kaggle.com/code/asfarhossainsitab/notebook-5-xai-analysis-final/edit)) |

</div>

## 🏋️ Training Made Easy

### Train a Single Model

```bash
python train_all_models.py \
  --model vit_base_patch16_224 \  # Try different models!
  --data-dir ./Fruits_Original \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --output-dir results
```

### Compare Multiple Models

```bash
# Train all top models with one command!
./scripts/train_all.sh
```

## 🔮 Prediction - See the Magic!

```python
from predict import predict_image

# It's this simple!
result = predict_image(
    model_type="vit_base_patch16_224",
    image_path="your_fruit.jpg",
    model_path="Models/ViT-Base-16_variety_classification_best.pth"
)

print(f"Prediction: {result['class']} with {result['confidence']:.2f}% confidence!")
```

## 🌐 Web App - No Code Needed!

Our beautiful Streamlit app lets anyone use our models:

```bash
streamlit run app.py
```

**App Features:**
- 🖼️ Drag-and-drop image upload
- 🤖 Multiple model selection
- 🔍 Real-time predictions with confidence scores
- 🌋 XAI heatmap visualizations
- 📊 Performance comparison charts
- 📱 Mobile-responsive design

## 🎨 XAI - See What the Model Sees!

<div align="center">
  
![XAI Heatmap](https://via.placeholder.com/600x200/FF6B6B/FFFFFF?text=XAI+Heatmap+Visualization+Example)

</div>

Understand model decisions with our integrated Explainable AI features:
- **Grad-CAM heatmaps**
- **Saliency maps**
- **Attention visualization** for Vision Transformers
- **Class activation maps**

## 📊 Dataset Information

We use the **FruitVision** dataset with 5 classes:

| Fruit | Training Images | Validation Images | Test Images |
|-------|-----------------|-------------------|-------------|
| 🍎 Apple | 1,200 | 300 | 200 |
| 🍌 Banana | 1,150 | 287 | 191 |
| 🍇 Grape | 1,100 | 275 | 183 |
| 🥭 Mango | 1,250 | 312 | 208 |
| 🍊 Orange | 1,180 | 295 | 196 |

## 💡 Why Choose This Project?

### For Students 👨‍🎓
- **Complete learning pipeline** from data to deployment
- **Well-documented code** with extensive comments
- **Comparative analysis** of modern architectures
- **Perfect for courses** and personal projects

### For Researchers 🔬
- **Reproducible experiments** with detailed configurations
- **XAI integration** for model interpretability
- **State-of-the-art implementations**
- **Comprehensive evaluation metrics**

### For Developers 💻
- **Production-ready code** with proper modularization
- **Easy deployment** with Streamlit
- **GPU optimization** for fast inference
- **REST API ready** structure

## 🛠️ Technical Stack

- **Framework**: PyTorch 2.0+
- **Vision Models**: TIMM library
- **Web Interface**: Streamlit
- **Visualization**: Matplotlib, Plotly
- **XAI**: Captum, TorchCAM
- **GPU Acceleration**: CUDA, CuDNN

## 🚀 Performance Tips

1. **Use GPU**: 10x faster training with CUDA
2. **Mixed Precision**: 2x speedup with AMP
3. **Data Augmentation**: Improves generalization
4. **Model Pruning**: Reduce size without losing accuracy
5. **Quantization**: Faster inference on edge devices

## 🤝 Contributing

We love contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin amazing-feature`
5. Open a Pull Request

**Looking for ideas?**
- Add new model architectures
- Improve data augmentation
- Enhance the web interface
- Add REST API support
- Create mobile app version

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FruitVision dataset** providers for the comprehensive dataset
- **PyTorch team** for the excellent deep learning framework
- **TIMM library** for Vision Transformer implementations
- **Streamlit** for the amazing web framework
- **Our professors** for guidance and support

## 📞 Support & Community

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Email**: [your-email@university.edu]
- **Office Hours**: [Day, Time, Location]

## 🎓 Educational Value

This project is perfect for:
- Deep learning courses
- Computer vision projects
- Transfer learning experiments
- Model comparison studies
- XAI research
- Deployment tutorials

---

<div align="center">

**Ready to become a fruit classification expert?** 🚀

[⭐ Star this repo] | [📋 Open an issue] | [🔄 Fork it]

</div>

---

**Note**: This project is designed for educational purposes. Always ensure proper attribution and compliance with dataset licenses when using in academic or commercial applications.

