# Fruit Classification Project - CSE366 Term Project

A comprehensive deep learning project for fruit variety classification using multiple state-of-the-art models including Vision Transformers, CNNs, and custom architectures. This project implements a complete image classification pipeline from data preparation to deployment.

## 📋 Project Overview

This project implements and compares various deep learning models for fruit classification with 5 classes: Apple, Banana, Grape, Mango, and Orange. The project includes pre-trained models, training scripts, inference utilities, and an interactive Streamlit web application.

## 🚀 Features

- **Multiple Model Architectures**:
  - Vision Transformers (ViT, DeiT variants)
  - CNN Models (VGG16, ConvNeXt-Tiny, EfficientNet-B0, DenseNet121)
  - Custom CNN Architectures

- **Complete Pipeline**: Data preparation, model training, evaluation, and deployment
- **Unified Training Interface**: Single script to train all model types
- **Inference Pipeline**: Easy prediction on new images
- **Model Comparison**: Tools to compare performance across models
- **XAI Integration**: Explainable AI with heatmap visualizations
- **Interactive Web App**: Streamlit application for easy model interaction
- **GPU Optimized**: Full CUDA support with mixed precision training

## 📁 Project Structure

```
.
├── Models/                    # Model implementations
│   ├── vit_model.py          # Vision Transformer implementation
│   ├── vgg_model.py          # VGG16 model
│   ├── convnext_model.py     # ConvNeXt model
│   ├── efficientnet_model.py # EfficientNet model
│   ├── densenet_model.py     # DenseNet model
│   ├── custom_cnn.py         # Custom CNN architectures
│   ├── model_utils.py        # Model loading utilities
│   └── *.pth                 # Pre-trained model weights
├── Notebook/                 # Jupyter notebooks
│   ├── vision-transformer-notebook-4-1.ipynb
│   ├── convnext_and_vgg_16_model_fruit(1).py
│   ├── eficientnet_80_and_densenet_121_final.py
│   └── custom-cnn-model-notebook-1.ipynb
├── train_all_models.py       # Unified training script
├── predict.py               # Inference script
├── data_utils.py           # Data loading utilities
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔗 Kaggle Notebooks

<div align="center">

[![Custom CNN Notebook]([https://img.shields.io/badge/Kaggle-Custom_CNN_Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/asfarhossainsitab/customcnn-cse366-group-c/edit](https://www.kaggle.com/code/asfarhossainsitab/notebook-1-customcnn-cse366-group-c))
[![EfficientNet & DenseNet Notebook](https://img.shields.io/badge/Kaggle-EfficientNet_&_DenseNet-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/asfarhossainsitab/notebook-2-eficientnet-80-and-densenet-121)
[![ConvNeXt & VGG16 Notebook](https://img.shields.io/badge/Kaggle-ConvNeXt_&_VGG16-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/asfarhossainsitab/notebook-3-convnext-and-vgg-16-model)
[![Vision Transformer Notebook](https://img.shields.io/badge/Kaggle-Vision_Transformer-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/asfarhossainsitab/notebook-4-vision-transformer)
[![XAI_Analysis Notebook](https://www.kaggle.com/code/asfarhossainsitab/notebook-5-xai-analysis-1)

</div>

### Individual Notebook Links:
- **Custom CNN Implementation**: [Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/customcnn-cse366-group-c/edit)
- **EfficientNet-B0 & DenseNet-121**: [Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/notebook-2-eficientnet-80-and-densenet-121)
- **ConvNeXt-Tiny & VGG16**: [Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/notebook-3-convnext-and-vgg-16-model)
- **Vision Transformer**: [Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/notebook-4-vision-transformer)
- **XAI_Analysis**:  [Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/notebook-5-xai-analysis-1)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fruit-classification-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support** (optional):
   Install the appropriate CUDA version of PyTorch:
   ```bash
   # Example for CUDA 11.7
   pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
   ```

## 📊 Available Models

### Vision Transformers
- `vit_tiny_patch16_224` - Tiny Vision Transformer
- `vit_small_patch16_224` - Small Vision Transformer  
- `vit_base_patch16_224` - Base Vision Transformer (Recommended)
- `vit_large_patch16_224` - Large Vision Transformer
- `deit_tiny_patch16_224` - DeiT Tiny
- `deit_small_patch16_224` - DeiT Small
- `deit_base_patch16_224` - DeiT Base

### CNN Models
- `vgg16` - VGG16 with custom classifier
- `convnext_tiny` - ConvNeXt-Tiny
- `efficientnet_b0` - EfficientNet-B0
- `densenet121` - DenseNet-121

### Custom Models
- `custom_cnn` - Simple custom CNN
- `custom_cnn_v2` - Enhanced custom CNN with residuals

## 🏋️ Training

### Single Model Training
```bash
python train_all_models.py \
  --model vit_base_patch16_224 \
  --data-dir /path/to/fruit/dataset \
  --batch-size 32 \
  --epochs 20 \
  --learning-rate 0.0001 \
  --output-dir results
```

### Batch Training Multiple Models
```bash
# Example script to train multiple models
for model in vit_base_patch16_224 vgg16 efficientnet_b0 densenet121; do
  python train_all_models.py \
    --model $model \
    --data-dir /path/to/fruit/dataset \
    --batch-size 32 \
    --epochs 20 \
    --output-dir results/$model
done
```

### Training Parameters
- `--model`: Model architecture to train
- `--data-dir`: Path to dataset directory (should have class subfolders)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 20)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--freeze-backbone`: Freeze backbone weights (default: True)
- `--output-dir`: Output directory for results (default: results)

## 🔮 Inference

### Single Image Prediction
```bash
python predict.py \
  --model-type vit_base_patch16_224 \
  --model-path Models/ViT-Base-16_variety_classification_best.pth \
  --image path/to/your/image.jpg \
  --num-classes 5 \
  --class-names Apple Banana Grape Mango Orange
```

### Batch Prediction
The script can be modified to process multiple images by providing a directory path instead of a single image.

## 📈 Model Comparison

To compare model performance:

1. Train multiple models using the training script
2. Use the evaluation results saved in JSON format
3. Compare accuracy, F1 scores, and training metrics

## 🗃️ Dataset

The project is designed to work with the FruitVision dataset or any dataset organized in the following structure:

```
dataset/
├── Apple/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Banana/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Grape/
│   └── ...
├── Mango/
│   └── ...
└── Orange/
    └── ...
```

## 🎯 Performance

Pre-trained models achieve the following performance on the test set:

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| ViT-Base-16 | ~95% | ~0.94 | 86M |
| VGG16 | ~92% | ~0.91 | 138M |
| EfficientNet-B0 | ~93% | ~0.92 | 5.3M |
| DenseNet121 | ~94% | ~0.93 | 8.1M |
| Custom CNN (VGG16) | ~93.7% | ~0.93 | 138M |

## 🌐 Streamlit Web Application

Run the interactive web application:

```bash
streamlit run app.py
```

The application provides:
- Model selection from available pre-trained models
- Image upload functionality
- Real-time predictions with confidence scores
- XAI heatmap visualizations
- Model performance comparison

## 🚦 Usage Examples

### Example 1: Train Vision Transformer
```bash
python train_all_models.py --model vit_base_patch16_224 --data-dir ./Fruits_Original --epochs 25
```

### Example 2: Predict on New Image
```bash
python predict.py --model-type vit_base_patch16_224 --model-path results/vit_base_patch16_224/vit_base_patch16_224_variety_classification_best.pth --image test_apple.jpg
```

### Example 3: Use Pre-trained Model
```bash
python predict.py --model-type vit_base_patch16_224 --model-path Models/ViT-Base-16_variety_classification_best.pth --image new_fruit.jpg
```

### Example 4: Launch Web App
```bash
streamlit run app.py
```

## 🔬 XAI Implementation

The project includes Explainable AI (XAI) features using Grad-CAM and other saliency methods to provide heatmap visualizations that show which parts of the image contributed most to the classification decision.

## 📝 Technical Report

A comprehensive technical report is available that includes:
- Abstract and introduction
- Related work with citations
- Dataset analysis and preprocessing
- Methodology details for all models
- Experimental results and comparisons
- XAI analysis and insights
- Streamlit deployment details
- Conclusion and future work

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FruitVision dataset providers
- PyTorch and torchvision teams
- TIMM library for Vision Transformer implementations
- Research papers and authors of the implemented models

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the examples in the Notebooks directory
- Review the model documentation in each model file

---

**Note**: This project is for educational and research purposes. Always ensure you have the right to use any datasets and comply with relevant licenses.

