# ChurnPredict AI

A sophisticated machine learning solution for customer behavior analysis and salary prediction using deep neural networks.

## 🎯 Overview

ChurnPredict AI is an end-to-end machine learning project that leverages Artificial Neural Networks to solve two critical business problems:

- **Customer Churn Prediction**: Identify customers likely to leave your service
- **Salary Estimation**: Predict customer income levels for better segmentation

Built with TensorFlow and designed for production deployment.

## 📁 Project Structure

```
ChurnPredict-AI/
├── notebooks/
│   ├── experiments.ipynb          # Churn prediction model
│   └── salaryregression.ipynb     # Salary estimation model
├── models/
│   ├── model.h5                   # Trained churn model
│   ├── regression_model.h5        # Trained salary model
│   └── encoders/
│       ├── label_encoder_gender.pkl
│       ├── onehot_encoder_geo.pkl
│       └── scaler.pkl
├── logs/                          # Training logs
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
```bash
git clone <repository-url>
cd ChurnPredict-AI
pip install -r requirements.txt
```

## 🧠 Model Architecture

### Deep Learning Pipeline
Both models use a carefully designed 3-layer architecture:

```
Input Layer → Dense(64, relu) → Dense(32, relu) → Output Layer
```

**Key Features:**
- Optimized for financial data patterns
- Early stopping to prevent overfitting
- Real-time training monitoring with TensorBoard

## 📊 Performance Metrics

| Model | Task | Accuracy/MAE | Training Time |
|-------|------|--------------|---------------|
| Churn Predictor | Classification | 86.5% | ~17 epochs |
| Salary Estimator | Regression | MAE: 50K | ~47 epochs |

## 🛠️ Usage

### Train Models
```python
# Churn Prediction
jupyter notebook notebooks/experiments.ipynb

# Salary Estimation  
jupyter notebook notebooks/salaryregression.ipynb
```

### Monitor Training
```bash
# View training progress
tensorboard --logdir logs/fit
```

## 📈 Dataset Features

**Customer Demographics:**
- Credit Score, Age, Geography, Gender
- Account Balance, Tenure, Product Usage
- Activity Status, Credit Card Ownership

**Target Variables:**
- Churn Status (Binary)
- Estimated Salary (Continuous)

## 🔧 Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn  
- **Visualization**: TensorBoard, Matplotlib
- **Deployment Ready**: Streamlit integration

## 📋 Dependencies

```txt
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras
```

## 🎨 Key Features

✅ **Production Ready**: Serialized models and preprocessing pipelines  
✅ **Monitoring**: Integrated TensorBoard logging  
✅ **Scalable**: Modular architecture for easy extension  
✅ **Validated**: Proper train/test splitting and early stopping  

## 🚀 Future Roadmap

- [ ] Hyperparameter optimization
- [ ] Model ensemble methods  
- [ ] REST API deployment
- [ ] Real-time prediction dashboard
- [ ] A/B testing framework

## 📝 License

MIT License - Feel free to use for commercial projects.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

---
*Built with ❤️ for better customer insights*
