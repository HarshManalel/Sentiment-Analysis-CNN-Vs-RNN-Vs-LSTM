# Comparing 1d CNN, RNN & LSTM for Sentiment Analysis 🧠📊

The project aims to compare and evaluate different deep learning models for sentiment analysis tasks, focusing on their ability to capture various text patterns and dependencies. The models were trained and evaluated using GPU acceleration for faster computation and scikit-learn for performance metrics.

## Key Features 🔑

- **Model Architectures:** Implemented 1D CNN, RNN, and LSTM models.
- **Dataset:** Utilized the IMDB dataset for sentiment analysis.
- **Performance Evaluation:** Metrics include accuracy, precision, recall, F1 score, and AUC-ROC.
- **Visualization:** Used matplotlib for visualizing training/validation metrics and ROC curves.
- **GPU Acceleration:** Leveraged NVIDIA GPU with CUDA support for faster model training.
- **Natural Language Processing:** Applied nltk for tokenization and preprocessing tasks.

## Results and Insights 📈

- **1D CNN Model:** Achieved 83.9% accuracy, suitable for capturing local patterns in text sequences.
- **RNN Model:** Demonstrated 89.6% AUC-ROC score, effective in handling temporal dependencies.
- **LSTM Model:** Offered robust performance in capturing long-term dependencies, although with higher computational demands.
