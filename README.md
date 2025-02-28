# Real-Time ECG Anomaly Detection Using Deep Learning

This project focused on **developing deep neural networks (DNNs) for real-time anomaly detection in electrocardiogram (ECG) signals**. Using the **ECG5000 dataset**, we built and evaluated multiple deep learning models for **anomaly detection, classification and time series forecasting** to identify abnormal heartbeats in real-time.

## Key Contributions & Achievements

### 📌 Data Preprocessing & Handling Imbalanced Data
- Implemented **oversampling techniques** and **class weighting** to balance the dataset.
- Standardized features and applied **Principal Component Analysis (PCA)** for dimensionality reduction.
- Handled **missing data** and ensured **consistent time series formatting**.

### 📌 Deep Learning Models for Anomaly Detection
- Developed **two autoencoder-based models**:
  - **Standard Autoencoder**
  - **Variational Autoencoder (VAE)**
- **Evaluation Metrics:**
  - **VAE achieved the highest accuracy (88.2%)** and **precision (100%)**, making it the optimal choice.
  - Detected **abnormal heartbeats** with **high recall (88.2%)** and **low false positive rate**.

### 📌 Classification of ECG Signals
- Built **four classification models**:
  - **Baseline Neural Network**
  - **Class-Weighted Model**
  - **Oversampling Model**
  - **Optimised Deep Neural Network**
- **Evaluation Metrics:**
  - **Optimised DNN achieved 59% accuracy**, improving **precision (58.8%)** and **recall (58.4%)**.
  - **Oversampling improved precision** but still struggled with recall.
  - **Class-weighted model had poor recall**, highlighting **limitations in handling imbalanced ECG data**.

### 📌 Time Series Forecasting with Long Short-Term Memory (LSTM) Networks
- Implemented **LSTM-based forecasting models** to predict heartbeat patterns.
- **Hyperparameter tuning improved accuracy from 96.2% to 97%**, reducing loss and improving model stability.
- Introduced **dropout regularisation** to prevent overfitting.

### 📌 Time Instance Forecasting Using LSTMs
- Designed models to **predict individual time instances in ECG signals**.
- Achieved **high accuracy (96.6%)** with optimised hyperparameters.
- Used **Synthetic Minority Oversampling Technique (SMOTE)** to address **class imbalance**.

### 📌 Development of an Interactive Web-Based GUI
- Built a **Graphical User Interface (GUI)** for **real-time ECG signal analysis and anomaly detection**.
- Users can **upload ECG data** and receive **anomaly predictions** using deep learning models.
- Implemented a **REST API** to send ECG data to a server, execute the best deep learning model and return results.

## Key Findings & Business Impact
- **Variational Autoencoder (VAE) is the most effective anomaly detection model**, showing **the highest recall and precision**.
- **Optimized deep neural network is the best classification model**, balancing **accuracy and recall**.
- **LSTM-based time series forecasting effectively predicts future ECG signals**, supporting **early detection of anomalies**.
- **The interactive GUI provides a user-friendly way for healthcare professionals to analyse ECG data**.

---

## 🛠 Technologies & Tools Used
- **Python** (TensorFlow, Keras, NumPy, Pandas, Scikit-Learn)
- **Deep Learning** (Autoencoders, LSTMs, Fully Connected Neural Networks)
- **Anomaly Detection & Classification** (Variational Autoencoders, Class-Weighted Models)
- **Time Series Analysis** (LSTM, Forecasting Models)
- **Data Visualisation** (Matplotlib, Seaborn)
- **Web Deployment** (REST API, Flask, Interactive GUI)

🚀 **This project demonstrated my ability to apply deep learning techniques, optimise models for real-world data and develop a functional AI-powered system for anomaly detection in medical diagnostics.**
