This project demonstrates how to classify emotions in text using a **Convolutional Neural Network (CNN)**. The goal of this model is to classify sentences or text data into one of the following emotion categories:
- **Joy**
- **Fear**
- **Anger**
- **Sadness**
- **Neutral**

We use a dataset of labeled text data (with emotional categories) and process it to train a CNN model. The model uses pre-trained word embeddings from **FastText** to map words to vectors that capture their meanings, and it then uses CNN layers to learn patterns and classify emotions.

---
www.kaggle.com/datasets/yashgupta1161/cnn-dataset

## Libraries Used

### 1. **Pandas**: 
   - **Purpose**: Data manipulation and loading (e.g., reading CSV files).
   - **Functionality**: Helps read and clean the dataset, merge training and testing data, and perform other data preprocessing tasks.

### 2. **NumPy**: 
   - **Purpose**: Numerical operations and array handling.
   - **Functionality**: Used for matrix operations (embedding matrix) and mathematical manipulations on data.

### 3. **Keras**: 
   - **Purpose**: High-level deep learning API for building models.
   - **Functionality**: Used to define and train the neural network. Keras provides layers like Embedding, Conv1D, and Dense that are crucial for the model.

### 4. **TensorFlow**: 
   - **Purpose**: Backend for deep learning computations.
   - **Functionality**: Powers Keras for model training and inference.

### 5. **NLTK**:
   - **Purpose**: Natural language processing.
   - **Functionality**: Used to tokenize and preprocess the text (removing unnecessary characters and breaking text into words).

### 6. **Matplotlib**: 
   - **Purpose**: Plotting graphs and charts.
   - **Functionality**: Used to plot training and validation accuracy/loss to evaluate the model performance visually.

### 7. **Scikit-learn (Sklearn)**: 
   - **Purpose**: Evaluation metrics.
   - **Functionality**: Provides tools to calculate accuracy, F1 score, and confusion matrix for model evaluation.

---

## Steps Involved

### 1. **Data Loading and Preprocessing**:
   - The data is loaded from CSV files.
   - Text data is cleaned by removing special characters, hashtags, and mentions.
   - Sentences are tokenized and transformed into sequences of integers.
   - The sequences are padded to ensure all inputs have the same length.

### 2. **Word Embeddings**:
   - We use pre-trained **FastText word embeddings** (300-dimensional vectors) to represent each word in a sentence. These embeddings capture semantic meaning and are used as input to the model.
   - An **embedding matrix** is created using these pre-trained vectors, and this matrix is used to initialize the embedding layer of the CNN model.

### 3. **Building the CNN Model**:
   - **Embedding Layer**: Converts words into vectors using the pre-trained embedding matrix.
   - **Conv1D Layer**: Convolutional layer that learns spatial patterns in the word sequences.
   - **GlobalMaxPooling1D Layer**: Reduces dimensionality by selecting the most important features.
   - **Dense Layer**: Fully connected layer that produces the final output, predicting one of the five emotion classes.

### 4. **Model Training**:
   - The model is compiled with the **Adam optimizer** and **categorical crossentropy loss**.
   - The model is trained over 6 epochs with a batch size of 256.

### 5. **Evaluation**:
   - **Accuracy**: Measures the percentage of correct predictions.
   - **F1 Score**: A balanced metric for evaluating both precision and recall, especially useful in imbalanced datasets.
   - **Confusion Matrix**: A matrix that shows the true vs. predicted class labels. It helps us understand where the model is making mistakes.

---

## Model Performance

### 1. **Accuracy Plot**:
   - Shows the **training accuracy** and **validation accuracy** during the training process (over epochs).
   - This plot helps to visualize how well the model is performing on both the training and validation datasets.

### 2. **Loss Plot**:
   - Shows the **training loss** and **validation loss**.
   - A decreasing loss curve indicates that the model is improving and learning to minimize error.

### 3. **Confusion Matrix**:
   - A visualization of how well the model is classifying each emotion class.
   - The **normalized confusion matrix** shows the percentage of correct predictions for each emotion.

---

## How to Use

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/emotion-classification.git
   ```

2. **Install dependencies**:
   - Create a Python environment (optional, but recommended):
     ```bash
     python -m venv env
     source env/bin/activate  # For Linux/macOS
     env\Scripts\activate  # For Windows
     ```
   - Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the code**:
   - Ensure that the dataset is available in the specified paths (`/kaggle/input/emotion-dataset/`).
   - Run the script to train the model:
     ```bash
     python train_model.py
     ```

4. **Test the model**:
   - To predict the emotion of a new message, use the following code:
     ```python
     message = ['I am so excited for this event!']
     seq = tokenizer.texts_to_sequences(message)
     padded = pad_sequences(seq, maxlen=max_seq_len)
     pred = model.predict(padded)
     print('Predicted emotion:', class_names[np.argmax(pred)])
     ```

---

## Results

### **Accuracy**: 
   - The model achieves an accuracy of **75.15%** on the test dataset, which indicates a good generalization on unseen data.

### **F1 Score**:
   - The **F1 Score** is calculated to be **75.15%**, reflecting a balanced performance between precision and recall.

---

## Example Output

**Prediction for a Test Sentence**:
```plaintext
Message: "Delivery was an hour late and my pizza was cold!"
Predicted Emotion: Anger (Time taken: 0.02 seconds)
```

---

## Acknowledgements

- The dataset was sourced from **[Kaggle Emotion Dataset](https://www.kaggle.com/datasets)**.
- **FastText** for pre-trained word embeddings.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Notes:
- You can modify the number of epochs, batch size, or other hyperparameters to optimize the model.
- The confusion matrix and F1 score are helpful for diagnosing specific classes the model struggles with, which can be improved by adjusting model parameters or using data augmentation.

---
