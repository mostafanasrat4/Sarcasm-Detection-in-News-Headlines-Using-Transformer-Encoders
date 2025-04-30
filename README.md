# Sarcasm Detection Using Transformer-Based Text Classification

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Text Preprocessing](#text-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Requirements](#requirements)

## Overview
This project focuses on building a sarcasm detection model using a custom transformer encoder architecture. The objective is to classify news headlines as sarcastic or genuine based on textual content. This includes data preprocessing, exploratory data analysis (EDA), entity recognition, model training, and performance evaluation.

---

## Dataset
The project utilizes two publicly available JSON datasets for sarcasm detection:

- `Sarcasm_Headlines_Dataset_v2.json`
- `Sarcasm_Headlines_Dataset.json`

Both datasets contain two columns:
- `headline`: News headline (string)
- `is_sarcastic`: Label (0 = genuine, 1 = sarcastic)

After loading and merging, the combined dataset is cleaned and analyzed.

---

## Exploratory Data Analysis

### Class Distribution
A bar chart is used to visualize the balance of sarcastic and genuine headlines:
```python
px.bar(data.groupby('is_sarcastic').count().reset_index(), x='headline', title='Count of Sarcastic and Genuine Headlines')
```

### Headline Length Distribution
Histograms are used to analyze headline lengths and detect outliers:
```python
px.histogram(data, x="sentence_length", height=700, color='is_sarcastic', title="Headlines Length Distribution", marginal="box")
```

> After outlier removal (e.g., sentence length = 107), the majority of headlines were within the 5–15 word range.

### Word Clouds
Visual representation of the most common words:
```python
wordcloud = WordCloud(max_words=50, width=600, background_color='white').generate(" ".join(sarcastic))
plt.imshow(wordcloud, interpolation='bilinear')
```

![Wordcloud Example](https://user-images.githubusercontent.com/example/sarcastic_wordcloud.png)

---

## Text Preprocessing

### Cleaning
- Remove special characters
- Lowercasing
- Stopword removal (excluding "not")
- Lemmatization

```python
def text_cleaning(x):
    headline = re.sub('[^a-zA-Z0-9]', ' ', x).lower().split()
    return ' '.join([lemm.lemmatize(word, "v") for word in headline if word not in stop_words])
```

### Entity Extraction (NER)
Utilizing SpaCy to extract entities such as DATE, TIME, CARDINAL:
```python
def get_entities(x):
    return ",".join([word.label_ for word in spacy_eng(x).ents])
```

This step helps determine the relevance of numeric tokens.

---

## Model Architecture

### Tokenization & Padding
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=20)
```

### Custom Transformer Encoder Block
```python
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, heads, neurons):
        ... # Uses MultiHeadAttention, LayerNorm, and FeedForward layers
```

### Model Summary
```python
inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
...
outputs = layers.Dense(1, activation="sigmoid")(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Training
```python
callbacks=[
    EarlyStopping(monitor='val_loss', patience=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, callbacks=callbacks)
```

---

## Evaluation

### Learning Curves
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

![Loss Curve](https://user-images.githubusercontent.com/example/loss_curve.png)

### ROC Curve & AUC Score
```python
y_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test,  y_pred)
plt.plot(fpr, tpr, label="AUC=" + str(auc))
```

![ROC Curve](https://user-images.githubusercontent.com/example/roc_curve.png)

### Classification Report
```python
print(classification_report(y_test, y_pred))
```

| Metric       | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Sarcastic    | 0.88      | 0.85   | 0.86     |
| Genuine      | 0.86      | 0.89   | 0.87     |

---

## Conclusion
- A custom transformer-based encoder model effectively classifies sarcastic headlines.
- Strong performance is observed across standard metrics with proper regularization and attention mechanisms.

---

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NLTK
- SpaCy
- Plotly, Seaborn, Matplotlib
- tqdm, pandas, numpy

```bash
pip install -r requirements.txt
```

---

> This notebook was developed by [Mostafa Nasrat](https://www.linkedin.com/in/mostafanasrat4/)

