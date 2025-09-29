import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load and Prepare Dataset
# ---------------------------

# Load the dataset
data = pd.read_csv("data/imdb_reviews.csv")

# Extract texts and labels
texts = data['review'].values
labels = data['sentiment'].map({'positive': 1, 'negative': 0}).values

# Convert labels to numeric (if not already)
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -------------------------------
# 2. Tokenization and Preprocessing
# -------------------------------

# Set vocabulary size and tokenizer
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to same length
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# ----------------------
# 3. Build the LSTM Model
# ----------------------

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Show model architecture
model.summary()

# -------------------
# 4. Train the Model
# -------------------

history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_test_pad, y_test),
    epochs=5,
    batch_size=32,
    verbose=1
)

# ---------------------
# 5. Evaluate and Save
# ---------------------

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2f}")

# Save the trained model
model.save("text_classifier.h5")
print("💾 Model saved as text_classifier.h5")

# ------------------------
# 6. Plot Training Results
# ------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# 7. Predict Function
# -----------------------

def predict_text(text):
    """Predict sentiment for a given input text."""
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(pad)[0][0]
    label = "Positive" if pred > 0.5 else "Negative"
    print(f"📝 '{text}' ➜ {label} ({pred:.2f})")

# -----------------------
# 8. Sample Predictions
# -----------------------

predict_text("The movie was fantastic and very entertaining!")
predict_text("It was the worst film I’ve ever seen.")
