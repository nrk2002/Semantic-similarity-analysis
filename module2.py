import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dot, Flatten, Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Function to generate frequency-based negative samples
def get_negative_samples(context_words, vocab_size, word_freq, num_neg_samples):
    negative_samples = []
    while len(negative_samples) < num_neg_samples:
        neg = random.randint(1, vocab_size - 1)
        if neg not in context_words:
            # Sample based on frequency (Unigram ^ 0.75)
            prob = word_freq.get(neg, 1) ** 0.75
            if random.random() < prob:
                negative_samples.append(neg)
    return negative_samples

# Generate Skip-Gram pairs with frequency-based negative sampling
def generate_skipgram_pairs(sentences, vocab_size, window_size=2, negative_samples=4):
    word_pairs = []
    labels = []

    sequences = tokenizer.texts_to_sequences(sentences)
    flat_words = [w for seq in sequences for w in seq]
    word_counts = Counter(flat_words)

    for seq in sequences:
        for idx, word in enumerate(seq):
            start = max(idx - window_size, 0)
            end = min(idx + window_size + 1, len(seq))
            context_words = seq[start:idx] + seq[idx+1:end]

            for context in context_words:
                word_pairs.append((word, context))
                labels.append(1)

                # Negative samples (smart sampling)
                negatives = get_negative_samples(context_words, vocab_size, word_counts, negative_samples)
                for neg in negatives:
                    word_pairs.append((word, neg))
                    labels.append(0)

    return np.array(word_pairs), np.array(labels)

def train_skipgram_model(sentences, embedding_dim=100, epochs=30, window_size=2):
    global tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    vocab_size = len(tokenizer.word_index) + 1

    pairs, labels = generate_skipgram_pairs(sentences, vocab_size, window_size)

    # Shuffle pairs
    indices = np.arange(len(pairs))
    np.random.shuffle(indices)
    pairs = pairs[indices]
    labels = labels[indices]

    # Train-validation split
    split = int(0.8 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Define model
    target_input = Input(shape=(1,))
    context_input = Input(shape=(1,))

    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="word_embedding")
    target_emb = embedding(target_input)
    context_emb = embedding(context_input)

    dot = Dot(axes=-1)([Flatten()(target_emb), Flatten()(context_emb)])
    output = Dense(1, activation='sigmoid')(dot)

    model = Model(inputs=[target_input, context_input], outputs=output)
    model.compile(optimizer=Adam(0.005), loss='binary_crossentropy', metrics=['accuracy'])

    # Compute class weights to balance positives and negatives
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_dict = dict(enumerate(class_weights))

    # Train
    history = model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]], train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        epochs=epochs,
        batch_size=128,
        class_weight=class_weights_dict,
        verbose=1
    )

    # Evaluate ROC
    val_preds = model.predict([val_pairs[:, 0], val_pairs[:, 1]]).flatten()
    fpr, tpr, _ = roc_curve(val_labels, val_preds)
    roc_auc = auc(fpr, tpr)

    # Save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.title('Improved Skip-Gram ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig("static/skipgram_roc_curve.png")
    plt.close()

    # Save results
    word_embeddings = model.get_layer("word_embedding").get_weights()[0]
    metrics = {
        'training_history': history.history,
        'roc_auc': roc_auc,
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
    }

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    np.save("word_embeddings.npy", word_embeddings)
    with open("skipgram_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    return tokenizer, word_embeddings, metrics

# Execution
if __name__ == "__main__":
    with open("train_words.pkl", "rb") as f:
        train_words = pickle.load(f)

    tokenizer, word_embeddings, metrics = train_skipgram_model(train_words)

    print("\nSkip-Gram Training Complete!")
    print(f"Train Accuracy: {metrics['final_train_accuracy']:.4f}")
    print(f"Val Accuracy: {metrics['final_val_accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Artifacts saved: tokenizer.pkl, word_embeddings.npy, skipgram_metrics.pkl, skipgram_roc_curve.png")
