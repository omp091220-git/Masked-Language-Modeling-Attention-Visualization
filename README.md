# Attention - Masked Language Modeling & Attention Visualization

An AI tool that predicts masked words in natural language sequences using BERT (Bidirectional Encoder Representations from Transformers). This project is part of **CS50's Introduction to Artificial Intelligence with Python**.

In addition to predicting missing tokens, this application visualizes the attention weights of all **144 attention heads** (12 layers $\times$ 12 heads) to analyze how transformer models process the contextual relationships between words.

---

## 🚀 How It Works

BERT uses a transformer architecture powered by self-attention mechanisms. This program:
1. Takes a user-input sentence containing a `[MASK]` token.
2. Identifies the numerical token ID of the mask[cite: 1].
3. Feeds the tokenized sequence into a pre-trained BERT model (`bert-base-uncased`) to predict the top 3 most likely words to fill the blank[cite: 1].
4. Extracts the raw multi-dimensional attention tensors from the model's forward pass[cite: 1].
5. Generates heatmaps (PNG diagrams) mapping how strongly each word in the sentence attends to every other word[cite: 1].

---

## 🛠️ Key Implementations

The core logic of the model interaction is handled in `mask.py`, with three custom implementations[cite: 1]:

*   **`get_mask_token_index`**: Scans the Hugging Face `BatchEncoding` input tensor to locate the exact position index of the `[MASK]` token[cite: 1].
*   **`get_color_for_attention_score`**: Linearly scales raw attention weights ($0.0 \le score \le 1.0$) into an RGB grayscale color tuple (`0` = black, `255` = white) to draw clear visual representations[cite: 1].
*   **`visualize_attentions`**: Loops through all 12 layers and 12 heads, extracting attention scores and invoking the image generation utility for each of the 144 attention heads[cite: 1].

---
