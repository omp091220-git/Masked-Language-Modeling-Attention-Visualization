import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Accepts the ID of the mask token (int) and the tokenizer-generated inputs.
    Returns the 0-indexed position of the mask token in the input sequence.
    If the mask token is not present, returns None.
    """
    # inputs["input_ids"] is typically a 2D tensor (batch_size, sequence_length)
    # Since we are processing a single sentence, we look at the first sequence [0]
    input_ids = inputs["input_ids"][0]
    
    # Convert tensor to a standard Python list/array if it's a TensorFlow tensor
    if hasattr(input_ids, "numpy"):
        input_ids = input_ids.numpy()
        
    for i, token_id in enumerate(input_ids):
        if token_id == mask_token_id:
            return i
            
    return None


def get_color_for_attention_score(attention_score):
    """
    Accepts an attention score (a value between 0 and 1, inclusive).
    Returns an RGB tuple (R, G, B) scaling linearly with the score (0 = black, 1 = white).
    """
    # Convert TensorFlow EagerTensor or NumPy scalar to a pure Python float
    if hasattr(attention_score, "numpy"):
        attention_score = float(attention_score.numpy())
    else:
        attention_score = float(attention_score)
        
    val = int(round(attention_score * 255))
    return (val, val, val)


def visualize_attentions(tokens, attentions):
    """
    Accepts a list of token strings and all generated attention tensors.
    Generates attention diagrams for every attention head across every layer.
    """
    # attentions is a tuple of tensors (one tensor per layer)
    num_layers = len(attentions)
    
    for i in range(num_layers):
        # attentions[i] has shape (batch_size, num_heads, sequence_length, sequence_length)
        # We find the number of heads in this layer
        num_heads = attentions[i].shape[1]
        
        for k in range(num_heads):
            # The assignment specifies layer and head numbers must be 1-indexed
            layer_number = i + 1
            head_number = k + 1
            
            # Extract the specific 2D attention matrix for this head
            # j (beam index) is always 0 in our case
            head_attentions = attentions[i][0][k]
            
            # Call the helper function to write the diagram file
            generate_diagram(layer_number, head_number, tokens, head_attentions)


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
