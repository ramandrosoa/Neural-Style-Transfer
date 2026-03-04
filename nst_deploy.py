import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from keras.applications.vgg19 import VGG19

# ── Config (user-facing defaults) ─────────────────────────────────
IMG_SIZE      = 256
DEFAULT_LR    = 0.05
DEFAULT_EPOCHS = 50
DEFAULT_ALPHA  = 30    # content weight
DEFAULT_BETA   = 10    # style weight

STYLE_LAYERS = [
    ('block1_conv1', 0.1),
    ('block2_conv1', 0.1),
    ('block3_conv1', 0.3),
    ('block4_conv1', 0.3),
    ('block5_conv1', 0.2),
]
CONTENT_LAYERS = [('block5_conv4', 1)]

# ── Load VGG19 once at startup ─────────────────────────────────────
# This runs when the module is imported, so the model is ready for
# every request without reloading weights each time.
print("Loading VGG19 weights...")
_vgg = VGG19(include_top=False,
             input_shape=(IMG_SIZE, IMG_SIZE, 3),
             weights='imagenet')
_vgg.trainable = False

def _get_layer_outputs(vgg_model, layer_names):
    outputs = [vgg_model.get_layer(layer[0]).output for layer in layer_names]
    return tf.keras.Model([vgg_model.input], outputs)

_model_outputs = _get_layer_outputs(_vgg, STYLE_LAYERS + CONTENT_LAYERS)
print("VGG19 ready.")

# ── Helper functions ───────────────────────────────────────────────
def _load_image_from_bytes(img_bytes):
    """Load image bytes → float32 tf.Variable of shape (1, H, W, 3)."""
    img = np.array(
        Image.open(BytesIO(img_bytes)).resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    )
    img = tf.constant(np.reshape(img, (1,) + img.shape))
    return tf.Variable(tf.image.convert_image_dtype(img, tf.float32))

def gram_matrix(A):
    return tf.linalg.matmul(A, tf.transpose(A))

def compute_layer_style_cost(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    GS  = gram_matrix(tf.transpose(a_S))
    GG  = gram_matrix(tf.transpose(a_G))
    return tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * (n_C ** 2) * (n_H * n_W) ** 2)

def compute_style_cost(style_output, generated_output):
    J_style = 0
    a_S = style_output[:-1]
    a_G = generated_output[:-1]
    for i, (_, weight) in enumerate(STYLE_LAYERS):
        J_style += weight * compute_layer_style_cost(a_S[i], a_G[i])
    return J_style

def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, [1, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [1, n_H * n_W, n_C])
    return tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)

def total_cost(J_content, J_style, alpha, beta):
    return (alpha * J_content) + (beta * J_style)

def clip(image):
    return tf.clip_by_value(image, 0.0, 1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# ── Main function ──────────────────────────────────────────────────
def run_nst(content_bytes: bytes,
            style_bytes:   bytes,
            learning_rate: float = DEFAULT_LR,
            epochs:        int   = DEFAULT_EPOCHS,
            alpha:         float = DEFAULT_ALPHA,
            beta:          float = DEFAULT_BETA) -> bytes:
    """
    Run Neural Style Transfer and return the result as PNG bytes.

    Parameters
    ----------
    content_bytes   : raw bytes of the content image file
    style_bytes     : raw bytes of the style image file
    learning_rate   : Adam learning rate (user-configurable)
    epochs          : number of optimisation steps
    alpha           : content loss weight
    beta            : style loss weight

    Returns
    -------
    bytes : PNG-encoded generated image
    """

    # Fresh optimizer per call — avoids stale momentum across requests
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Preprocess both images
    preprocessed_content = _load_image_from_bytes(content_bytes)
    preprocessed_style   = _load_image_from_bytes(style_bytes)

    # Encode content and style through VGG19 (fixed — no gradients needed)
    a_C = _model_outputs(preprocessed_content)
    a_S = _model_outputs(preprocessed_style)

    # Initialise generated image = content + small noise
    generated_image = tf.Variable(preprocessed_content)
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.Variable(clip(generated_image + noise))

    # ── Inner train step (compiled once per call) ──────────────────
    @tf.function()
    def train_step(gen_img):
        with tf.GradientTape() as tape:
            a_G       = _model_outputs(gen_img)
            J_style   = compute_style_cost(a_S, a_G)
            J_content = compute_content_cost(a_C, a_G)
            J         = total_cost(J_content, J_style, alpha=alpha, beta=beta)
        grad = tape.gradient(J, gen_img)
        optimizer.apply_gradients([(grad, gen_img)])
        gen_img.assign(clip(gen_img))
        return J

    # ── Optimisation loop ──────────────────────────────────────────
    for i in range(epochs):
        loss = train_step(generated_image)
        if i % 100 == 0:
            print(f"  Epoch {i:>4} / {epochs}  —  loss: {float(loss):.4f}")

    # Encode result as PNG bytes and return
    buf = BytesIO()
    tensor_to_image(generated_image).save(buf, format="PNG")
    return buf.getvalue()