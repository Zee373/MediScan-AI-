import os
import numpy as np
from PIL import Image, ImageFilter

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# 14 NIH diseases + "No Finding"
LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

# --------------------
# Try to import TensorFlow/Keras. If not available, we will gracefully fallback.
# --------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    tf = None
    keras_load_model = None

def load_keras_model(model_path: str):
    """Load Keras model if available. Otherwise return None."""
    if keras_load_model is None:
        print("TensorFlow not installed. Using dummy heatmap.")
        return None
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Using dummy heatmap.")
        return None
    try:
        model = keras_load_model(model_path, compile=False)
        return model
    except Exception as e:
        print("Model load error:", e)
        return None

def _prepare_image_for_model(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_model(model, img_path):
    """Return (label, confidence). When model is None, returns ('No Finding', 0.0) as placeholder."""
    if model is None:
        return "No Finding", 0.0
    try:
        arr = _prepare_image_for_model(img_path)
        preds = model.predict(arr)
        if preds.ndim == 2 and preds.shape[1] >= 1:
            idx = int(np.argmax(preds[0]))
            conf = float(preds[0][idx])
        else:
            idx = 0
            conf = 0.0
        label = LABELS[idx] if idx < len(LABELS) else f"Class_{idx}"
        return label, conf
    except Exception as e:
        print("Prediction error:", e)
        return "Unknown", 0.0

def generate_gradcam_or_dummy(model, img_path, out_path):
    """Generate a Grad-CAM heatmap if TF is available, otherwise a soft focus mask (dummy)."""
    try:
        if model is not None and tf is not None:
            # Use the last conv layer; try to find one
            last_conv_layer = None
            for layer in reversed(model.layers):
                if 'conv' in layer.name or 'bn' in layer.name or 'block' in layer.name:
                    last_conv_layer = layer
                    break
            if last_conv_layer is None:
                # Fallback: use the last layer prior to classification if it's 4D
                for layer in reversed(model.layers):
                    try:
                        if len(layer.output_shape) == 4:
                            last_conv_layer = layer
                            break
                    except Exception:
                        continue

            img = Image.open(img_path).convert("RGB").resize((224, 224))
            img_array = np.expand_dims(np.array(img).astype("float32")/255.0, axis=0)

            grad_model = tf.keras.models.Model(
                [model.inputs], [last_conv_layer.output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
            heatmap = (heatmap * 255).numpy().astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap).resize(img.size)
            heatmap_img = heatmap_img.convert("RGBA")

            # Colorize heatmap (simple palette: red)
            colorized = Image.new("RGBA", heatmap_img.size)
            r = heatmap_img.split()[0]
            colorized = Image.merge("RGBA", (r, Image.new('L', r.size), Image.new('L', r.size), r))
            base = img.convert("RGBA")
            overlay = Image.blend(base, colorized, alpha=0.5)
            overlay = overlay.convert("RGB")
            overlay.save(out_path)
            return out_path

        # Dummy heatmap: blur + red vignette
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=8))
        # Create simple red overlay mask
        w, h = img.size
        y, x = np.ogrid[-h/2:h/2, -w/2:w/2]
        mask = np.exp(-(x*x + y*y) / (2*(min(w,h)/4.0)**2))
        mask = (mask * 255).astype(np.uint8)
        from PIL import ImageOps
        mask_img = Image.fromarray(mask).resize(img.size)
        red = Image.new("RGB", img.size, (255, 0, 0))
        overlay = Image.blend(blurred, red, alpha=0.35)
        overlay = Image.composite(overlay, img, ImageOps.invert(mask_img.convert("L")))
        overlay.save(out_path)
        return out_path
    except Exception as e:
        print("Grad-CAM error:", e)
        # As last resort, copy original
        Image.open(img_path).save(out_path)
        return out_path
