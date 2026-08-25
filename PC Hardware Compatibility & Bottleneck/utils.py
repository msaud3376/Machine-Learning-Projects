"""
===============================================================
AI-Powered PC Hardware Compatibility & Bottleneck Analyzer
===============================================================
utils.py - Shared Utility Functions

Provides:
  - load_artifacts()      → loads model + all encoders
  - predict_hardware()    → runs inference on a single build
  - get_llm_recommendation() → GROQ-powered smart recommendation
  - bottleneck_gauge_data()  → helper for Plotly gauge chart
===============================================================
"""

import os
import json
import joblib
import numpy as np
import requests

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
MODEL_PATH     = "hardware_model.keras"
ENCODERS_PKL   = "encoders.pkl"
TARGET_ENC_PKL = "target_encoders.pkl"
META_JSON      = "model_meta.json"

GROQ_API_KEY   = "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"
GROQ_MODEL     = "llama-3.3-70b-versatile"
GROQ_ENDPOINT  = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------------------------------------------------
# ARTIFACT LOADER
# ---------------------------------------------------------------
_cache = {}  # module-level cache so we don't reload on each call

def load_artifacts():
    """
    Load model, input encoders, target encoders, and metadata.
    Results are cached after the first call.

    Returns
    -------
    model          : tf.keras.Model
    input_encoders : dict[str, LabelEncoder]   (for building dropdowns)
    target_encoders: dict[str, LabelEncoder]   (for decoding predictions)
    meta           : dict                       (vocab sizes, class counts)
    """
    global _cache

    if _cache:
        return (
            _cache["model"],
            _cache["input_encoders"],
            _cache["target_encoders"],
            _cache["meta"],
        )

    # Import here so utils.py can be imported even before TF is installed
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run train_model.py first."
        )

    model          = tf.keras.models.load_model(MODEL_PATH)
    input_encoders = joblib.load(ENCODERS_PKL)
    target_encoders = joblib.load(TARGET_ENC_PKL)

    with open(META_JSON, "r") as f:
        meta = json.load(f)

    _cache = {
        "model": model,
        "input_encoders": input_encoders,
        "target_encoders": target_encoders,
        "meta": meta,
    }

    return model, input_encoders, target_encoders, meta


# ---------------------------------------------------------------
# SINGLE-BUILD INFERENCE
# ---------------------------------------------------------------
def predict_hardware(cpu_model: str, gpu_model: str,
                     motherboard_model: str, ram_capacity: str,
                     target_resolution: str) -> dict:
    """
    Run multi-task inference for a single PC hardware build.

    Parameters
    ----------
    cpu_model          : str  (must be a value seen during training)
    gpu_model          : str
    motherboard_model  : str
    ram_capacity       : str
    target_resolution  : str

    Returns
    -------
    dict with keys:
        is_compatible        : str   ("Compatible" | "Not Compatible")
        compatibility_conf   : float (0.0 – 1.0)
        bottleneck_pct       : float (clamped 0 – 100)
        bottleneck_component : str
        component_confidence : float
    """
    model, input_encoders, target_encoders, meta = load_artifacts()

    # --- Encode each input ---
    def safe_encode(encoder, value):
        """Transform value; if unseen, map to class index 0."""
        classes = list(encoder.classes_)
        if value not in classes:
            return 0
        return int(encoder.transform([value])[0])

    cpu_enc = safe_encode(input_encoders["CPU_Model"],         cpu_model)
    gpu_enc = safe_encode(input_encoders["GPU_Model"],         gpu_model)
    mb_enc  = safe_encode(input_encoders["Motherboard_Model"], motherboard_model)
    ram_enc = safe_encode(input_encoders["RAM_Capacity"],      ram_capacity)
    res_enc = safe_encode(input_encoders["Target_Resolution"], target_resolution)

    # Model expects shape (batch, 1) for each input
    inputs = {
        "cpu_input": np.array([[cpu_enc]]),
        "gpu_input": np.array([[gpu_enc]]),
        "mb_input":  np.array([[mb_enc]]),
        "ram_input": np.array([[ram_enc]]),
        "res_input": np.array([[res_enc]]),
    }

    # --- Predict ---
    preds = model.predict(inputs, verbose=0)
    compat_prob   = float(preds[0][0][0])     # sigmoid
    bottleneck    = float(preds[1][0][0])     # linear
    component_p   = preds[2][0]               # softmax probabilities

    # --- Decode ---
    le_compat    = target_encoders["Is_Compatible"]
    le_component = target_encoders["Primary_Bottleneck_Component"]

    compat_idx  = int(compat_prob >= 0.5)
    compat_label = le_compat.inverse_transform([compat_idx])[0]
    is_compatible = "Compatible" if compat_label == "Yes" else "Not Compatible"
    compat_conf   = compat_prob if compat_idx == 1 else (1 - compat_prob)

    comp_idx       = int(np.argmax(component_p))
    comp_label     = le_component.inverse_transform([comp_idx])[0]
    comp_conf      = float(component_p[comp_idx])

    bottleneck_clamped = float(np.clip(bottleneck, 0, 100))

    return {
        "is_compatible":        is_compatible,
        "compatibility_conf":   float(compat_conf),
        "bottleneck_pct":       bottleneck_clamped,
        "bottleneck_component": comp_label,
        "component_confidence": comp_conf,
    }


# ---------------------------------------------------------------
# SMART RECOMMENDATION ENGINE (GROQ / LLaMA-3)
# ---------------------------------------------------------------
def get_llm_recommendation(cpu: str, gpu: str, motherboard: str,
                            ram: str, resolution: str,
                            prediction: dict) -> str:
    """
    Call GROQ's LLaMA-3.3-70B to generate a smart PC build recommendation
    based on the hardware configuration and model predictions.

    Returns a markdown-formatted recommendation string.
    """

    system_prompt = (
        "You are an expert PC hardware consultant with deep knowledge of "
        "CPU and GPU compatibility, gaming performance, and system optimization. "
        "Analyze the provided PC build configuration and AI predictions, then "
        "give clear, actionable recommendations. Be concise but thorough. "
        "Use markdown formatting with headers and bullet points."
    )

    user_message = f"""
## PC Build Analysis Request

**Hardware Configuration:**
- CPU: {cpu}
- GPU: {gpu}
- Motherboard: {motherboard}
- RAM: {ram}
- Target Resolution: {resolution}

**AI Model Predictions:**
- Compatibility: {prediction['is_compatible']} (Confidence: {prediction['compatibility_conf']*100:.1f}%)
- Bottleneck Percentage: {prediction['bottleneck_pct']:.1f}%
- Primary Bottleneck: {prediction['bottleneck_component']} (Confidence: {prediction['component_confidence']*100:.1f}%)

Please provide:
1. **Build Assessment** – Overall evaluation of this configuration
2. **Bottleneck Analysis** – What the bottleneck means for real-world performance
3. **Upgrade Recommendations** – Specific components to upgrade (with examples)
4. **Optimization Tips** – Settings or tweaks to improve performance without hardware changes
5. **Value Assessment** – Whether this build is cost-effective for the target resolution
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Recommendation engine timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ API Error: {e.response.status_code} – {e.response.text[:200]}"
    except Exception as e:
        return f"⚠️ Could not fetch recommendation: {str(e)}"


# ---------------------------------------------------------------
# GAUGE CHART HELPER
# ---------------------------------------------------------------
def bottleneck_color(pct: float) -> str:
    """Return a color string based on bottleneck percentage severity."""
    if pct < 15:
        return "#2ecc71"   # Green – balanced
    elif pct < 30:
        return "#f39c12"   # Orange – moderate
    else:
        return "#e74c3c"   # Red – severe


def bottleneck_label(pct: float) -> str:
    """Return a human-readable severity label."""
    if pct < 15:
        return "Balanced"
    elif pct < 30:
        return "Moderate Bottleneck"
    else:
        return "Severe Bottleneck"