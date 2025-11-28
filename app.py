import json
import numpy as np
import streamlit as st
from scipy.sparse import csr_matrix

# ================================
# Load Saved Hybrid Model Artifact
# ================================

ART_PATH = "hybrid_recommender_custom.json"

with open(ART_PATH, "r", encoding="utf-8") as f:
    art = json.load(f)   # <-- indented properly

# Extract stored components
gmean = art["global_mean"]
rmin = art["rating_min"]
rmax = art["rating_max"]
best_alpha = art["best_alpha"]

user_mean = np.array(art["user_mean"])
item_mean = np.array(art["item_mean"])

# Debugging: check keys
print("Keys in artifact:", art.keys())

# ================================
# Reconstruct the CSR matrix
# ================================

R_csr_data = art.get("R_csr")
if R_csr_data is None:
    raise KeyError("R_csr key not found in the artifact. Available keys: " + str(list(art.keys())))
else:
    R_data = np.array(R_csr_data["data"])
    R_indices = np.array(R_csr_data["indices"])
    R_indptr = np.array(R_csr_data["indptr"])
    R_shape = tuple(R_csr_data["shape"])
    Rmat = csr_matrix((R_data, R_indices, R_indptr), shape=R_shape)
    print("R_csr matrix loaded successfully, shape:", Rmat.shape)

# ================================
# User and Item IDs
# ================================

user_ids = art["user_ids"]
item_ids = art["item_ids"]

# Build index mappings
u_to_idx = {u: i for i, u in enumerate(user_ids)}
i_to_idx = {m: i for i, m in enumerate(item_ids)}

# ================================
# Hybrid Prediction Function
# ================================

def hybrid_predict(user, anime):
    """
    Predict rating using:
    hybrid = alpha * CF + (1 - alpha) * Content
    """
    if user not in u_to_idx or anime not in i_to_idx:
        return None

    u = u_to_idx[user]
    i = i_to_idx[anime]

    # Collaborative Filtering component
    cf_pred = gmean + (user_mean[u] + item_mean[i])

    # Content-based baseline (simple: anime average)
    content_pred = item_mean[i] if item_mean[i] != 0 else gmean

    # Hybrid Weighted Prediction
    hybrid = (best_alpha * cf_pred) + ((1 - best_alpha) * content_pred)

    # Clip to rating scale
    hybrid = max(rmin, min(rmax, hybrid))

    return hybrid

# ================================
# Streamlit App UI
# ================================

st.set_page_config(page_title="Anime Rating Predictor", page_icon="🎌")

st.title("🎌 Anime Hybrid Rating Predictor")
st.write("Select a **User ID** and an **Anime ID**, then click *Predict Rating*.")

# Dropdowns
user_selected = st.selectbox("Select User ID", user_ids)
anime_selected = st.selectbox("Select Anime ID", item_ids)

# Prediction Button
if st.button("Predict Rating"):
    pred = hybrid_predict(user_selected, anime_selected)
    if pred is None:
        st.error("⚠️ User or Anime not found in model.")
    else:
        st.success(f"⭐ **Predicted Rating: {pred:.2f} / {rmax}**")

# Footer
st.markdown("---")
st.caption("Built by Mahen • Streamlit UI")