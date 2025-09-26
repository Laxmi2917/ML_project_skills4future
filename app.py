
import streamlit as st
import joblib
import os
import numpy as np

st.set_page_config(page_title="Waste Recycling Helper", page_icon="♻️")
st.title("Waste-to-Recycling Helper ♻️")
st.write("Type the waste item (e.g., 'plastic bottle', 'old phone', 'banana peel') and get a simple recycling suggestion.")

# Load model (expects model file in the same folder as app)
MODEL_FILENAME = "waste_model.pkl"
if not os.path.exists(MODEL_FILENAME):
    st.error("Model file not found. Make sure 'waste_model.pkl' is in the same folder as this app.")
else:
    model = joblib.load(MODEL_FILENAME)

    user_input = st.text_input("Enter waste item or short description", placeholder="e.g., plastic bottle, cardboard box, old phone")
    if st.button("Get recycling suggestion") or user_input:
        if not user_input.strip():
            st.warning("Please type something (e.g., 'plastic bottle').")
        else:
            pred = model.predict([user_input])[0]
            probs = None
            try:
                probs = model.predict_proba([user_input])[0]
                # Get top 3 predictions
                top_idx = np.argsort(probs)[::-1][:3]
                classes = model.classes_
                st.subheader("Top suggestions")
                for i in top_idx:
                    st.write(f"- **{classes[i]}** (confidence: {probs[i]:.2f})")
            except Exception:
                # Some classifiers may not support predict_proba
                pass

            st.subheader("Recommended recycling method")
            st.info(pred)

st.markdown("---")
st.markdown("**Notes:** This is a simple demo model for assessment purposes. For production use, expand the dataset, include localization (different city rules), and add images or barcode lookup for higher accuracy.")
