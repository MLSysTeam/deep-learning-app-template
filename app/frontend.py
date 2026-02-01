import streamlit as st
import requests
import os
from PIL import Image
import io
import json
from datetime import datetime

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def main():
    st.set_page_config(page_title="Image Classification App", layout="wide")
    st.title("🖼️ Deep Learning Image Classification")
    st.markdown("""
    Upload an image to classify it using our AI model. 
    The results will be stored in the database for future reference.
    """)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add classify button
            classify_button = st.button("Classify Image")
            
            if classify_button:
                # Show a spinner while processing
                with st.spinner('Classifying image...'):
                    try:
                        # Send the image to the backend
                        response = requests.post(
                            f"{BACKEND_URL}/classify/",
                            files={"file": (uploaded_file.name, uploaded_file, "multipart/form-data")}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store result in session state
                            st.session_state['last_result'] = result
                            
                            st.success(f"Prediction: {result['predicted_class']} (Confidence: {result['confidence']:.2f})")
                        else:
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    with col2:
        st.header("Classification Result")
        
        # Check if we have a result to display
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            st.subheader("Prediction Result:")
            st.markdown(f"**Predicted Class:** {result['predicted_class']}")
            st.markdown(f"**Confidence:** {result['confidence']:.2f}")
            st.markdown(f"**Image Path:** {result['image_path']}")
            st.markdown(f"**Time:** {result['created_at']}")
        else:
            st.info("Upload an image and click 'Classify Image' to see the result here.")

    # Add a section to view recent classifications
    st.header("Recent Classifications")
    if st.button("Refresh Recent Classifications"):
        with st.spinner('Loading recent classifications...'):
            try:
                response = requests.get(f"{BACKEND_URL}/classifications/?limit=10")
                
                if response.status_code == 200:
                    classifications = response.json()
                    
                    if classifications:
                        for cls in classifications:
                            st.write(f"**{cls['predicted_class']}** - Confidence: {cls['confidence']} - "
                                    f"At: {cls['created_at'][:19] if isinstance(cls['created_at'], str) else cls['created_at'].isoformat()}")
                    else:
                        st.warning("No classifications found in the database.")
                else:
                    st.error(f"Failed to fetch classifications: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred while fetching classifications: {str(e)}")


if __name__ == "__main__":
    main()