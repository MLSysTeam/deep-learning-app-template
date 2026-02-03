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
    st.set_page_config(page_title="Image Segmentation App", layout="wide")
    st.title("🖼️ Deep Learning Image Segmentation")
    st.markdown("""
    Upload an image to segment it using our AI model. 
    You can provide a text prompt to specify what to segment. 
    The results will be stored in the database for future reference.
    """)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Image & Prompt")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        print(uploaded_file)
        
        # Add text input for segmentation prompt
        text_prompt = st.text_input("Text prompt for segmentation", "object")

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Add segment button
            segment_button = st.button("Segment Image")
            
            if segment_button:
                # Show a spinner while processing
                with st.spinner('Segmenting image...'):
                    try:
                        # Create form data for the request
                        # We need to send both the file and the text prompt
                        file_bytes = uploaded_file.getvalue()
                        files = {
                            "file": (uploaded_file.name, file_bytes, uploaded_file.type)
                        }
                        data = {"text_prompt": text_prompt}
                        
                        # Send the image and text prompt to the backend
                        response = requests.post(
                            f"{BACKEND_URL}/segment/",
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store result in session state
                            st.session_state['last_result'] = result
                            
                            # Show segmentation results
                            st.success(f"Segmentation completed! Found {result['num_objects_detected']} objects")
                        else:
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    with col2:
        st.header("Segmentation Result")
        
        # Check if we have a result to display
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            st.subheader("Segmentation Result:")
            
            # Display the segmented image if available
            if 'segmented_image_path' in result and result['segmented_image_path']:
                try:
                    # Load and display the segmented image
                    segmented_img_path = result['segmented_image_path']
                    if os.path.exists(segmented_img_path):
                        segmented_img = Image.open(segmented_img_path)
                        st.image(segmented_img, caption='Segmented Image', use_column_width=True)
                    else:
                        st.warning("Segmented image file not found.")
                except Exception as e:
                    st.warning(f"Could not display segmented image: {str(e)}")
            else:
                st.warning("No segmented image path returned from backend.")
            
            # Show segmentation details
            st.markdown(f"**Number of Objects Detected:** {result['num_objects_detected']}")
            st.markdown(f"**Original Image Path:** {result['image_path']}")
            st.markdown(f"**Time:** {result['created_at']}")
            
            # Show detected objects
            if 'objects' in result:
                st.subheader("Detected Objects:")
                for obj in result['objects']:
                    label = obj.get('label', 'Unknown')
                    confidence = obj.get('confidence', 0)
                    st.markdown(f"- **{label}** - Confidence: {confidence:.2f}")
        else:
            st.info("Upload an image and click 'Segment Image' to see the result here.")

    # Add a section to view recent segmentations
    st.header("Recent Segmentations")
    if st.button("Refresh Recent Segmentations"):
        with st.spinner('Loading recent segmentations...'):
            try:
                response = requests.get(f"{BACKEND_URL}/segmentations/?limit=10")
                
                if response.status_code == 200:
                    segmentations = response.json()
                    
                    if segmentations:
                        for seg in segmentations:
                            # Parse segmented classes from JSON
                            classes = json.loads(seg['segmented_classes']) if seg['segmented_classes'] else []
                            confidences = json.loads(seg['confidence_scores']) if seg['confidence_scores'] else []
                            
                            class_list = ", ".join([f"{cls}({conf:.2f})" for cls, conf in zip(classes, confidences)])
                            st.write(f"**Objects:** {class_list} - Count: {seg['mask_count']} - "
                                    f"At: {seg['created_at'][:19] if isinstance(seg['created_at'], str) else seg['created_at'].isoformat()}")
                    else:
                        st.warning("No segmentations found in the database.")
                else:
                    st.error(f"Failed to fetch segmentations: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred while fetching segmentations: {str(e)}")


if __name__ == "__main__":
    main()