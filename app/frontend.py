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
    st.set_page_config(page_title="Video Analysis App", layout="wide")
    st.title("🖼️/video️⃣ Deep Learning Video Analysis")
    st.markdown("""
    Upload an image to classify it using our AI model, or upload a video to analyze it frame by frame.
    The results will be stored in the database for future reference.
    """)

    # Create tabs for image and video analysis
    tab1, tab2 = st.tabs(["🖼️ Image Analysis", "🎥 Video Analysis"])

    with tab1:
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

    with tab2:
        st.header("Upload Video")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_video is not None:
            # Add frame interval selection
            frame_interval = st.slider("Frame analysis interval (seconds)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            st.info(f"Will analyze every {frame_interval} second(s) of the video")

            # Add analyze button
            analyze_video_button = st.button("Analyze Video")

            if analyze_video_button:
                # Show a spinner while processing
                with st.spinner(f'Analyzing video, processing every {frame_interval} second(s)...'):
                    try:
                        # Send the video to the backend
                        response = requests.post(
                            f"{BACKEND_URL}/analyze_video/?frame_interval={frame_interval}",
                            files={"file": (uploaded_video.name, uploaded_video, "multipart/form-data")}
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Store result in session state
                            st.session_state['last_video_result'] = result

                            st.success(f"Video analysis complete! Processed {result['total_frames_processed']} frames over {result['duration_seconds']:.2f} seconds")
                        else:
                            st.error(f"Error: {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # Display results based on what was analyzed
    col1, col2 = st.columns([1, 1])

    with col1:
        # Check if we have a result to display
        if 'last_result' in st.session_state and 'last_video_result' not in st.session_state:
            result = st.session_state['last_result']
            st.header("Image Classification Result")
            st.subheader("Prediction Result:")
            st.markdown(f"**Predicted Class:** {result['predicted_class']}")
            st.markdown(f"**Confidence:** {result['confidence']:.2f}")
            st.markdown(f"**Image Path:** {result['image_path']}")
            st.markdown(f"**Time:** {result['created_at']}")
        elif 'last_video_result' in st.session_state:
            result = st.session_state['last_video_result']
            detailed_results = result['summary_results']
            st.header("Video Analysis Summary")
            st.subheader("Summary Statistics:")
            st.markdown(f"**Total Frames Processed:** {detailed_results['total_frames_processed']}")
            st.markdown(f"**Video Duration:** {detailed_results['duration_seconds']:.2f} seconds")
            st.markdown(f"**Frame Interval:** {detailed_results['frame_interval_used']} seconds")
            st.markdown(f"**Average Confidence:** {detailed_results['average_confidence']:.3f}")

            st.subheader("Class Distribution:")
            class_counts = detailed_results['class_counts']
            for class_name, count in class_counts.items():
                st.markdown(f"- **{class_name}:** {count} frames")
        else:
            st.info("Upload an image or video and click 'Classify' or 'Analyze Video' to see the result here.")

    with col2:
        # Display detailed video frame results if available
        if 'last_video_result' in st.session_state:
            result = st.session_state['last_video_result']
            detailed_results = result['summary_results']
            st.header("Frame-by-Frame Results")

            # Show detailed results of each analyzed frame
            for idx, frame_result in enumerate(detailed_results['detailed_results'][:20]):  # Only show first 20 for performance
                st.markdown(f"**Frame {idx+1}** (Time: {frame_result['timestamp']:.2f}s):")
                st.markdown(f"- Class: {frame_result['predicted_class']}")
                st.markdown(f"- Confidence: {frame_result['confidence']:.3f}")

            if len(detailed_results['detailed_results']) > 20:
                st.info(f"Showing first 20 of {len(detailed_results['detailed_results'])} total frames analyzed")

        # Show recent image classifications
        else:
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

    # Add a section to view recent video classifications
    st.header("Recent Video Analyses")
    if st.button("Refresh Recent Video Analyses"):
        with st.spinner('Loading recent video analyses...'):
            try:
                response = requests.get(f"{BACKEND_URL}/video_classifications/?limit=10")

                if response.status_code == 200:
                    video_classifications = response.json()

                    if video_classifications:
                        for vc in video_classifications:
                            st.write(f"**{os.path.basename(vc['video_path'])}** - "
                                    f"Duration: {vc['duration_seconds']:.2f}s - "
                                    f"Frames: {vc['total_frames_processed']} - "
                                    f"At: {vc['created_at'][:19] if isinstance(vc['created_at'], str) else vc['created_at'].isoformat()}")
                    else:
                        st.warning("No video analyses found in the database.")
                else:
                    st.error(f"Failed to fetch video analyses: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to backend at {BACKEND_URL}. Make sure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred while fetching video analyses: {str(e)}")


if __name__ == "__main__":
    main()