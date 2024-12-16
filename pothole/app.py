from ultralytics import YOLO
import cv2
import json
import os
import streamlit as st

# Function to process an image
def process_image(image_path, model):
    image = cv2.imread(image_path)
    results = model.predict(source=image_path, show=False)

    # Initialize an empty list to store detection results
    output_data = {
        "file": os.path.basename(image_path),
        "detections": []
    }

    for box in results[0].boxes:
        # Extract bounding box data
        x, y, w, h = box.xywh[0].tolist()
        x1, y1 = int(x - w / 2), int(y - h / 2)  # Convert center-based to top-left
        x2, y2 = int(x + w / 2), int(y + h / 2)
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        # Add detection to output data
        output_data["detections"].append({
            "class": class_id,
            "confidence": confidence,
            "bbox": [x, y, w, h]
        })

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label text
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the processed image
    output_image_path = "output_image_with_bboxes.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Processed image saved as: {output_image_path}")

    return output_data

# Function to process a video
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4 output
    video_output = "output/output_video_with_bboxes.mp4"
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    # Perform predictions
    results = model.predict(source=video_path, show=False)

    # Initialize an empty list to store structured results
    output_data = []

    # Iterate through frames and YOLO predictions
    for i, result in enumerate(results):
        success, frame = cap.read()
        if not success:
            break  # Exit loop if video frame reading fails

        frame_data = {
            "frame": i,
            "detections": []
        }

        # Draw each bounding box on the frame
        for box in result.boxes:
            # Extract bounding box data
            x, y, w, h = box.xywh[0].tolist()
            x1, y1 = int(x - w / 2), int(y - h / 2)  # Convert center-based to top-left
            x2, y2 = int(x + w / 2), int(y + h / 2)
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            # Add detection to frame_data
            frame_data["detections"].append({
                "class": class_id,
                "confidence": confidence,
                "bbox": [x, y, w, h]
            })

            # Draw the bounding box
            color = (0, 255, 0)  # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label text
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_data.append(frame_data)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Video with bounding boxes saved as: {video_output}")
    return output_data

# Streamlit app
st.title("YOLO Object Detection")
st.write("Upload an image or a video for object detection.")

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to disk
    input_path = f"temp_{uploaded_file.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load the YOLO model
    model = YOLO("y8best.pt")

    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        detections = process_image(input_path, model)
        output_json = "output/image_results.json"
        with open(output_json, "w") as json_file:
            json.dump(detections, json_file, indent=4)
        st.image("output_image_with_bboxes.jpg", caption="Processed Image with Bounding Boxes")
        st.json(detections)

    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        detections = process_video(input_path, model)
        output_json = "output/video_results.json"
        with open(output_json, "w") as json_file:
            json.dump(detections, json_file, indent=4)
        # st.video("output/oustretput_video_with_bboxes.mp4")
        st.write("Video has been processed and saved in the 'output' folder.")
        st.json(detections)

    else:
        st.error("Unsupported file type.")
# Streamlit app
# st.title("YOLO Object Detection")
# st.write("Upload an image or a video for object detection.")

# uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

# if uploaded_file is not None:
#     # Save the uploaded file to disk
#     input_path = f"temp_{uploaded_file.name}"
#     with open(input_path, "wb") as f:
#         f.write(uploaded_file.read())

#     # Load the YOLO model
#     model = YOLO("y8best.pt")

#     if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#         # Process image
#         detections = process_image(input_path, model)
#         output_json = "output/image_results.json"
#         with open(output_json, "w") as json_file:
#             json.dump(detections, json_file, indent=4)
#         st.image("output_image_with_bboxes.jpg", caption="Processed Image with Bounding Boxes")
#         st.json(detections)

#     elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
#         # Video processing is skipped; just display a message
#         st.write("Video has been processed and saved in the 'output' folder.")
#         # If you need to display any related information or data, you can do it here
#         detections = {"message": "Video saved successfully in 'output' folder"}
#         output_json = "output/video_results.json"
#         with open(output_json, "w") as json_file:
#             json.dump(detections, json_file, indent=4)
#         st.json(detections)

#     else:
#         st.error("Unsupported file type.")

