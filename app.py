from flask import Flask, request, render_template
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import numpy as np
import base64
import os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Initialize MTCNN detector
detector = MTCNN()
DB_PATH = "./database"  # Define the database path directly in the code

# Function to convert OpenCV image to base64
def convert_image_to_base64(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to draw bounding boxes and labels
def draw_face(img, x1, y1, x2, y2, name, is_unknown):
    color = (0, 0, 255) if is_unknown else (0, 255, 0)
    label = 'Unknown' if is_unknown else name
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Process DeepFace results and draw bounding boxes on detected faces
def process_results(results, img, distance_threshold):
    found_person = False
    for idx, row in results.iterrows():
        is_unknown = row['distance'] > distance_threshold
        if not is_unknown:
            found_person = True
        person_name = os.path.basename(os.path.dirname(row['identity'])) if not is_unknown else 'Unknown'
        x1, y1 = int(row['source_x']), int(row['source_y'])
        x2, y2 = x1 + int(row['source_w']), y1 + int(row['source_h'])
        draw_face(img, x1, y1, x2, y2, person_name, is_unknown)
    return found_person

# Find faces in the image and match them with a database
def find_and_draw_boxes(img_path, db_path, model_name='Facenet', detector_backend='mtcnn', enforce_detection=False, distance_threshold=0.5):
    try:
        dfs = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection
        )
    except Exception as e:
        print(f"Error during face finding with DeepFace: {e}")
        return None, False

    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image, check the image path.")
        return None, False

    found_person = False

    if isinstance(dfs, list):
        for df in dfs:
            if not df.empty:
                found_person = process_results(df, img, distance_threshold) or found_person
    elif not dfs.empty:
        found_person = process_results(dfs, img, distance_threshold)

    if not found_person:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            draw_face(img, x, y, x + w, y + h, 'Unknown', True)

    return img, found_person

# Verify faces in two images and draw bounding boxes
def verify_and_draw_boxes(img1, img2, model_name='Facenet512', detector_backend='mtcnn', distance_metric='cosine', enforce_detection=False):
    faces1 = detector.detect_faces(img1)
    faces2 = detector.detect_faces(img2)

    if not faces1 or not faces2:
        print("No faces detected in one or both images.")
        return img1, img2

    similarity_labels_img1 = ["Different"] * len(faces1)
    similarity_labels_img2 = ["Different"] * len(faces2)

    for i, face1 in enumerate(faces1):
        x1, y1, w1, h1 = face1['box']
        face1_region = img1[y1:y1 + h1, x1:x1 + w1]

        for j, face2 in enumerate(faces2):
            x2, y2, w2, h2 = face2['box']
            face2_region = img2[y2:y2 + h2, x2:x2 + w2]

            result = DeepFace.verify(
                img1_path=face1_region,
                img2_path=face2_region,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=enforce_detection
            )

            if result['distance'] <= 0.50:
                similarity_labels_img1[i] = "Similar"
                similarity_labels_img2[j] = "Similar"
                break

    for i, face1 in enumerate(faces1):
        x1, y1, w1, h1 = face1['box']
        cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (250, 250, 250), 1)
        cv2.putText(img1, similarity_labels_img1[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    for j, face2 in enumerate(faces2):
        x2, y2, w2, h2 = face2['box']
        cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), (250, 250, 250), 1)
        cv2.putText(img2, similarity_labels_img2[j], (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return img1, img2

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        img1 = request.files['img1']
        img2 = request.files['img2']

        img1_np = np.frombuffer(img1.read(), np.uint8)
        img2_np = np.frombuffer(img2.read(), np.uint8)
        img1_cv = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
        img2_cv = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)

        result_img1, result_img2 = verify_and_draw_boxes(img1_cv, img2_cv)

        img1_base64 = convert_image_to_base64(result_img1)
        img2_base64 = convert_image_to_base64(result_img2)

        return render_template('results.html', img1_data=img1_base64, img2_data=img2_base64)

    return render_template('index.html')

@app.route('/match_faces_with_db', methods=['POST'])
def match_faces_with_db_route():
    input_img = request.files['input_img']
    input_img_path = "input_img.jpg"
    input_img.save(input_img_path)

    img, found_match = find_and_draw_boxes(input_img_path, DB_PATH)

    if img is not None:
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return render_template('results.html', match_image=found_match, match_image_base64=img_base64)
    else:
        return render_template('results.html', match_image=False)

if __name__ == '__main__':
    app.run(debug=True)
