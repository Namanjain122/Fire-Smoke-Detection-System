import os
import zipfile
from pathlib import Path
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import moviepy.editor as mp

# Initialize Flask
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

# Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv", "zip"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        ext = filename.rsplit(".", 1)[1].lower()

        # 📦 Handle ZIP (batch processing)
        if ext == "zip":
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(UPLOAD_FOLDER)

            # Process each file inside the ZIP
            for extracted_file in Path(UPLOAD_FOLDER).rglob("*"):
                if extracted_file.is_file() and allowed_file(extracted_file.name):
                    sub_ext = extracted_file.suffix.lower()[1:]

                    # Image inside ZIP
                    if sub_ext in ["png", "jpg", "jpeg"]:
                        model(str(extracted_file), save=True, project=RESULTS_FOLDER, name="zip", exist_ok=True)

                    # Video inside ZIP
                    elif sub_ext in ["mp4", "avi", "mov", "mkv"]:
                        results = model(str(extracted_file), save=True, project=RESULTS_FOLDER, name="zip", exist_ok=True)
                        save_dir = Path(results[0].save_dir)
                        files = list(save_dir.glob("*"))
                        if files:
                            processed_file = files[0]
                            mp4_path = str(processed_file.with_suffix(".mp4"))
                            clip = mp.VideoFileClip(str(processed_file))
                            clip.write_videofile(mp4_path, codec="libx264")

            return render_template("index.html", zip_done=True, zip_path=RESULTS_FOLDER)

        # 🖼 Handle Image Upload
        elif ext in ["png", "jpg", "jpeg"]:
            results = model(file_path, save=True, project=RESULTS_FOLDER, name="image", exist_ok=True)
            save_dir = Path(results[0].save_dir)
            output_file = list(save_dir.glob("*.jpg"))[0]
            rel_path = os.path.relpath(output_file, "static").replace("\\", "/")
            return render_template("index.html", image_file=rel_path)
    
        # 🎥 Handle Video Upload
        elif ext in ["mp4", "avi", "mov", "mkv"]:
            results = model(file_path, save=True, project=RESULTS_FOLDER, name="video", exist_ok=True)
            save_dir = Path(results[0].save_dir)
            files = list(save_dir.glob("*"))
            if not files:
                return "Error: No output video generated"

            processed_file = files[0]

            # Convert to MP4 (always)
            mp4_path = str(processed_file.with_suffix(".mp4"))
            clip = mp.VideoFileClip(str(processed_file))
            clip.write_videofile(mp4_path, codec="libx264")
            processed_file = Path(mp4_path)

            rel_path = os.path.relpath(processed_file, "static").replace("\\", "/")
            return render_template("index.html", video_file=rel_path)

    return "Invalid file type!"


if __name__ == "__main__":
    app.run(debug=True)
