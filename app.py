
import os
import base64
import uuid
from io import BytesIO

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from reconstruct import reconstruct_face

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    original_image = None
    generated_image = None

    if request.method == "POST":
        try:
            uploaded_file = request.files.get("image")
            mask_data = request.form.get("mask_data", "")

            if not uploaded_file or not uploaded_file.filename:
                raise ValueError("Please select a photo first.")

            if not mask_data.startswith("data:image/"):
                raise ValueError(
                    "Please paint over the face mask before generating."
                )

            # Read the uploaded photograph.
            image = Image.open(uploaded_file.stream).convert("RGB")

            # Limit image dimensions to keep processing manageable.
            image.thumbnail((1024, 1024))

            # Decode the transparent canvas sent by the browser.
            encoded_mask = mask_data.split(",", 1)[1]
            canvas_bytes = base64.b64decode(encoded_mask)

            canvas_image = Image.open(
                BytesIO(canvas_bytes)
            ).convert("RGBA")

            # Transparent pixels remain black.
            # Painted, opaque pixels become white.
            alpha = canvas_image.getchannel("A")
            mask = Image.new("L", canvas_image.size, 0)
            mask.paste(255, mask=alpha)

            mask = mask.resize(
                image.size, Image.Resampling.NEAREST
            )

            if mask.getbbox() is None:
                raise ValueError(
                    "The mask is empty. Paint over the face mask."
                )

            filename = f"{uuid.uuid4().hex}.png"

            original_path = os.path.join(
                UPLOAD_FOLDER, filename
            )
            result_path = os.path.join(
                RESULT_FOLDER, filename
            )

            # Save the original photograph.
            image.save(original_path, format="PNG")

            # Call the Google AI function in reconstruct.py.
            result = reconstruct_face(image, mask)
            result.save(result_path, format="PNG")

            original_image = f"uploads/{filename}"
            generated_image = f"results/{filename}"

        except (ValueError, UnidentifiedImageError) as error:
            error_message = str(error)

        except Exception:
            app.logger.exception("Face reconstruction failed")
            error_message = (
                "Image generation failed. Check the VS Code terminal "
                "for the error, then verify your API key and model access."
            )

    return render_template(
        "index.html",
        original_image=original_image,
        generated_image=generated_image,
        error_message=error_message,
    )


@app.errorhandler(413)
def file_too_large(error):
    return "The image is too large. Please upload an image under 10 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)
