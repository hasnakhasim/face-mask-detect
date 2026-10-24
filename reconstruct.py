
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

print("Loading local inpainting model...")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)

pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

print("Model loaded successfully!")


def reconstruct_face(image, mask):
    image = image.convert("RGB")
    mask = mask.convert("L").resize(
        image.size, Image.Resampling.NEAREST
    )

    # White = regenerate; black = preserve.
    mask = mask.point(lambda p: 255 if p > 10 else 0)

    if mask.getbbox() is None:
        raise ValueError(
            "No painted area found. Paint over the entire face mask."
        )

    # Expand the painted area to include mask edges and straps.
    mask = mask.filter(ImageFilter.MaxFilter(31))

    # Save a diagnostic image so you can check the selected area.
    mask.save("debug_mask.png")

    # Prepare the input for Stable Diffusion.
    width, height = image.size
    scale = min(512 / width, 512 / height)
    new_width = max(8, int(width * scale) // 8 * 8)
    new_height = max(8, int(height * scale) // 8 * 8)

    working_image = image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS
    )
    working_mask = mask.resize(
        (new_width, new_height), Image.Resampling.NEAREST
    )

    prompt = (
        "photorealistic portrait of an unmasked person, "
        "bare human nose with visible nostrils, natural uncovered "
        "lips, realistic mouth and chin, natural skin texture, "
        "matching the person's visible skin tone, lighting and "
        "facial proportions, unobstructed lower face"
    )

    negative_prompt = (
        "face mask, surgical mask, cloth mask, fabric covering "
        "nose, fabric covering mouth, mask straps, covered mouth, "
        "covered nose, face covering, black patch, blurry, "
        "deformed lips, distorted mouth"
    )

    print("Generating the uncovered-face estimate...")

    with torch.inference_mode():
        generated = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working_image,
            mask_image=working_mask,
            num_inference_steps=30,
            guidance_scale=8.0,
            strength=1.0,
        ).images[0]

    generated = generated.resize(
        image.size, Image.Resampling.LANCZOS
    )

    # Replace the selected area while preserving the rest.
    result = Image.composite(generated, image, mask)

    return result.convert("RGB")
