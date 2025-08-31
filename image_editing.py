import base64

import replicate
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from loguru import logger

from file_manager import FileManager

load_dotenv()

# Global file manager instance
_file_manager = FileManager()


def set_input_image(image_data) -> None:
    """Set the current input image."""
    _file_manager.set_input_image(image_data)


def get_latest_generated_image_path():
    """Get the path to the most recently generated image."""
    return _file_manager.get_latest_image_path()


def _encode_image_to_data_uri(image_path: str) -> str:
    """Convert a local image file to a base64-encoded data URI."""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine MIME type based on file extension
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"  # Default to PNG

    return f"data:{mime_type};base64,{base64_image}"


@tool("edit_image_tool")
def edit_image(prompt: str) -> str:
    """
    Edit the current input image using an AI model.

    Args:
        prompt: Text description of the edits to make to the image

    Returns:
        String response with success/error message
    """
    try:
        # Check if we have an input image
        current_input = _file_manager.current_input_image
        if not current_input:
            return "❌ ERROR | No input image provided. Please upload an image first."

        # Generate output path
        output_path = _file_manager.generate_output_path()

        # Convert input image to data URI
        image_input = _encode_image_to_data_uri(current_input)
        logger.debug(f"🎨 Editing image with prompt: {prompt}")

        # Call the Nano Banana model
        output = replicate.run(
            "google/nano-banana",
            input={
                "prompt": prompt,
                "image_input": [image_input],  # Single image as list
            },
        )

        # Download the generated image
        response = requests.get(output)
        response.raise_for_status()

        # Save the generated image
        with open(output_path, "wb") as file:
            file.write(response.content)

        logger.debug(f"✅ Image successfully edited and saved to: {output_path}")
        return "✅ Image successfully edited"

    except FileNotFoundError as e:
        return f"❌ ERROR | Input image not found: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"❌ ERROR | Failed to download generated image: {str(e)}"
    except Exception as e:
        return f"❌ ERROR | Image editing failed: {str(e)}"


if __name__ == "__main__":
    set_input_image("data/original.png")
    result = edit_image("change the background to a jungle")
    path = get_latest_generated_image_path()
    print(result)
    print(path)
