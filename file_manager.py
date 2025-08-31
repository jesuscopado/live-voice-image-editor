import glob
import os
from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image


class FileManager:
    """Handles all file operations for the image editing tool."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.temp_dir = os.path.join(base_dir, "temp")
        self.generated_image_prefix = "generated_image"
        self._current_input_image: Optional[str] = None

        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize index by scanning existing files once
        self._current_index = self._get_starting_index()

        logger.info(
            f"📁 FileManager initialized: {base_dir}, starting index: {self._current_index}"
        )

    def _get_starting_index(self) -> int:
        """Scan existing files once to determine the starting index."""
        pattern = os.path.join(self.base_dir, f"{self.generated_image_prefix}_*.png")
        files = glob.glob(pattern)
        if not files:
            return 0

        indices = []
        for f in files:
            try:
                idx = int(os.path.basename(f).split("_")[2].split(".")[0])
                indices.append(idx)
            except (ValueError, IndexError):
                continue

        return max(indices, default=-1) + 1

    def generate_output_path(self) -> str:
        """Generate a unique output path for a new image."""
        output_path = os.path.join(
            self.base_dir, f"{self.generated_image_prefix}_{self._current_index}.png"
        )
        self._current_index += 1
        return output_path

    def get_latest_image_path(self) -> Optional[str]:
        """Get the path to the most recently generated image."""
        if self._current_index == 0:
            return None

        latest_path = os.path.join(
            self.base_dir,
            f"{self.generated_image_prefix}_{self._current_index - 1}.png",
        )
        return latest_path if os.path.exists(latest_path) else None

    def set_input_image(self, image_data) -> bool:
        """Set the current input image. Returns True if successful."""
        if image_data is None:
            self._current_input_image = None
            return False

        # Handle numpy array from Gradio
        if isinstance(image_data, np.ndarray):
            return self._save_numpy_array(image_data)
        # Handle file path
        elif isinstance(image_data, str) and os.path.exists(image_data):
            self._current_input_image = image_data
            logger.debug(f"📷 Input image set: {image_data}")
            return True
        else:
            logger.error(f"📷 Invalid image data: {type(image_data)}")
            self._current_input_image = None
            return False

    def _save_numpy_array(self, image_array: np.ndarray) -> bool:
        """Save numpy array as temporary file."""
        try:
            pil_image = Image.fromarray(image_array.astype("uint8"))
            temp_path = os.path.join(self.temp_dir, "current_input.png")
            pil_image.save(temp_path)
            self._current_input_image = temp_path
            logger.debug(f"📷 Numpy array saved: {temp_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save numpy array: {e}")
            self._current_input_image = None
            return False

    @property
    def current_input_image(self) -> Optional[str]:
        """Get the current input image path."""
        return self._current_input_image
