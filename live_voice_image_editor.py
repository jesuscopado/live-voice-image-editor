import os
import sys
from typing import Generator, Optional, Tuple

import gradio as gr
import numpy as np
from fastrtc import (
    AdditionalOutputs,
    AlgoOptions,
    ReplyOnPause,
    Stream,
    get_stt_model,
    get_tts_model,
)
from loguru import logger

from image_editor_agent import ImageEditorAgent


class VoiceLiveImageEditor:
    """
    A voice-controlled image editing application using FastRTC and AI agents.

    This app allows users to upload images and edit them using voice commands,
    with real-time audio processing and visual feedback.
    """

    def __init__(
        self,
        model_provider: str = "ollama",
        model_name: str = "qwen3:1.7b",
        speech_threshold: float = 0.1,
        log_level: str = "DEBUG",
    ):
        """
        Initialize the Voice Image Editing App.

        Args:
            model_provider: AI model provider ('ollama' or 'groq')
            model_name: Specific model to use
            speech_threshold: Voice activity detection threshold
            log_level: Logging level
        """
        # Setup logging
        self._setup_logging(log_level)

        # Initialize models
        self.stt_model = get_stt_model()  # moonshine/base
        self.tts_model = get_tts_model()  # kokoro

        # Initialize agent
        self.agent = ImageEditorAgent(
            model_provider=model_provider, model_name=model_name
        )

        # Algorithm options
        self.algo_options = AlgoOptions(speech_threshold=speech_threshold)

        # UI configuration
        self.ui_config = {
            "title": "Voice Live Image Editor",
            "subtitle": "Edit images in real-time using Gemini 2.5 Flash Image (Nano Banana 🍌)",
        }

        logger.info("🎙️ VoiceImageEditingApp initialized")

    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        logger.remove(0)
        logger.add(sys.stderr, level=log_level)

    def _get_generated_image_path(self, agent_response: dict) -> Optional[str]:
        """
        Get the generated image path if image editing tool was used.

        This checks if the image editing tool was used successfully and
        returns the latest generated image path.

        Args:
            agent_response: The agent's response dictionary

        Returns:
            Path to the latest generated image, or None if no image was generated
        """
        if self.agent.was_image_editing_tool_used(agent_response):
            return self.agent.get_latest_generated_image_path()
        return None

    def _update_image_output(self, component, image_path: Optional[str]):
        """Handler to update the image output when a new image is generated."""
        if image_path and os.path.exists(image_path):
            logger.debug(f"🖼️ Updating image output with: {image_path}")
            return image_path
        return None

    def _process_voice_command(
        self, audio: Tuple[int, np.ndarray], input_image: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, AdditionalOutputs], None, None]:
        """
        Process voice command and generate response with optional image output.

        Args:
            audio: Audio data from FastRTC
            input_image: Optional input image from Gradio component

        Yields:
            Tuple of audio chunks and additional outputs
        """
        try:
            # Convert speech to text
            transcript = self.stt_model.stt(audio)
            logger.debug(f"🎤 Transcript: {transcript}")

            # Set the input image for the image editing tool
            if input_image is not None:
                self.agent.set_input_image(input_image)
                logger.debug(
                    f"📷 Input image received: {type(input_image)} - shape: {getattr(input_image, 'shape', 'N/A')}"
                )
            else:
                logger.debug("📷 No input image provided")

            # Process with agent
            logger.debug("🧠 Running agent...")
            agent_response = self.agent.invoke(
                {"messages": [{"role": "user", "content": transcript}]}
            )

            # Extract response text and image path
            response_text = self.agent.get_response_text(agent_response)
            image_path = self._get_generated_image_path(agent_response)

            logger.debug(f"🤖 Agent response: {response_text}")

            # Yield image path first
            yield AdditionalOutputs(image_path)

            # Stream TTS response
            yield from self.tts_model.stream_tts_sync(response_text)

        except Exception as e:
            logger.error(f"❌ Error processing voice command: {e}")
            error_message = "I'm sorry, there was an error processing your request."
            yield AdditionalOutputs(None)
            yield from self.tts_model.stream_tts_sync(error_message)

    def create_stream(self) -> Stream:
        """Create and configure the FastRTC stream."""
        return Stream(
            ReplyOnPause(
                self._process_voice_command,
                # algo_options=self.algo_options,
            ),
            modality="audio",
            mode="send-receive",
            additional_inputs=[gr.Image()],
            additional_outputs=[gr.Image(interactive=False)],
            additional_outputs_handler=self._update_image_output,
            ui_args=self.ui_config,
        )

    def launch(self, server_port: int = 7860, share: bool = False, **kwargs) -> None:
        """
        Launch the application.

        Args:
            server_port: Port to run the server on
            share: Whether to create a shareable link
            **kwargs: Additional arguments for Gradio launch
        """
        stream = self.create_stream()

        logger.info(f"🚀 Launching Voice Image Editing App on port {server_port}")
        stream.ui.launch(server_port=server_port, share=share, **kwargs)


def main():
    """Main entry point for the application."""
    # You can customize the app configuration here
    app = VoiceLiveImageEditor(
        model_provider="ollama",  # or "groq"
        model_name="qwen3:1.7b",  # or "qwen/qwen3-32b" for Groq
        speech_threshold=0.1,
        log_level="DEBUG",
    )

    app.launch()


if __name__ == "__main__":
    main()
