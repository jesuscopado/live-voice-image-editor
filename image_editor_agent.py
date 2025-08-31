import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger

import image_editing

load_dotenv()


class ImageEditorAgent:
    """
    A voice-controlled image editing agent using LangGraph and AI models.

    This agent can process voice commands to edit images using various AI models
    and maintains conversation memory for better user experience.
    """

    def __init__(
        self,
        model_provider: str = "ollama",
        model_name: str = "qwen3:1.7b",
        thread_id: str = "default_user",
    ):
        """
        Initialize the ImageEditorAgent.

        Args:
            model_provider: Either 'groq' or 'ollama'
            model_name: The specific model to use
            thread_id: Unique identifier for conversation thread
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.thread_id = thread_id
        self.model = self._initialize_model()
        self.edit_image_tool = image_editing.edit_image
        self.tools = [self.edit_image_tool]
        self.memory = InMemorySaver()
        self.system_prompt = """You are Samantha, a helpful assistant with a warm personality.
        You can help with image editing by using your tools.
        Always use the tools when asked to edit images.
        Your output will be converted to audio so avoid using emojis or special characters in your answers.
        Keep your responses friendly and conversational.
        """
        self.agent_config = {"configurable": {"thread_id": self.thread_id}}

        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=self.system_prompt,
            checkpointer=self.memory,
        )

        logger.info(
            f"🤖 ImageEditorAgent initialized with {model_provider}:{model_name}"
        )

    def _initialize_model(self):
        """Initialize the AI model based on provider."""
        if self.model_provider.lower() == "groq":
            return ChatGroq(
                model=self.model_name,
                reasoning_effort="none",
            )
        elif self.model_provider.lower() == "ollama":
            return ChatOllama(
                model=self.model_name,
                reasoning=False,
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def set_input_image(self, image_data):
        """Set the input image for editing operations."""
        image_editing.set_input_image(image_data)

    def invoke(
        self, messages: Dict[str, Any], config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Invoke the agent with a message.

        Args:
            messages: Dictionary containing the messages to process
            config: Optional configuration overrides

        Returns:
            Agent response dictionary
        """
        if config is None:
            config = self.agent_config

        try:
            response = self.agent.invoke(messages, config=config)
            return response
        except Exception as e:
            logger.error(f"❌ Agent invocation failed: {e}")
            raise

    @staticmethod
    def _clean_text_for_tts(text: str) -> str:
        """
        Clean text for TTS by removing emojis and other special characters.

        Args:
            text: Raw text that may contain emojis and special characters

        Returns:
            Cleaned text suitable for TTS processing
        """
        # Remove emojis (Unicode ranges for most emojis)
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002500-\U00002bef"  # chinese char
            "\U00002702-\U000027b0"  # dingbats
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2b55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # diacritical marks
            "\u3030"
            "]+",
            flags=re.UNICODE,
        )

        # Clean the text
        cleaned_text = emoji_pattern.sub("", text)

        # Remove any multiple spaces that might have been left
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)

        # Strip leading/trailing whitespace
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def get_response_text(self, agent_response: Dict[str, Any]) -> str:
        """
        Extract and clean the response text from agent response.

        The text is cleaned to remove emojis and special characters
        that could interfere with TTS processing.
        """
        raw_text = agent_response["messages"][-1].content
        return self._clean_text_for_tts(raw_text)

    def get_latest_generated_image_path(self) -> Optional[str]:
        """
        Get the path to the most recently generated image.

        This method checks if the image editing tool was used and returns
        the latest generated image path if one exists.

        Returns:
            Path to the latest generated image, or None if no images exist
        """
        return image_editing.get_latest_generated_image_path()

    def was_image_editing_tool_used(self, agent_response: Dict[str, Any]) -> bool:
        """
        Check if the image editing tool was used in the agent response.

        Args:
            agent_response: The agent's response dictionary

        Returns:
            True if the image editing tool was called, False otherwise
        """
        for message in agent_response["messages"]:
            # Check if this is a tool message from the edit image tool
            if hasattr(message, "type") and message.type == "tool":
                if (
                    hasattr(message, "name")
                    and self.edit_image_tool.name in message.name
                ):
                    # Check if it was successful (not an error)
                    content = message.content
                    if "✅" in content and "ERROR" not in content:
                        return True
        return False


if __name__ == "__main__":
    # Test the agent
    test_agent = ImageEditorAgent()
    test_agent.set_input_image("data/original.png")
    agent_response = test_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hey Samantha! Can you make him hold a banana",
                }
            ]
        }
    )
    response_text = test_agent.get_response_text(agent_response)
    print(response_text)
