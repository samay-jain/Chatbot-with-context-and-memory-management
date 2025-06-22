import json
import requests
from typing import Dict, List
import logging
from datetime import datetime
import uuid

def setup_logging():
    """
    Configure logging to save logs in both JSON format (for file) and readable format (for console).
    """
    logger = logging.getLogger("chatbot")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler logs raw JSON strings to a file
        file_handler = logging.FileHandler("chatbot_logs.json")
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler logs human-readable logs to the terminal
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


class ChatBot:
    def __init__(self):
        """
        Initialize the chatbot with a new session ID, logging setup, model name, and initial messages.
        """
        self.logger = setup_logging()
        self.session_id = str(uuid.uuid4())
        self.model_name = "llama3.2"
        self.messages = self.create_initial_messages()

    @staticmethod
    def create_initial_messages() -> List[Dict[str, str]]:
        """
        Returns the initial system message to establish context for the chatbot.
        """
        return [{"role": "system", "content": "Hello, how can I help you today?"}]

    def chat(self, user_input: str) -> str:
        """
        Sends the user input to the LLM backend, receives a response, and logs the conversation.

        Args:
            user_input (str): The input from the user.

        Returns:
            str: The model-generated response or an error message.
        """
        try:
            # Log the user's message
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "type": "user_input",
                "user_input": user_input,
                "metadata": {"session_id": self.session_id, "model": self.model_name}
            }
            self.logger.info(json.dumps(log_entry))

            # Add user message to history
            self.messages.append({"role": "user", "content": user_input})

            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": self.messages
            }

            # Make the POST request to the local API and time the request
            start_time = datetime.now()
            response = requests.post("http://localhost:11434/api/chat", json=payload, stream=True)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()

            # Check if the request was successful
            if response.status_code == 200:
                full_response = ""

                # Parse each line in the streaming response
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_data = json.loads(line)
                            if "message" in json_data and "content" in json_data["message"]:
                                full_response += json_data["message"]["content"]
                        except json.JSONDecodeError:
                            print(f"\nFailed to parse line: {line}")

                # Log the assistant's response
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "INFO",
                    "type": "model_response",
                    "response_content": full_response,
                    "metadata": {
                        "session_id": self.session_id,
                        "model": self.model_name,
                        "response_time": response_time,
                        "tokens_used": None  # Placeholder
                    }
                }
                self.logger.info(json.dumps(log_entry))

                # Append the assistant's response to conversation history
                self.messages.append({"role": "assistant", "content": full_response})
                return full_response

            else:
                # Log API error response (non-200)
                error_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": "ERROR",
                    "type": "http_error",
                    "error_message": response.text,
                    "metadata": {
                        "session_id": self.session_id,
                        "model": self.model_name,
                        "status_code": response.status_code
                    }
                }
                self.logger.error(json.dumps(error_entry))
                return f"Error: {response.status_code} - {response.text}"

        except Exception as e:
            # Catch-all error logging for exceptions (e.g., connection issues)
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "type": "exception",
                "error_message": str(e),
                "metadata": {
                    "session_id": self.session_id,
                    "model": self.model_name,
                }
            }
            self.logger.error(json.dumps(error_entry))
            return f"Sorry, something went wrong: {str(e)}"

    def summarize_messages(self) -> List[Dict[str, str]]:
        """
        Summarizes the last few messages to reduce token usage.

        Returns:
            List[Dict[str, str]]: A summarized version of the conversation history.
        """
        summary = "Previous conversation summarized: " + " ".join(
            [m["content"][:50] + "..." for m in self.messages[-5:]]
        )
        return [{"role": "system", "content": summary}] + self.messages[-5:]

    def save_conversation(self, filename: str = "conversation.json"):
        """
        Saves the current conversation to a JSON file.

        Args:
            filename (str): The name of the file to save the conversation in.
        """
        with open(filename, "w") as f:
            json.dump(self.messages, f)

    def load_conversation(self, filename: str = "conversation.json"):
        """
        Loads conversation history from a file if it exists.

        Args:
            filename (str): The name of the file to load conversation history from.
        """
        try:
            with open(filename, "r") as f:
                self.messages = json.load(f)
        except FileNotFoundError:
            print(f"No conversation file found at {filename}")
            self.messages = self.create_initial_messages()


def main():
    """
    Entry point for the chatbot interface.
    Handles user input, conversation flow, and special commands like save/load/summary.
    """
    chatbot = ChatBot()

    print("\n=== Chat Session Started ===")
    print("Type 'exit' to end the conversation")
    print(f"Session ID: {chatbot.session_id}\n")

    print("Available commands:")
    print("- 'save': Save conversation")
    print("- 'load': Load conversation")
    print("- 'summary': Summarize conversation")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            print("Goodbye! 👋")
            break
        elif user_input.lower() == 'save':
            chatbot.save_conversation()
            print("Conversation saved!")
            continue
        elif user_input.lower() == 'load':
            chatbot.load_conversation()
            print("Conversation loaded!")
            continue
        elif user_input.lower() == 'summary':
            chatbot.messages = chatbot.summarize_messages()
            print("Conversation summarized!")
            continue

        # Send input to chatbot and display response
        response = chatbot.chat(user_input)
        print(f"\nAssistant: {response}")

        # Auto-summarize long conversations
        if len(chatbot.messages) > 10:
            chatbot.messages = chatbot.summarize_messages()
            print("\n(Conversation automatically summarized)")

# Run chatbot in CLI
if __name__ == "__main__":
    main()
