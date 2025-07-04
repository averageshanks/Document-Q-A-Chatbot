import os
import getpass
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# Load environment variables from .env file
load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

def get_llm():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    llm = init_chat_model(
        "gemini-2.0-flash", model_provider="google_genai", temperature=0
    )
    return llm

# if __name__ == "__main__":
#     llm = get_llm()
#     print("LLM initialized successfully.")
#     # You can add more functionality here if needed