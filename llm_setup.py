import os
import getpass
from langchain.chat_models import init_chat_model


def get_llm():
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    try:
        # load environment variables from .env file (requires `python-dotenv`)
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

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
    # Initialize the LLM with callback manager and verbose for debugging
    llm = init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai",
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm
    