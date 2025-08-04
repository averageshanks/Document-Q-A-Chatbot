import re
import dateparser
from langchain.agents import Tool
from langchain_core.tools import tool


@tool
def validate_email(email: str) -> str:
    """Validates if the input is a correct email address."""
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return "Valid" if re.match(pattern, email) else "Invalid"

@tool
def validate_phone(phone: str) -> str:
    """Validates if the input is a correct phone number."""
    pattern = r"^\+?\d{7,15}$"
    return "Valid" if re.match(pattern, phone) else "Invalid"

@tool
def extract_date(text: str) -> str:
    """Extracts and normalizes date from user input in YYYY-MM-DD format."""
    dt = dateparser.parse(text)
    return dt.strftime("%Y-%m-%d") if dt else "No valid date found"

# def get_tools():
#     return [
#         Tool(
#             name="ValidateEmail",
#             func=validate_email,
#             description="Validates if the input is a correct email address."
#         ),
#         Tool(
#             name="ValidatePhone",
#             func=validate_phone,
#             description="Validates if the input is a correct phone number."
#         ),
#         Tool(
#             name="ExtractDate",
#             func=extract_date,
#             description="Extracts and normalizes date from user input."
#         ),
#     ]

def get_tools():
    return [
        validate_email,
        validate_phone,
        extract_date,
    ]