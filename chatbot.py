from llm_setup import get_llm
from ragQA import create_qa_chain
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# import os
from validation_tools import validate_email, validate_phone, extract_date

load_dotenv()

CHROMA_PATH = "chroma"

def main():
    # Load retriever
    retriever = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ).as_retriever()

    # Load LLM and QA chain
    llm = get_llm()
    qa_chain = create_qa_chain(llm, retriever)



    print("🤖 Chatbot ready! Ask questions about the document, or say something like:")
    print("   'Can you call me?' or 'Book an appointment next Friday'")

    user_data = {"name": "", "email": "", "phone": "", "date": ""}
    collecting_form = False

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        if any(keyword in query.lower() for keyword in ["call me", "book", "appointment", "contact"]):
            collecting_form = True

        if collecting_form:
            if not user_data["name"]:
                user_data["name"] = input("🤖 What's your name? ")

            if not user_data["email"]:
                email = input("🤖 What's your email address? ")
                result = validate_email(email)
                if "Valid" in result:
                    user_data["email"] = email
                else:
                    print("❌ Invalid email. Please try again.")
                    continue

            if not user_data["phone"]:
                phone = input("🤖 What's your phone number? ")
                result = validate_phone(phone)
                if "Valid" in result:
                    user_data["phone"] = phone
                else:
                    print("❌ Invalid phone. Please try again.")
                    continue

            if not user_data["date"]:
                date_input = input("🤖 When would you like to book the appointment? ")
                result = extract_date(date_input)
                if "No valid date found" not in result:
                    user_data["date"] = result
                else:
                    print("❌ Couldn't extract a date. Please rephrase (e.g., 'next Monday').")
                    continue

            # All fields collected
            print("\n✅ Appointment Info Collected:")
            print(user_data)
            print("📞 We will contact you soon!")
            collecting_form = False
            user_data = {"name": "", "email": "", "phone": "", "date": ""}
        else:
            # Run RAG chain for document QA
            answer = qa_chain.invoke(query)
            print("🤖", answer)


if __name__ == "__main__":
    main()
