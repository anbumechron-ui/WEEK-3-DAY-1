
import os
from dotenv import load_dotenv

from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

print("NeMo Guardrails configuration loaded successfully!")

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from .env")

# 1. Create the NeMo Guardrails configuration
config = RailsConfig.from_path("./config")
rails = LLMRails(config)


# 2. Custom input guardrail action
def check_topic_allowed(context=None, **kwargs):
    user_message = ""

    if context:
        user_message = str(context.get("user_message", ""))

    allowed_topics = [
        "python", "langchain", "ai", "llm",
        "machine learning", "deep learning",
        "guardrails", "deepeval", "rag"
    ]

    return any(topic in user_message.lower()
               for topic in allowed_topics)


# 3. Custom output check
def check_answer_quality(context=None, **kwargs):
    if not context:
        return True

    answer = str(context.get("bot_message", ""))
    return len(answer.strip()) > 10


rails.register_action(
    check_topic_allowed,
    name="check_topic_allowed"
)

rails.register_action(
    check_answer_quality,
    name="check_answer_quality"
)


# 4. LangChain model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful Python and AI tutor. "
               "Give clear, beginner-friendly answers."),
    ("human", "{question}")
])

chain = prompt | llm


# 5. Run the application through NeMo Guardrails
def ask_ai(question: str) -> str:
    response = rails.generate(
        messages=[
            {"role": "user", "content": question}
        ]
    )

    return response["content"]


if __name__ == "__main__":
    question = input("Ask a Python/AI question: ")
    answer = ask_ai(question)

    print("\nAI Response:")
    print(answer)
