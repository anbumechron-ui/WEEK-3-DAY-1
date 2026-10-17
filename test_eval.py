
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from main import ask_ai

question = "What is LangChain in Python?"
answer = ask_ai(question)

print("Question:", question)
print("Answer:", answer)

test_case = LLMTestCase(
    input=question,
    actual_output=answer
)

relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
    include_reason=True
)

evaluate(
    test_cases=[test_case],
    metrics=[relevancy]
)
