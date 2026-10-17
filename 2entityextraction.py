
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI

# 1. Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Add it to your .env file."
    )


# 2. Define the structure of an entity
class Entity(BaseModel):
    name: str = Field(
        description="The exact name of the entity in the text"
    )
    entity_type: str = Field(
        description=(
            "Category such as PERSON, ORGANIZATION, "
            "STATE, DISTRICT, PROJECT or CONCEPT"
        )
    )
    description: str = Field(
        description="A short description supported by the text"
    )


# 3. Define the expected extraction result
class EntityExtraction(BaseModel):
    entities: list[Entity] = Field(
        description="All relevant unique entities found in the text"
    )


# 4. Load the text document
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

text_content = "\n".join(
    document.page_content for document in documents
)

# 5. Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 6. Ask the model to return structured entities
entity_extractor = llm.with_structured_output(EntityExtraction)

prompt = f"""
Extract important entities from the text below.

Instructions:
- Identify people, organisations, departments, locations,
  projects and important concepts.
- Use concise, meaningful entity names.
- Assign an appropriate entity type.
- Do not invent entities or facts.
- Include only entities supported by the text.
- Avoid duplicate entities.
- Do not extract relationships in this step.

TEXT:
{text_content}
"""

# 7. Extract entities
result = entity_extractor.invoke(prompt)

# 8. Display the extracted entities
print("\nENTITY EXTRACTION RESULTS")
print("=" * 50)

for index, entity in enumerate(result.entities, start=1):
    print(f"\nEntity {index}")
    print("Name       :", entity.name)
    print("Type       :", entity.entity_type)
    print("Description:", entity.description)

print("\nTotal entities:", len(result.entities))
