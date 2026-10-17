
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI


# 1. Load API key from the project folder
project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is missing from .env")


# 2. Define the entity structure
class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity category")
    description: str = Field(
        description="Description supported by the text"
    )


# 3. Define the relationship structure
class Relationship(BaseModel):
    subject: str = Field(
        description="Name of the starting entity"
    )
    predicate: str = Field(
        description="Short relationship label in uppercase"
    )
    object: str = Field(
        description="Name of the connected entity"
    )
    evidence: str = Field(
        description="Text supporting this relationship"
    )


# 4. Define the complete graph extraction result
class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(
        description="Unique entities in the document"
    )
    relationships: list[Relationship] = Field(
        description="Relationships supported by the document"
    )


# 5. Load the text file
loader = TextLoader(
    str(project_dir / "sample.txt"),
    encoding="utf-8"
)
documents = loader.load()

text_content = "\n".join(
    document.page_content for document in documents
)


# 6. Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

graph_extractor = llm.with_structured_output(KnowledgeGraph)


# 7. Ask the LLM to extract entities and relationships
prompt = f"""
Extract a knowledge graph from the text below.

Rules:
1. Extract entities explicitly mentioned in the text.
2. Extract only relationships supported by the text.
3. Use concise, uppercase predicates such as LOCATED_IN,
   IMPLEMENTS, WORKS_WITH, and IMPROVES.
4. The subject and object must match entity names in the
   extracted entity list.
5. Do not invent facts or infer unsupported relationships.
6. Include a short evidence quotation for each relationship.
7. Avoid duplicate entities and duplicate relationships.
8. Return an empty list if no valid relationships exist.

TEXT:
{text_content}
"""

result = graph_extractor.invoke(prompt)


# 8. Display entities
print("\n=== ENTITIES ===")

for entity in result.entities:
    print(
        f"{entity.name} | {entity.entity_type} | "
        f"{entity.description}"
    )


# 9. Validate and display relationships
entity_names = {entity.name for entity in result.entities}

print("\n=== RELATIONSHIPS ===")

valid_relationships = []

for rel in result.relationships:
    if rel.subject not in entity_names:
        print("Skipped: unknown subject:", rel.subject)
        continue

    if rel.object not in entity_names:
        print("Skipped: unknown object:", rel.object)
        continue

    valid_relationships.append(rel)

for rel in valid_relationships:
    print(f"\n({rel.subject})")
    print(f"    --[{rel.predicate}]--> ({rel.object})")
    print(f"    Evidence: {rel.evidence}")


# 10. Summary
print("\n=== SUMMARY ===")
print("Total entities:", len(result.entities))
print("Total valid relationships:", len(valid_relationships))
