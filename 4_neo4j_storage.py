
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase


# 1. Load configuration
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([
    OPENAI_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
]):
    raise ValueError(
        "Check OPENAI_API_KEY and Neo4j credentials in .env"
    )


# 2. Define graph data structures
class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity category")
    description: str = Field(
        description="Description supported by the text"
    )


class Relationship(BaseModel):
    subject: str
    predicate: str = Field(
        description="Uppercase relationship label"
    )
    object: str
    evidence: str = Field(
        description="Supporting text from the document"
    )


class KnowledgeGraph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# 3. Load text
loader = TextLoader(
    str(BASE_DIR / "sample.txt"),
    encoding="utf-8"
)
documents = loader.load()

text_content = "\n".join(
    doc.page_content for doc in documents
)


# 4. Extract graph data using the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

extractor = llm.with_structured_output(KnowledgeGraph)

prompt = f"""
Extract a knowledge graph from this text.

Rules:
- Extract important entities and their types.
- Extract only explicitly supported relationships.
- Use uppercase relationship labels with underscores.
- Subject and object must match entity names exactly.
- Provide evidence for every relationship.
- Do not invent facts.
- Avoid duplicates.

TEXT:
{text_content}
"""

graph_data = extractor.invoke(prompt)


# 5. Connect to Neo4j AuraDB
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

try:
    driver.verify_connectivity()
    print("Connected to Neo4j AuraDB successfully.")

    with driver.session() as session:

        # Create a uniqueness constraint
        session.run("""
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.name IS UNIQUE
        """).consume()

        # 6. Store entity nodes
        for entity in graph_data.entities:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.entity_type = $entity_type,
                    e.description = $description
            """,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description
            ).consume()

        print(f"Stored {len(graph_data.entities)} entities.")

        # 7. Store relationships
        entity_names = {
            entity.name for entity in graph_data.entities
        }

        stored_relationships = 0

        for rel in graph_data.relationships:

            # Ensure both endpoints exist
            if (
                rel.subject not in entity_names
                or rel.object not in entity_names
            ):
                print("Skipped relationship with unknown entity.")
                continue

            # Only permit safe relationship type names
            predicate = rel.predicate.strip().upper()

            if not re.fullmatch(
                r"[A-Z][A-Z0-9_]*", predicate
            ):
                print("Skipped invalid relationship type.")
                continue

            # Predicate is validated before entering the query.
            query = f"""
                MATCH (s:Entity {{name: $subject}})
                MATCH (o:Entity {{name: $object}})
                MERGE (s)-[r:{predicate}]->(o)
                SET r.evidence = $evidence
            """

            session.run(
                query,
                subject=rel.subject,
                object=rel.object,
                evidence=rel.evidence
            ).consume()

            stored_relationships += 1

        print(
            f"Stored {stored_relationships} relationships."
        )

        # 8. Read back a sample of the stored graph
        result = session.run("""
            MATCH (s:Entity)-[r]->(o:Entity)
            RETURN s.name AS subject,
                   type(r) AS relationship,
                   o.name AS object
            LIMIT 20
        """)

        print("\n=== STORED GRAPH ===")

        for record in result:
            print(
                f"{record['subject']} "
                f"--[{record['relationship']}]--> "
                f"{record['object']}"
            )

finally:
    driver.close()
    print("\nNeo4j connection closed.")
