
import json
import os
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI


# 1. Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing from your .env file."
    )


# 2. Define entities and relationships
class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity category")
    description: str = Field(
        description="Description supported by the text"
    )


class Relationship(BaseModel):
    subject: str
    predicate: str = Field(
        description="Relationship label in uppercase"
    )
    object: str
    evidence: str = Field(
        description="Evidence from the original text"
    )


class KnowledgeGraph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# 3. Load the text document
loader = TextLoader(
    str(BASE_DIR / "sample.txt"),
    encoding="utf-8"
)
documents = loader.load()

text_content = "\n".join(
    doc.page_content for doc in documents
)

print("Text document loaded successfully.")


# 4. Extract the knowledge graph with LangChain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

extractor = llm.with_structured_output(KnowledgeGraph)

prompt = f"""
Extract entities and relationships from the text.

Rules:
- Extract entities explicitly supported by the text.
- Give every entity a name, type and description.
- Use uppercase relationship labels with underscores.
- The subject and object must match entity names exactly.
- Include evidence for every relationship.
- Do not invent facts.
- Avoid duplicate entities and relationships.

TEXT:
{text_content}
"""

data = extractor.invoke(prompt)


# 5. Create a directed NetworkX graph
graph = nx.MultiDiGraph()

for entity in data.entities:
    graph.add_node(
        entity.name,
        entity_type=entity.entity_type,
        description=entity.description
    )

# 6. Add relationships as directed edges
entity_names = set(graph.nodes)
added_edges = 0

for rel in data.relationships:
    if (
        rel.subject not in entity_names
        or rel.object not in entity_names
    ):
        print(
            "Skipping relationship with unknown entity:",
            rel.subject, rel.object
        )
        continue

    graph.add_edge(
        rel.subject,
        rel.object,
        relation=rel.predicate.upper(),
        evidence=rel.evidence
    )
    added_edges += 1


# 7. Display all nodes
print("\n=== NETWORKX NODES ===")

for node, attributes in graph.nodes(data=True):
    print(f"\nName: {node}")
    print("Type:", attributes.get("entity_type"))
    print("Description:", attributes.get("description"))


# 8. Display all relationships
print("\n=== NETWORKX RELATIONSHIPS ===")

for source, target, attributes in graph.edges(data=True):
    print(
        f"{source} --[{attributes['relation']}]--> {target}"
    )
    print("Evidence:", attributes.get("evidence"))


# 9. Save graph data as JSON
graph_data = nx.node_link_data(graph)

output_path = BASE_DIR / "knowledge_graph.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(
        graph_data,
        file,
        ensure_ascii=False,
        indent=2
    )

print("\n=== SUMMARY ===")
print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())
print("Relationships added:", added_edges)
print("Graph saved to:", output_path)


# 10. Visualize the graph
if graph.number_of_nodes() > 0:
    plt.figure(figsize=(13, 8))

    positions = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph, positions, node_size=2200
    )

    nx.draw_networkx_labels(
        graph, positions, font_size=8
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.08"
    )

    edge_labels = {
        (source, target, key): attrs["relation"]
        for source, target, key, attrs
        in graph.edges(keys=True, data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.5
    )

    plt.title("Knowledge Graph RAG - NetworkX")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No entities were extracted; nothing to visualize.")
