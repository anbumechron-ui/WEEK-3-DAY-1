
from langchain_community.document_loaders import TextLoader

# Load the text file
loader = TextLoader("sample.txt", encoding="utf-8")

# Convert the text into LangChain documents
documents = loader.load()

# Display the results
print("Text file loaded successfully!")
print("Number of documents:", len(documents))

print("\nDocument content:")
print(documents[0].page_content)

print("\nDocument metadata:")
print(documents[0].metadata)
