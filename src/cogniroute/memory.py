import chromadb

# Initialize a local ChromaDB client
client = chromadb.PersistentClient(path="./chroma_data")

# Create or get a collection
collection = client.get_or_create_collection(name="user_memory")

def add_memory(query: str, route: str, user_id: str):
    """Stores what the user asked, tagged with their user_id."""
    count = collection.count()
    collection.add(
        documents=[query],
        metadatas=[{"route": route, "user_id": user_id}],
        ids=[f"memory-{count}-{user_id}"] # Unique ID
    )

def get_relevant_memories(query: str, user_id: str, n_results: int = 2) -> str:
    """Searches the database for past queries similar to the current one FOR THIS SPECIFIC USER."""
    if collection.count() == 0:
        return ""
    
    # The 'where' clause is the multi-tenancy filter!
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        where={"user_id": user_id}
    )
    
    memories = results['documents'][0]
    if not memories:
        return ""
        
    formatted = "Previous context from the user:\n"
    for mem in memories:
        formatted += f"- {mem}\n"
    return formatted