import pinecone
import time
# Initialize Pinecone
pinecone.init(api_key="acc66926-3691-46f8-baa2-20539770fa16", environment="gcp-starter")

index_name = "example-index"

# Delete the index if it exists (optional)

# pinecone.delete_index(index_name)

# Create the index
# pinecone.create_index(index_name, dimension=8, metric="euclidean")
index = pinecone.Index("example-index")

# Insert data into the index
index.upsert([
    ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    ("B", [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
    ("C", [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    ("D", [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
    ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
])

# Describe index statistics


# Define a different query vector
data = index.describe_index_stats()
# print(data)
# Perform the query
query_response=index.query(
  vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
  top_k=3,
  include_values=True
)
# Print query response
# print(query_response)
for matches in query_response.matches:
    print(matches.score)

# List active indexes
active_indexes = pinecone.list_indexes()
print(active_indexes)
