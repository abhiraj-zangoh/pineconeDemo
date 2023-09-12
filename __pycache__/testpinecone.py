from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import pinecone
import os
from tqdm.auto import tqdm
pinecone.init(api_key="acc66926-3691-46f8-baa2-20539770fa16", environment="gcp-starter")


dataset = load_dataset("quora")
dataset = load_dataset('quora', split='train[240000:320000]')
# print(dataset[:5])
questions = []

for record in dataset['questions']:
    questions.extend(record['text'])
  
# remove duplicates
questions = list(set(questions))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device != 'cuda':
    print(f"You are using {device}. This is much slower than using "
          "a CUDA-enabled GPU. If on Colab you can change this by "
          "clicking Runtime > Change runtime type > GPU.")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

query = 'which city is the most populated in the world?'

xq = model.encode(query)
print(xq.shape)
_id = '0'
metadata = {'text': query}

vectors = [(_id, xq, metadata)]
# print('\n'.join(questions[:5]))
# print(len(questions))
index_name = 'semantic-search'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=model.get_sentence_embedding_dimension(),
        metric='cosine'
    )
    
index = pinecone.GRPCIndex(index_name)
batch_size = 128


# check number of records in the index
print(index.describe_index_stats())
query = "which city has the highest population in the world?"

# create the query vector
xq = model.encode(query).tolist()

# now query
xc = index.query(xq, top_k=5, include_metadata=True)
print(xc)