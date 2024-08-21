# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3
# The sentences to encode
sentences1 = "java"
sentences2 = "java"

# STEP 4
# 2. Calculate embeddings by calling model.encode()
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)
# [3, 384]

# STEP 5
# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
