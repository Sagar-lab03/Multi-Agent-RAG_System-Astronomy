from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text = "Tell me about gravitational redshift"
emb = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)

print(emb.shape, emb[:5])