
class Configuration:
    embedding_model = None
    splitting_strategy = None
    chunk_size = None
    chunk_overlap = None

    def __init__(self, embedding_model, splitting_strategy, chunk_size, chunk_overlap):
        self.embedding_model = embedding_model
        self.splitting_strategy = splitting_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap