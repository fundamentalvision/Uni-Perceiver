
from uniperceiver.utils.registry import Registry

EMBEDDING_REGISTRY = Registry("EMBEDDING")
EMBEDDING_REGISTRY.__doc__ = """
Registry for embedding
"""

def build_embeddings(cfg, name):
    embeddings = None if name.lower() == "none" else EMBEDDING_REGISTRY.get(name)(cfg)
    return embeddings