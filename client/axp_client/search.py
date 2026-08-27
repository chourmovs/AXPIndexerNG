from axp_core.hybrid import search as hybrid_search
def search(con,embedder,query,limit=20): return hybrid_search(con,query,embedder.embed_query(query),limit)
