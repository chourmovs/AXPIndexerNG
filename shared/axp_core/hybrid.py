from .fts import search as lexical_search
from .vectors import search as vector_search

def search(con, query, query_vector, limit=20, rrf_k=60):
    lexical=lexical_search(con,query,limit); vector=vector_search(con,query_vector,limit)
    merged={}
    for kind,rows in [('lexical',lexical),('vector',vector)]:
        for rank,row in enumerate(rows,1):
            item=merged.setdefault(row['chunk_id'],dict(row,lexical_rank=None,vector_rank=None,combined_rank=0.0))
            item[f'{kind}_rank']=rank; item['combined_rank']+=1/(rrf_k+rank)
            if kind=='lexical': item['snippet']=row['snippet']
    return sorted(merged.values(),key=lambda x:(-x['combined_rank'],x['chunk_id']))[:limit]
