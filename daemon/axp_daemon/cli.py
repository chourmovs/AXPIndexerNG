import argparse,json,os
from axp_core.database import capability_report,connect
from axp_core.metadata import ensure_model
from .embeddings import Embedder
from .indexer import scan

def main(argv=None):
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
 for name in ('health','status'): q=sub.add_parser(name); q.add_argument('--db',required=True)
 q=sub.add_parser('scan'); q.add_argument('--db',required=True); q.add_argument('--root',required=True); q.add_argument('--model-cache'); q.add_argument('--allow-download',action='store_true')
 a=p.parse_args(argv)
 if a.cmd=='health':
  con=connect(a.db); print(json.dumps(capability_report(con))); return
 if a.cmd=='status':
  con=connect(a.db); print(json.dumps({'documents':con.execute('SELECT count(*) FROM documents').fetchone()[0],'chunks':con.execute('SELECT count(*) FROM chunks').fetchone()[0]})); return
 e=Embedder(cache_dir=a.model_cache or os.getenv('FASTEMBED_CACHE_PATH'),local_only=not a.allow_download); con=connect(a.db,dimension=e.dimension); ensure_model(con,e.model_id,e.dimension); print(json.dumps(scan(con,a.root,e)))
