#!/usr/bin/env python3
"""Reproducible cold/warm and trace overhead benchmark. Synthetic != user corpus.

Run this script with --code-root pointing at each checkout, against the same DB.
--db is strictly read-only; --synthetic creates/updates only the named fixture.
"""
import argparse
import json
import statistics
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path

QUERIES = ('domino sequence', 'densité MTBE', 'zzqv_unobtainium')


class SyntheticEmbedder:
    model_id, dimension = 'test', 3
    def embed_query(self, text):
        text = text.lower()
        return [float('reactor' in text or 'pressure' in text), float('storage' in text), float(len(text) % 7) / 7]


def fixture(path, documents):
    from axp_core.database import connect
    from axp_core.sources import add_source
    from axp_core.metadata import ensure_model
    from axp_core.vectors import upsert
    c = connect(path, dimension=3)
    try:
        if c.execute('SELECT count(*) FROM documents').fetchone()[0]: return
        ensure_model(c, 'test', 3)
        source = add_source(c, '/synthetic/études')['id']
        for number in range(1, documents + 1):
            title = 'domino sequence' if number == 1 else 'densité MTBE' if number == 2 else f'Étude validation {number}'
            name = f'/synthetic/études/{title}.txt'
            c.execute("INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(?,?,?, ?,'.txt',0,100,'',0,?,?)",
                      (number, source, name, name, title, title+'.txt'))
            for no in range(3):
                text = title+' Pressure 12 bar. Density of MTBE 0.74 g/cm3.' if number <= 2 else 'Generic synthetic content.'
                chunk = c.execute('INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,?,?)', (number,no,text)).lastrowid
                upsert(c, chunk, [1,0,0] if number <= 2 else [0,1,0])
        c.commit()
    finally: c.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--db', type=Path)
    target.add_argument('--synthetic', type=Path)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--documents', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--overhead-pairs', type=int, default=0)
    parser.add_argument('--writer-activity', action='store_true', help='Synthetic source-status writes every 20 ms (not the real daemon).')
    args = parser.parse_args()
    if args.writer_activity and not args.synthetic: parser.error('--writer-activity requires --synthetic')
    for folder in ('shared','daemon','client'):
        sys.path.insert(0, str(args.code_root / folder))
    from axp_core.database import SearchReaderPool, connect
    from axp_client.search import search
    if args.trace or args.overhead_pairs:
        from axp_core.tracing import Trace, span
    setup = time.perf_counter()
    if args.synthetic:
        fixture(args.synthetic, args.documents)
        embedder, path = SyntheticEmbedder(), args.synthetic
    else:
        from axp_daemon.embeddings import embedder_for_index
        path = args.db
        c = connect(path, readonly=True)
        try: embedder = embedder_for_index(c)
        finally: c.close()
    print(json.dumps({'kind':'setup', 'synthetic':bool(args.synthetic), 'setup_ms':(time.perf_counter()-setup)*1000,
                      'writer_activity':args.writer_activity, 'code_root':str(args.code_root)}))
    stop, ready = threading.Event(), threading.Event()
    writer_errors = []
    def writes():
        try:
            writer = connect(path)
            try:
                while not stop.is_set():
                    writer.execute("UPDATE sources SET status=CASE status WHEN 'idle' THEN 'scanning' ELSE 'idle' END")
                    writer.commit(); ready.set(); stop.wait(.02)
            finally: writer.close()
        except Exception as exc:
            writer_errors.append(type(exc).__name__); ready.set()
    worker = threading.Thread(target=writes, daemon=True) if args.writer_activity else None
    if worker:
        worker.start()
        if not ready.wait(5): raise RuntimeError('writer not ready')
    pool = SearchReaderPool(path)
    def run(query, tracing=False):
        trace = Trace() if tracing else None
        began = time.perf_counter()
        with trace.bind() if trace else nullcontext():
            with span('Benchmark recherche', 'benchmark.run') if trace else nullcontext():
                with pool.acquire(with_timings=True) as (con, reused, reader):
                    result = search(con, embedder, query, explain=True)
        elapsed = (time.perf_counter()-began)*1000
        if trace:
            trace.finish()
            export = time.perf_counter(); diagnostic = trace.snapshot(); encoded = json.dumps(diagnostic)
            export_ms = (time.perf_counter()-export)*1000
        else: export_ms, encoded = 0, ''
        return {'query':query, 'elapsed_ms':elapsed, 'reader_reused':reused, 'reader':reader,
                'timings':result['timings'], 'ids':[r['document_id'] for r in result['results']],
                'scores':[round(r.get('document_score',0),8) for r in result['results']],
                'result_count':len(result['results']), 'catalog_snapshot_stale':result.get('catalog_snapshot_stale'),
                'trace_spans':len(diagnostic['spans']) if trace else 0, 'trace_export_ms':export_ms,
                'trace_bytes':len(encoded), 'traced':tracing}
    try:
        for query in QUERIES:
            for repeat in range(args.repeat):
                print(json.dumps({'kind':'search','run':repeat, **run(query,args.trace)}, ensure_ascii=False))
        if args.overhead_pairs:
            off, on, export = [], [], []
            for n in range(args.overhead_pairs):
                # Alternate order to avoid always favoring the second run.
                for tracing in ((False,True) if n%2 else (True,False)):
                    result = run(QUERIES[n%len(QUERIES)], tracing)
                    (on if tracing else off).append(result['elapsed_ms'])
                    if tracing: export.append(result['trace_export_ms'])
            print(json.dumps({'kind':'overhead', 'pairs':args.overhead_pairs,
                              'disabled_median_ms':statistics.median(off), 'enabled_median_ms':statistics.median(on),
                              'export_median_ms':statistics.median(export), 'disabled_samples_ms':off,
                              'enabled_samples_ms':on, 'export_samples_ms':export}))
    finally:
        pool.close(); stop.set()
        if worker: worker.join(5)
    if writer_errors: raise RuntimeError(f'writer failed: {writer_errors}')


if __name__ == '__main__': main()
