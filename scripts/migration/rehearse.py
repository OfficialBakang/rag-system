import os
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
new = Path(sys.executable)
legacy = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    env = os.environ.copy()
    for key in list(env):
        if "API_KEY" in key:
            env.pop(key)
    env.pop("SSLKEYLOGFILE", None)
    env["PYTHONPATH"] = str(repo)
    program = """
import socket
def blocked(*a,**kw): raise RuntimeError('Network prohibited')
socket.socket.connect=blocked
socket.getaddrinfo=blocked
import chromadb
from chromadb.config import Settings
from scripts.migration.export_chroma import export
c=chromadb.PersistentClient(path='legacy',settings=Settings(anonymized_telemetry=False))
col=c.create_collection('rag_documents',embedding_function=None,metadata={'hnsw:space':'cosine'})
col.upsert(ids=['a','b'],embeddings=[[3.,4.],[1.,0.]],documents=['alpha','beta'],metadatas=[dict(doc_id='doc',chunk_index=i,token_count=1,source='fixture',flag=True) for i in range(2)])
assert col.query(query_embeddings=[[3.,4.]],n_results=2)['ids'][0]==['a','b']
c.close()
export('legacy','bundle','rag_documents')
"""
    subprocess.run(
        [str(legacy), "-c", program], cwd=root, env=env, check=True, timeout=60
    )
    program = """
from scripts.migration.import_bundle import import_bundle, read_bundle
from core.vector_store import VectorStore
rows=read_bundle('bundle')
import_bundle('bundle','target')
import_bundle('bundle','target')
s=VectorStore('target')
hits=s.query([3.,4.],top_k=2,where={'source':'fixture'})
assert [h.chunk.chunk_id for h in hits]==['a','b']
assert [h.score for h in hits]==[1.,0.6]
assert {h.chunk.chunk_id for h in hits}=={r['id'] for r in rows}
for h in hits:
 r=next(r for r in rows if r['id']==h.chunk.chunk_id)
 assert h.chunk.text==r['text'] and h.chunk.metadata==r['metadata'] and h.chunk.doc_id==r['metadata']['doc_id']
s.close()
s=VectorStore('target')
assert s.count()==2 and s.list_sources()==['fixture']
assert s.delete_document('doc')==2 and s.delete_document('doc')==0
s.close()
"""
    subprocess.run([str(new), "-c", program], cwd=root, env=env, check=True, timeout=60)
print(
    "Synthetic legacy export, idempotent import, IDs/text/typed metadata, filtered cosine parity, restart and deletion passed."
)
