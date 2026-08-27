from axp_core.database import connect
from axp_core.hybrid import search
from axp_daemon.indexer import scan
from conftest import FakeEmbedder
def test_e2e_delete(tmp_path):
 root=tmp_path/'root';root.mkdir();r=root/'reactor.txt';r.write_text('The pressure control valve regulates reactor pressure.');(root/'warehouse.txt').write_text('Storage and logistics inventory.')
 e=FakeEmbedder();c=connect(tmp_path/'x.db',dimension=e.dimension);scan(c,root,e);assert search(c,'reactor pressure',e.embed_query('reactor pressure'))[0]['path'].endswith('reactor.txt');r.unlink();scan(c,root,e);assert not c.execute("select 1 from documents where path like '%reactor.txt'").fetchall();assert not search(c,'reactor',e.embed_query('reactor'))
