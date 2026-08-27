from axp_daemon.chunker import chunk_text
from axp_daemon.extractors.text import extract
def test_text_and_determinism(tmp_path):
 p=tmp_path/'a.txt';p.write_text('one two three');assert extract(p)[0][0]=='one two three';assert chunk_text(p.read_text())==chunk_text(p.read_text())
