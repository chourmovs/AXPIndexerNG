from axp_core.database import capability_report,connect
def test_database(tmp_path):
 c=connect(tmp_path/'x.db',dimension=3); assert c.execute('PRAGMA foreign_keys').fetchone()[0]==1; assert capability_report(c)['fts5']
