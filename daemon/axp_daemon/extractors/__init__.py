from . import docx,pdf,pptx,text
EXTRACTORS={'.txt':text.extract,'.md':text.extract,'.markdown':text.extract,'.pdf':pdf.extract,'.docx':docx.extract,'.pptx':pptx.extract}
def extract(path): return EXTRACTORS[path.suffix.lower()](path)
