"""Register verified bytes from a public-source recovery, retaining provenance."""
import argparse
import hashlib
import json
import re
import urllib.parse
from pathlib import Path
from paper_sources import ROOT,CACHE,PLAN,PageParser,dump,read

p=argparse.ArgumentParser()
p.add_argument('--id',required=True)
p.add_argument('--file',required=True)
p.add_argument('--url',required=True)
p.add_argument('--request-url')
p.add_argument('--version',required=True)
a=p.parse_args()
row=next(x for x in read(PLAN)['entries'] if x['id']==a.id)
folder=CACHE/a.id
folder.mkdir(parents=True,exist_ok=True)
old=folder/'source.json'
history=read(folder/'source-attempts.json') if (folder/'source-attempts.json').exists() else []
if old.exists(): history.append(read(old))
dump(folder/'source-attempts.json',history)
data=Path(a.file).read_bytes()
result={'id':a.id,'title':row['title'],'source_url':a.url,'request_url':a.request_url or a.url,'source_version':a.version,'observed_at':'2026-10-08','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'fetched','processing':'Public official source bytes recovered and cached; previous attempts retained.'}
if data.startswith(b'%PDF'):
 from pypdf import PdfReader
 (folder/'source.pdf').write_bytes(data)
 pages=[x.extract_text() or '' for x in PdfReader(folder/'source.pdf').pages]
 plain='\n\n'.join(f'===== PDF PAGE {i+1} =====\n{t}' for i,t in enumerate(pages))
 dump(folder/'pages.json',[{'pdf_page':i+1,'text':t} for i,t in enumerate(pages)])
 result.update({'kind':'pdf','pages':len(pages),'text_characters':len(plain)})
else:
 html=data.decode('utf-8-sig')
 parser=PageParser()
 parser.feed(html)
 plain=re.sub(r'\n\s*\n+','\n\n',re.sub(r'[ \t]+',' ',''.join(parser.text))).strip()
 assert len(plain)>500,'Refuse to register a shell or empty page'
 (folder/'source.html').write_bytes(data)
 result.update({'kind':'html','html_title':''.join(parser.title),'text_characters':len(plain),'images':[{'url':urllib.parse.urljoin(a.url,x['src']),'alt':x['alt']} for x in parser.images],'links':[urllib.parse.urljoin(a.url,x) for x in parser.links]})
(folder/'source.txt').write_text(plain,encoding='utf-8')
dump(folder/'source.json',result)
print({k:result.get(k) for k in ['id','status','kind','pages','text_characters','source_url']})
