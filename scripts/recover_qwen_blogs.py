"""Read public blog content using the GET endpoint declared by Qwen's own site."""
import hashlib
import re
import urllib.parse
import urllib.request
import uuid
from paper_sources import CACHE, PLAN, HEADERS, PageParser, dump, read

for row in read(PLAN)['entries']:
    if row['owner']!='root' or 'qwen.ai/blog?' not in (row['paper_url'] or ''):
        continue
    folder=CACHE/row['id']
    previous=read(folder/'source.json')
    path=urllib.parse.parse_qs(urllib.parse.urlsplit(row['paper_url']).query)['id'][0]
    if previous.get('kind')=='html' and previous.get('text_characters',0)>1000 and 'api/v2/article/' not in previous.get('request_url',''):
        continue
    history_path=folder/'source-attempts.json'
    history=read(history_path) if history_path.exists() else []
    history.append(previous)
    dump(history_path,history)
    if 'huggingface.co' in previous.get('request_url',''):
        (folder/'model-card.md').write_bytes((folder/'source.html').read_bytes())
        dump(folder/'model-card-source.json',previous)
    api='https://qwen.ai/api/v2/article/?'+urllib.parse.urlencode({'language':'en-US','path':path,'type':'qwen_ai'})
    result={'id':row['id'],'title':row['title'],'source_url':row['paper_url'],'request_url':api,
            'source_version':row['source_version'],'observed_at':'2026-10-08'}
    try:
        headers={**HEADERS,'X-Request-Id':str(uuid.uuid4())}
        with urllib.request.urlopen(urllib.request.Request(api,headers=headers),timeout=45) as response:
            raw=response.read(10*1024*1024)
        data=__import__('json').loads(raw)
        assert data.get('success'), 'Official API returned no article'
        content=data['data'].get('content') or ''
        assert len(content)>1000,'Official article content absent'
        parser=PageParser()
        parser.feed(content)
        plain=re.sub(r'\n\s*\n+', '\n\n',re.sub(r'[ \t]+',' ',''.join(parser.text))).strip()
        assert len(plain)>500,'Article content is a shell'
        base=row['paper_url']
        match=re.search(r'<link[^>]+rel=["\']?canonical["\']?[^>]+href=["\']?([^\s>"\']+)',content)
        if match:
            base=match.group(1)
        (folder/'source-response.json').write_bytes(raw)
        (folder/'source.html').write_text(content,encoding='utf-8')
        (folder/'source.txt').write_text(plain,encoding='utf-8')
        result.update({'status':'fetched','kind':'html','html_title':data['data'].get('title'),
                       'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                       'content_sha256':hashlib.sha256(content.encode()).hexdigest(),'text_characters':len(plain),
                       'canonical_url':base,'images':[{'url':urllib.parse.urljoin(base,x['src']),'alt':x['alt']} for x in parser.images],
                       'links':[urllib.parse.urljoin(base,x) for x in parser.links],
                       'processing':'Official public API JSON archived; HTML extracted from data.content'})
    except Exception as error:
        result.update({'status':'fetch_error','error':str(error)})
    dump(folder/'source.json',result)
    print({k:result.get(k) for k in ['id','status','html_title','text_characters','error']},flush=True)
