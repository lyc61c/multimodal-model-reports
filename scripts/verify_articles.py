"""Validate authored document integrity and report unfinished entries separately."""
import hashlib
import re
from collections import Counter
from pathlib import Path
from paper_sources import ROOT, PLAN, dump, read

expected=['论文信息与模型定位','快速概览','整体架构','核心创新与数学原理','数据构建与处理','训练流程与目标','实验设置与主要结果']
results=[]
for entry in read(PLAN)['entries']:
    folder=ROOT/entry['destination']
    metadata=read(folder/'metadata.json')
    status=metadata.get('article_status','planned')
    row={'id':entry['id'],'status':status,'destination':entry['destination'],'issues':[]}
    if status=='complete':
        article=folder/metadata.get('article_file','README.md')
        if not article.exists():
            row['issues'].append('Missing authored article')
        else:
            body=article.read_text(encoding='utf-8-sig')
            headings=re.findall(r'^## \d[.、] ?([^\n]+)$',body,re.M)
            if headings!=expected:
                row['issues'].append('Seven-section headings differ')
            if re.search(r'TODO|TBD|待填写|\ufffd',body):
                row['issues'].append('Placeholder or invalid character')
            if len(re.findall(r'[\u4e00-\u9fff]',body))<500:
                row['issues'].append('Article is too short to be a detailed introduction')
            for link in re.findall(r'\]\(([^)]+)\)',body):
                if '://' not in link and not link.startswith('#'):
                    if not (folder/link.split('#')[0]).resolve().exists():
                        row['issues'].append('Broken relative link: '+link)
            manifest_path=folder/'figures-manifest.json'
            if not manifest_path.exists():
                row['issues'].append('Missing figure provenance manifest')
            else:
                manifest=read(manifest_path)
                row['assets']=len(manifest.get('assets',[]))
                for asset in manifest.get('assets',[]):
                    file=folder/asset['asset_path']
                    if not file.exists():
                        row['issues'].append('Missing original asset: '+asset['asset_path'])
                    elif hashlib.sha256(file.read_bytes()).hexdigest()!=asset['asset_sha256']:
                        row['issues'].append('Asset hash mismatch: '+asset['asset_path'])
                    if asset['asset_path'] not in body:
                        row['issues'].append('Unreferenced asset: '+asset['asset_path'])
                    if not asset.get('source_url'):
                        row['issues'].append('Asset lacks original source URL')
            review_path=folder/'review.json'
            if not review_path.exists():
                row['issues'].append('Missing document review record')
            else:
                review=read(review_path)
                if review.get('article_sha256') and review['article_sha256']!=hashlib.sha256(article.read_bytes()).hexdigest():
                    row['issues'].append('Article changed after registered review')
    results.append(row)
report={'total':len(results),'states':dict(Counter(r['status'] for r in results)),
        'completed_with_integrity_issues':sum(bool(r['issues']) for r in results),
        'original_assets':sum(r.get('assets',0) for r in results),'entries':results,
        'limits':'Structural and provenance validation. Semantic accuracy is checked by the source-reading notes, not inferred from this script.'}
dump(ROOT/'data/article-verification.json',report)
print({k:v for k,v in report.items() if k!='entries'})
for r in results:
    if r['issues']:
        print(r['id'],r['issues'])
raise SystemExit(1 if report['completed_with_integrity_issues'] else 0)
