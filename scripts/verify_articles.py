"""Validate authored document integrity and report unfinished entries separately."""
import hashlib
import re
import xml.etree.ElementTree as ET
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
                    processing=asset.get('processing',{})
                    if processing.get('method')=='full_pdf_page_render':
                        row['issues'].append('Full-page screenshot must be replaced by a figure/table crop: '+asset['asset_path'])
                    if processing.get('method')=='pdf_region_render':
                        box=processing.get('crop_bbox')
                        size=processing.get('page_size')
                        if not box or not size:
                            row['issues'].append('PDF region lacks explicit crop coordinates or page dimensions: '+asset['asset_path'])
                        else:
                            x0,top,x1,bottom=box
                            width,height=size
                            if not (0 <= x0 < x1 <= width+0.01 and 0 <= top < bottom <= height+0.01):
                                row['issues'].append('PDF region lies outside source page: '+asset['asset_path'])
                            if (x1-x0)/width>=0.98 and (bottom-top)/height>=0.98:
                                row['issues'].append('Near-full-page screenshot must be replaced: '+asset['asset_path'])
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
            derived_path=folder/'derived-figures-manifest.json'
            row['derived_assets']=0
            if derived_path.exists():
                derived=read(derived_path)
                row['derived_assets']=len(derived.get('assets',[]))
                originals={asset['asset_path'] for asset in read(manifest_path).get('assets',[])} if manifest_path.exists() else set()
                for asset in derived.get('assets',[]):
                    label=asset.get('asset_path','')
                    file=(folder/label).resolve()
                    if not label or not file.is_relative_to(folder.resolve()) or label in originals:
                        row['issues'].append('Derived asset path is invalid or mixed with original figures: '+label)
                        continue
                    if asset.get('origin')!='self_drawn' or asset.get('kind') not in {'architecture','public_interface','method_workflow'}:
                        row['issues'].append('Derived asset lacks an explicit origin or diagram type: '+label)
                    sources=asset.get('sources',[])
                    if not sources or any(not source.get('url','').startswith('https://') or not source.get('locator') for source in sources):
                        row['issues'].append('Derived asset lacks located primary sources: '+label)
                    if not asset.get('scope') or label not in body:
                        row['issues'].append('Derived asset scope or article reference is missing: '+label)
                    if not file.is_file():
                        row['issues'].append('Missing derived asset: '+label)
                        continue
                    if hashlib.sha256(file.read_bytes()).hexdigest()!=asset.get('asset_sha256'):
                        row['issues'].append('Derived asset hash mismatch: '+label)
                    if file.suffix.lower()!='.svg':
                        row['issues'].append('Architecture diagrams must use native SVG: '+label)
                        continue
                    try:
                        svg=ET.fromstring(file.read_bytes())
                        ns={'s':'http://www.w3.org/2000/svg'}
                        if not svg.findtext('s:title',namespaces=ns) or not svg.findtext('s:desc',namespaces=ns) or not svg.get('viewBox'):
                            row['issues'].append('Derived SVG lacks an accessible title, description or viewBox: '+label)
                        if any(element.tag.rsplit('}',1)[-1] in {'script','foreignObject','image'} for element in svg.iter()):
                            row['issues'].append('Derived SVG contains executable or external raster content: '+label)
                        if any(key.rsplit('}',1)[-1].lower().startswith('on') or (key.rsplit('}',1)[-1]=='href' and not value.startswith('#'))
                               for element in svg.iter() for key,value in element.attrib.items()):
                            row['issues'].append('Derived SVG contains an external or executable reference: '+label)
                    except ET.ParseError:
                        row['issues'].append('Derived SVG is not valid XML: '+label)
            if not review_path.exists():
                row['issues'].append('Missing document review record')
            else:
                review=read(review_path)
                if review.get('article_sha256') and review['article_sha256']!=hashlib.sha256(article.read_bytes()).hexdigest():
                    row['issues'].append('Article changed after registered review')
    results.append(row)
report={'total':len(results),'states':dict(Counter(r['status'] for r in results)),
        'completed_with_integrity_issues':sum(bool(r['issues']) for r in results),
        'original_assets':sum(r.get('assets',0) for r in results),
        'derived_assets':sum(r.get('derived_assets',0) for r in results),'entries':results,
        'limits':'Structural and provenance validation. Semantic accuracy is checked by the source-reading notes, not inferred from this script.'}
dump(ROOT/'data/article-verification.json',report)
print({k:v for k,v in report.items() if k!='entries'})
for r in results:
    if r['issues']:
        print(r['id'],r['issues'])
raise SystemExit(1 if report['completed_with_integrity_issues'] else 0)
