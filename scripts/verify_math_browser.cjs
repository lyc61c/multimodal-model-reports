/* Exercise actual browser DOM parsing and MathJax, including the Gemini regression. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
}
const {chromium} = require(option('--playwright-root', 'playwright'));
const mathjaxRoot = path.resolve(root, option('--mathjax-root', 'node_modules/mathjax-full'));
const input = JSON.parse(fs.readFileSync(path.resolve(root, option('--input', 'tmp/math-manifest.json')), 'utf8'));
if(input.validation.status !== 'passed') throw Error('Exported source verification must pass');
const plan = JSON.parse(fs.readFileSync(path.join(root, 'data/article-plan.json'), 'utf8')).entries;
const manifest = [];
for(const entry of plan) {
  const metadata = JSON.parse(fs.readFileSync(path.join(root, entry.destination, 'metadata.json'), 'utf8'));
  if(metadata.article_status !== 'complete') continue;
  const article = entry.destination.replaceAll('\\', '/') + '/' + (metadata.article_file || 'README.md');
  manifest.push(article + ':' + crypto.createHash('sha256').update(fs.readFileSync(path.join(root, article))).digest('hex'));
}
const hash = crypto.createHash('sha256').update(manifest.sort().join('\n')).digest('hex');
if(hash !== input.validation.article_manifest_sha256 || manifest.length !== input.validation.articles)
  throw Error('Stale browser math input; regenerate the export');
let expressions = input.expressions, publishedRef = null;
if(option('--published-input', null)) {
  const published = JSON.parse(fs.readFileSync(path.resolve(root, option('--published-input')), 'utf8'));
  if(published.differences || published.affected_articles || published.articles !== input.validation.articles)
    throw Error('Published math payloads must match the current article export');
  const plain = values=>values.map(({article,tex,display})=>({article,tex,display}));
  if(JSON.stringify(plain(published.actual)) !== JSON.stringify(plain(expressions)))
    throw Error('Published expression order or payload differs from the current source');
  expressions = published.actual;
  publishedRef = published.ref;
}
const output = path.resolve(root, option('--output', 'data/browser-math-validation.json'));
if(!output.startsWith(root + path.sep)) throw Error('Output must stay inside the repository');

(async()=>{
  const browser = await chromium.launch({channel:option('--channel', 'msedge'), headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1120, height:800}});
    await page.route('http://**', route=>route.abort());
    await page.route('https://**', route=>route.abort());
    await page.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body></body></html>');
    await page.evaluate(()=>{
      window.MathJax = {tex:{packages:['base','ams']}, svg:{fontCache:'none'},
        startup:{typeset:false}, options:{enableMenu:false}};
    });
    await page.addScriptTag({path:path.join(mathjaxRoot, 'es5/tex-svg.js')});
    await page.evaluate(()=>window.MathJax.startup.promise);
    await page.evaluate(()=>document.head.appendChild(window.MathJax.svgStylesheet()));
    const report = await page.evaluate(expressions=>{
      const failures = [], differences = [];
      const htmlText = tex=>{
        const element = document.createElement('div');
        element.innerHTML = tex;
        return element.textContent;
      };
      const render = (tex, display)=>{
        const node = window.MathJax.tex2svg(tex, {display});
        const error = node.querySelector('[data-mjx-error]');
        return {node, error:error?.getAttribute('data-mjx-error') ||
          (node.querySelector('[data-mml-node="merror"]') ? 'MathJax merror' : null)};
      };
      const original = String.raw`\overline{\mathcal L}(n)=-\frac{1}{n}\sum_{t=1}^{n}\log p_\theta(x_t\mid x_{<t}).`;
      const truncated = htmlText(original);
      const negative = render(truncated, true);
      let rendered = 0, preserved = 0;
      const gemini = [];
      for(const expression of expressions) {
        const domTex = htmlText(expression.tex);
        if(domTex !== expression.tex) differences.push({article:expression.article,tex:expression.tex,dom_tex:domTex});
        else preserved++;
        try {
          const result = render(domTex, expression.display);
          if(result.error) failures.push({article:expression.article,tex:domTex,error:result.error});
          else rendered++;
          if(expression.article === 'papers/gemini/document-1855460a9712/README.md' && expression.display)
            gemini.push(result.node.outerHTML);
        } catch(error) {
          failures.push({article:expression.article,tex:domTex,error:error.message});
        }
      }
      document.body.innerHTML = '<h1>Gemini 1.5 · HTML / MathJax 回归检查</h1><h2>旧公式经 HTML 解析</h2>' +
        negative.node.outerHTML + '<h2>当前页面独立公式</h2>' + gemini.join('');
      document.body.style.cssText = 'margin:40px;font-family:Arial,sans-serif';
      return {rendered, preserved, failures, differences,
        regression:{original_tex:original,truncated_tex:truncated,error:negative.error,
          known_failure_detected:negative.error === 'Extra open brace or missing close brace'}};
    }, expressions);
    const result = {
      status:report.failures.length || report.differences.length || !report.regression.known_failure_detected ? 'failed':'passed',
      engine:'Headless Microsoft Edge DOM + MathJax 3.2.2 tex-svg base+ams',
      articles:input.validation.articles, expressions:input.expressions.length,
      article_manifest_sha256:hash, ...report,
      input_source:publishedRef ? 'GitHub Contents rendered HTML payloads' : 'Current article TeX export',
      published_ref:publishedRef,
      scope:'Local browser regression of HTML parsing followed by MathJax rendering, including an exact negative Gemini fixture. This is not a live authenticated GitHub frontend check.'
    };
    fs.writeFileSync(output, JSON.stringify(result,null,2)+'\n');
    await page.screenshot({path:path.join(root,'tmp/gemini-math-browser-regression.png'),fullPage:true});
    console.log(JSON.stringify(result,null,2));
    process.exitCode = result.status === 'passed' ? 0 : 1;
  } finally {
    await browser.close();
  }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
