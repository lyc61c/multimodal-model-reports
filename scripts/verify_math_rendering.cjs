/* Verify exact TeX preservation through GFM, then compile it without operatorname. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const args = process.argv.slice(2);
function option(name, fallback) {
  const index = args.indexOf(name);
  return index < 0 ? fallback : args[index + 1];
}
const root = path.resolve(__dirname, '..');
const mathjaxRoot = option('--mathjax-root', 'mathjax-full');
const markedRoot = option('--marked-root', 'marked');
const modulePath = name => mathjaxRoot === 'mathjax-full' ? 'mathjax-full/' + name : path.resolve(root, mathjaxRoot, name);
const {mathjax} = require(modulePath('js/mathjax.js'));
const {TeX} = require(modulePath('js/input/tex.js'));
const {SVG} = require(modulePath('js/output/svg.js'));
const {liteAdaptor} = require(modulePath('js/adaptors/liteAdaptor.js'));
const {RegisterHTMLHandler} = require(modulePath('js/handlers/html.js'));
require(modulePath('js/input/tex/ams/AmsConfiguration.js'));
const {marked} = require(markedRoot === 'marked' ? markedRoot : path.resolve(root,markedRoot));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document('', {InputJax:new TeX({packages:['base','ams']}),OutputJax:new SVG({fontCache:'none'})});
const input = JSON.parse(fs.readFileSync(path.resolve(root,option('--input','tmp/math-manifest.json')),'utf8'));
if (input.validation.status !== 'passed') throw Error('Source syntax validation must pass first');
const entries=JSON.parse(fs.readFileSync(path.join(root,'data/article-plan.json'),'utf8')).entries;
const currentManifest=[];
for(const entry of entries){
  const metadata=JSON.parse(fs.readFileSync(path.join(root,entry.destination,'metadata.json'),'utf8'));
  if(metadata.article_status!=='complete') continue;
  const article=entry.destination.replaceAll('\\','/')+'/'+(metadata.article_file||'README.md');
  const hash=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,article))).digest('hex');
  currentManifest.push(article+':'+hash);
}
const currentHash=crypto.createHash('sha256').update(currentManifest.sort().join('\n')).digest('hex');
if(currentManifest.length!==input.validation.articles || currentHash!==input.validation.article_manifest_sha256)
  throw Error('Article files changed after TeX export; regenerate the math manifest before rendering');
const failures = [], gfmFailures = [], artifacts = [], byArticle = new Map();
const htmlFailures = [];
for (const expression of input.expressions) {
  if (!byArticle.has(expression.article)) byArticle.set(expression.article,[]);
  byArticle.get(expression.article).push(expression);
}

function collectMath(tokens, result) {
  if (!Array.isArray(tokens)) return;
  for(let i=0;i<tokens.length;i++) {
    const token = tokens[i];
    if(token.type==='codespan' && tokens[i-1]?.type==='text' && tokens[i+1]?.type==='text'
       && tokens[i-1].text.endsWith('$') && tokens[i+1].text.startsWith('$'))
      result.push({tex:token.text.trim(),display:false,delimiter:'guarded'});
    if(token.type==='code' && token.lang==='math')
      result.push({tex:token.text.trim(),display:true,delimiter:'math_fence'});
    if(token.tokens) collectMath(token.tokens,result);
    if(token.items) token.items.forEach(item=>collectMath(item.tokens,result));
    if(token.header) token.header.forEach(cell=>collectMath(cell.tokens,result));
    if(token.rows) token.rows.forEach(row=>row.forEach(cell=>collectMath(cell.tokens,result)));
  }
}
let guardedPreserved = 0, displayPreserved = 0;
let htmlPreserved = 0;
function htmlText(tex) {
  const html = adaptor.parse('<html><body><div>' + tex + '</div></body></html>', 'text/html');
  return adaptor.textContent(adaptor.body(html));
}
for(const [article,expressions] of byArticle) {
  const body = fs.readFileSync(path.join(root,article),'utf8').replaceAll('\r\n','\n');
  const actual = [];
  collectMath(marked.lexer(body,{gfm:true}),actual);
  const expected = expressions.map(({tex,display,delimiter})=>({tex,display,delimiter}));
  if(JSON.stringify(actual)!==JSON.stringify(expected)) {
    gfmFailures.push({article,expected:expected.length,actual:actual.length,
      firstDifference:expected.findIndex((value,index)=>JSON.stringify(value)!==JSON.stringify(actual[index]))});
    continue;
  }
  guardedPreserved += actual.filter(value=>!value.display).length;
  displayPreserved += actual.filter(value=>value.display).length;
  // Compile the Markdown parser's payload, rather than silently falling back to source TeX.
  for(const expression of actual) {
    try {
      if (/\\operatorname\b/.test(expression.tex)) throw Error('operatorname is not permitted in this compatibility profile');
      // Test the DOM boundary too: e.g. x_{<t} can become x_{ if reused as HTML.
      const domTex = htmlText(expression.tex);
      if(domTex!==expression.tex) {
        htmlFailures.push({article,tex:expression.tex,dom_tex:domTex});
        throw Error('TeX changed after HTML parsing');
      }
      htmlPreserved++;
      const svg = adaptor.outerHTML(document.convert(domTex,{display:expression.display}));
      const error = svg.match(/data-mjx-error="([^"]+)"/);
      if(error || svg.includes('data-mml-node="merror"')) throw Error(error?.[1]||'MathJax merror');
      artifacts.push({article,...expression,svg});
    } catch(error) {failures.push({article,tex:expression.tex,error:error.message});}
  }
}
// A validator must reject the exact previously missed failure, not only pass new input.
const geminiFixture=String.raw`\overline{\mathcal L}(n)=-\frac{1}{n}\sum_{t=1}^{n}\log p_\theta(x_t\mid x_{<t}).`;
const truncatedFixture=htmlText(geminiFixture);
const negativeSvg=adaptor.outerHTML(document.convert(truncatedFixture,{display:true}));
const negativeDetected=truncatedFixture!==geminiFixture && negativeSvg.includes('Extra open brace or missing close brace');
if(!negativeDetected) throw Error('Gemini HTML-truncation regression did not reproduce the known brace error');
const result = {
  status:failures.length||gfmFailures.length||htmlFailures.length?'failed':'passed',
  engine:'MathJax 3.2.2 base+ams with operatorname excluded',
  equations:input.expressions.length,
  failures,
  gfm_protected_inline_expressions_preserved:guardedPreserved,
  gfm_display_expressions_preserved:displayPreserved,
  gfm_failures:gfmFailures,
  html_roundtrip_expressions_preserved:htmlPreserved,
  html_failures:htmlFailures,
  regression_fixture:{article:'papers/gemini/document-1855460a9712/README.md',
    original_tex:geminiFixture,truncated_tex:truncatedFixture,
    expected_error:'Extra open brace or missing close brace',known_failure_detected:negativeDetected},
  article_manifest_sha256:input.validation.article_manifest_sha256,
  scope:'All inline and fenced display TeX is compared with the full GFM token stream, tested for HTML roundtrip preservation, then compiled. A negative Gemini fixture must reproduce the exact brace error. This is not a live GitHub frontend check or a scientific correctness review.'
};
fs.writeFileSync(path.join(root,'data/math-validation.json'),JSON.stringify(result,null,2)+'\n');
fs.writeFileSync(path.join(root,'tmp/math-rendered.json'),JSON.stringify(artifacts));
console.log(JSON.stringify(result,null,2));
process.exitCode=result.status==='passed'?0:1;
