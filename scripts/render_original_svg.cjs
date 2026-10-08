// Render unchanged publisher SVG bytes for local visual inspection.
const path = require('node:path');
const modules = process.env.PAPER_NODE_MODULES || 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const sharp = require(path.join(modules, 'sharp'));
const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node render_original_svg.cjs input.svg output.png');
sharp(input, { density: 144 }).flatten({ background: '#ffffff' }).png().toFile(output).catch(error => {
  process.stderr.write(error.message + '\n');
  process.exitCode = 1;
});
