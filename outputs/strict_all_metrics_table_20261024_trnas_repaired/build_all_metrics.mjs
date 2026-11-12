import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from 'file:///C:/Users/22465/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(?=[A-Za-z]:)/, ''));
const root = path.resolve(here, '../..');
const sources = path.join(here, 'sources');
const read = async p => fs.readFile(p, 'utf8');

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const headers = rows.shift();
  return rows.filter(r => r.length > 1).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ''])));
}
function csvCell(v) {
  const s = v == null ? '' : String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
}
function assert(x, message) { if (!x) throw Error(message); }

const strict = parseCsv(await read(path.join(sources, 'ranking', 'strict_ranking.csv')));
const excluded = parseCsv(await read(path.join(sources, 'ranking', 'excluded_metrics.csv')));
const raw = parseCsv(await read(path.join(sources, 'metric_values_original_strict.csv')));
const repair = parseCsv(await read(path.join(sources, 'repair_rows.csv')));
const verification = JSON.parse(await read(path.join(sources, 'release_verification.json')));
const repairManifest = JSON.parse(await read(path.join(sources, 'repair_manifest.json')));
const provenance = parseCsv(await read(path.join(sources, 'previous_ranked_provenance.csv')));
const oldDoc = await read(path.join(sources, '实验管理_snapshot.md'));
const oldSection = oldDoc.split('### 完整 45-Metric 跨空间总表')[1].split('原始完整矩阵')[0];
const oldRows = [...oldSection.matchAll(/^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\|/gm)]
  .map(m => ({ name: m[2].replace(/\s*\(Ours\)$/, ''), attribution: m[3].trim() }));
const oldMap = new Map(oldRows.map(r => [r.name, r.attribution]));
const provMap = new Map(provenance.map(r => [r.metric_name, r]));
const strictNames = new Set(strict.map(r => r.metric_name));
const excludedNames = new Set(excluded.map(r => r.metric_name));
const allNames = new Set([...strictNames, ...excludedNames]);
assert(strict.length === 83 && strictNames.size === 83, 'Strict ranked count/uniqueness failed');
assert(excluded.length === 2 && excludedNames.size === 2, 'Excluded count/uniqueness failed');
assert(allNames.size === 85, 'Ranked/excluded overlap');
assert(oldRows.length === 45 && oldMap.size === 45, 'Historical 45 parse failed');
assert(oldRows.every(r => allNames.has(r.name)), 'Historical metric missing from new table');
assert(provenance.length === 80 && provMap.size === 80, 'Provenance table mismatch');
assert(raw.length === 47600, 'Raw row count mismatch');
assert(verification.status === 'PASS' && verification.ranked_metrics === 83 && verification.unique_raw_keys === 47600, 'Release verification failed');
assert(repair.length === 3 && repairManifest.replaced_count === 3, 'Repair count failed');
for (const fixed of repair) {
  const matches = raw.filter(r => r.metric_name === fixed.metric_name && r.setting === fixed.setting && r.model_or_arch_id === fixed.model_or_arch_id);
  assert(matches.length === 1 && matches[0].status === 'failed' && fixed.status === 'ok', `Unsafe repair: ${fixed.metric_name}`);
  for (const field of ['benchmark','setting','model_or_arch_id','checkpoint_epoch','metric_name','direction','target_value','seed','sample_offset','num_samples','epsilon','checkpoint_path']) {
    assert(matches[0][field] === fixed[field], `Frozen field mismatch ${fixed.metric_name}/${field}`);
  }
  Object.assign(matches[0], fixed);
}

const evidence = new Map();
for (const r of raw) {
  assert(allNames.has(r.metric_name), `Unknown raw metric ${r.metric_name}`);
  const e = evidence.get(r.metric_name) ?? { rows: 0, okFinite: 0, unique: new Set(), settingCounts: new Map() };
  e.rows++;
  const key = `${r.setting}|${r.model_or_arch_id}`;
  assert(!e.unique.has(key), `Duplicate metric/setting/model: ${r.metric_name}, ${key}`);
  e.unique.add(key);
  e.settingCounts.set(r.setting, (e.settingCounts.get(r.setting) ?? 0) + 1);
  if (r.status === 'ok' && r.metric_value !== '' && Number.isFinite(Number(r.metric_value))) e.okFinite++;
  evidence.set(r.metric_name, e);
}
assert(evidence.size === 85, 'Not all candidates have raw evidence');
for (const r of strict) {
  const e = evidence.get(r.metric_name);
  assert(e.rows === 560 && e.okFinite === 560 && e.unique.size === 560,
    `Incomplete strict metric ${r.metric_name}: ${e.rows}/${e.okFinite}`);
  assert(e.settingCounts.get('rb40_mature') === 40, `RB40 coverage failed: ${r.metric_name}`);
  assert(provMap.has(r.metric_name) || r.metric_name.startsWith('trnas_'), `Missing provenance: ${r.metric_name}`);
}

const scoreKeys = ['rb40_mature','nb201_e0','nb201_e5','nb201_e10','nb201_e25','nb201_e50','darts_e0','darts_e100'];
const headers = [
  'strict_rank','metric_name','strict_status','historical_45','source_type','specific_origin','literature_experiment_basis',
  'raw_rows','ok_finite_rows','required_rows','exclusion_category','exclusion_reason',
  ...scoreKeys,'nb201_mean','Zero-cost mean','all_setting_mean','release_id'
];
const releaseId = 'strict_trnas_direct_ode_release_20261024T000000Z';
const round3 = value => Number(value.toFixed(3));
const rows = [];
for (const r of strict) {
  const e = evidence.get(r.metric_name), p = provMap.get(r.metric_name);
  const scores = scoreKeys.map(k => Number(r[k]));
  assert(scores.every(Number.isFinite), `Non-finite strict score: ${r.metric_name}`);
  const nbMean = scoreKeys.slice(1,6).reduce((sum,k) => sum + Number(r[k]),0)/5;
  const zeroMean = (Number(r.nb201_e0)+Number(r.darts_e0))/2;
  rows.push([Number(r.rank),r.metric_name,'ranked',oldMap.has(r.metric_name) ? 'yes':'no',
    p?.['来源类型'] ?? '论文/已有方法',p?.['具体出处'] ?? oldMap.get(r.metric_name),p?.['文献_实验依据'] ?? 'TRNAS [11,12]; Kang direct-autograd numerical repair',e.rows,e.okFinite,560,'','',
    ...scores.map(round3),round3(nbMean),round3(zeroMean),round3(Number(r.all_setting_mean)),releaseId]);
}
for (const r of excluded) {
  const e = evidence.get(r.metric_name);
  const isTrnas = r.metric_name.startsWith('trnas_');
  const attribution = oldMap.get(r.metric_name) ?? '';
  rows.push([null,r.metric_name,'excluded',oldMap.has(r.metric_name) ? 'yes':'no',
    isTrnas ? '论文/已有方法' : '论文/已有方法', attribution || 'RACL [5] / E37',
    isTrnas ? 'TRNAS [11,12]; E21; E38 数值修复未通过' : 'RACL [5]; 严格发布排除清单',
    e.rows,e.okFinite,560,r.eligibility,r.reason,
    ...Array(scoreKeys.length).fill(null),null,null,null,releaseId]);
}
assert(rows.length === 85 && rows.filter(r => r[2]==='ranked').length === 83, 'Output count mismatch');
assert(rows.filter(r => r[3]==='yes').length === 45, 'Historical coverage mismatch');

const csv = [headers,...rows.map(row => row.map((value, index) =>
  index >= 12 && index <= 22 && value != null ? value.toFixed(3) : value))]
  .map(row => row.map(csvCell).join(',')).join('\r\n')+'\r\n';
const csvPath = path.join(here, 'RobustMetric_严格全部指标总表_TRNAS补全_85项_三位小数.csv');
await fs.writeFile(csvPath, '\ufeff'+csv, 'utf8');

const wb = Workbook.create();
const sheet = wb.worksheets.add('严格全部指标');
sheet.showGridLines = false;
sheet.getRangeByIndexes(0,0,rows.length+1,headers.length).values = [headers,...rows];
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
const endCol = 'X';
sheet.getRange(`A1:${endCol}86`).format.font = { name:'Arial', size:10, color:'#202830' };
sheet.getRange(`A1:${endCol}1`).format = { fill:'#17365D', font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'}, rowHeight:34, wrapText:true };
sheet.getRange(`A1:${endCol}1`).format.verticalAlignment = 'center';
sheet.getRange('A:A').format.columnWidth = 12;
sheet.getRange('B:B').format.columnWidth = 38;
sheet.getRange('C:D').format.columnWidth = 16;
sheet.getRange('E:E').format.columnWidth = 18;
sheet.getRange('F:G').format.columnWidth = 38;
sheet.getRange('H:J').format.columnWidth = 15;
sheet.getRange('K:K').format.columnWidth = 22;
sheet.getRange('L:L').format.columnWidth = 80;
sheet.getRange('M:W').format.columnWidth = 16;
sheet.getRange('X:X').format.columnWidth = 42;
sheet.getRange(`M2:W86`).format.numberFormat = '0.000';
sheet.getRange(`A85:${endCol}86`).format.fill = '#FFF1EA';
sheet.getRange('A85:C86').format.font = { name:'Arial',size:10,bold:true,color:'#963B25' };
sheet.getRange(`A2:${endCol}86`).format.rowHeight = 21;
wb.recalculate();
const check = await wb.inspect({kind:'table',range:'严格全部指标!A1:P5',include:'values,formulas',tableMaxRows:5,tableMaxCols:16,maxChars:3500});
console.log(check.ndjson);
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'final formula error scan'});
console.log(errors.ndjson);
const preview = await wb.render({ sheetName:'严格全部指标', range:'A1:P8', scale:1.3, format:'png' });
await fs.writeFile(path.join(here,'preview_三位小数.png'), new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
const xlsxPath = path.join(here,'RobustMetric_严格全部指标总表_TRNAS补全_85项_三位小数.xlsx');
await xlsx.save(xlsxPath);
console.log(JSON.stringify({csvPath,xlsxPath,ranked:strict.length,excluded:excluded.length,historical:oldRows.length,rawRows:raw.length,excludedAudit:excluded.map(r=>({metric:r.metric_name,...Object.fromEntries(['rows','okFinite'].map(k=>[k,evidence.get(r.metric_name)[k]]))}))},null,2));
