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
const raw = parseCsv(await read(path.join(sources, 'metric_values_merged.csv')));
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
assert(strict.length === 80 && strictNames.size === 80, 'Strict ranked count/uniqueness failed');
assert(excluded.length === 5 && excludedNames.size === 5, 'Excluded count/uniqueness failed');
assert(allNames.size === 85, 'Ranked/excluded overlap');
assert(oldRows.length === 45 && oldMap.size === 45, 'Historical 45 parse failed');
assert(oldRows.every(r => allNames.has(r.name)), 'Historical metric missing from new table');
assert(provenance.length === 80 && provMap.size === 80, 'Provenance table mismatch');
assert(raw.length === 47600, 'Raw row count mismatch');

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
  assert(provMap.has(r.metric_name), `Missing provenance: ${r.metric_name}`);
}

const scoreKeys = ['rb40_mature','nb201_e0','nb201_e5','nb201_e10','nb201_e25','nb201_e50','darts_e0','darts_e100'];
const headers = [
  'strict_rank','metric_name','strict_status','historical_45','source_type','specific_origin','literature_experiment_basis',
  'raw_rows','ok_finite_rows','required_rows','exclusion_category','exclusion_reason',
  ...scoreKeys,'nb201_mean','Zero-cost mean','all_setting_mean','release_id'
];
const releaseId = 'strict_final_release_20261008T000000Z';
const rows = [];
for (const r of strict) {
  const e = evidence.get(r.metric_name), p = provMap.get(r.metric_name);
  const scores = scoreKeys.map(k => Number(r[k]));
  assert(scores.every(Number.isFinite), `Non-finite strict score: ${r.metric_name}`);
  const nbMean = scoreKeys.slice(1,6).reduce((sum,k) => sum + Number(r[k]),0)/5;
  const zeroMean = (Number(r.nb201_e0)+Number(r.darts_e0))/2;
  rows.push([Number(r.rank),r.metric_name,'ranked',oldMap.has(r.metric_name) ? 'yes':'no',
    p['来源类型'],p['具体出处'],p['文献_实验依据'],e.rows,e.okFinite,560,'','',
    ...scores,nbMean,zeroMean,Number(r.all_setting_mean),releaseId]);
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
assert(rows.length === 85 && rows.filter(r => r[2]==='ranked').length === 80, 'Output count mismatch');
assert(rows.filter(r => r[3]==='yes').length === 45, 'Historical coverage mismatch');

const csv = [headers,...rows].map(r => r.map(csvCell).join(',')).join('\r\n')+'\r\n';
const csvPath = path.join(here, 'RobustMetric_严格全部指标总表_85项.csv');
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
sheet.getRange(`M2:W86`).format.numberFormat = '0.00';
sheet.getRange(`A82:${endCol}86`).format.fill = '#FFF1EA';
sheet.getRange('A82:C86').format.font = { name:'Arial',size:10,bold:true,color:'#963B25' };
sheet.getRange(`A2:${endCol}86`).format.rowHeight = 21;
wb.recalculate();
const preview = await wb.render({ sheetName:'严格全部指标', range:'A1:P8', scale:1.3, format:'png' });
await fs.writeFile(path.join(here,'preview.png'), new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(wb);
const xlsxPath = path.join(here,'RobustMetric_严格全部指标总表_85项.xlsx');
await xlsx.save(xlsxPath);
console.log(JSON.stringify({csvPath,xlsxPath,ranked:strict.length,excluded:excluded.length,historical:oldRows.length,rawRows:raw.length,excludedAudit:excluded.map(r=>({metric:r.metric_name,...Object.fromEntries(['rows','okFinite'].map(k=>[k,evidence.get(r.metric_name)[k]]))}))},null,2));
