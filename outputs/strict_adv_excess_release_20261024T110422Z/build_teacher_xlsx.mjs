import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const folder = path.dirname(new URL(import.meta.url).pathname).replace(/^\/(?=[A-Za-z]:)/, '');
const csvPath = path.join(folder, 'RobustMetric_严格全部指标总表_86项_三位小数.csv');
const xlsxPath = path.join(folder, 'RobustMetric_严格全部指标总表_86项_三位小数.xlsx');
const csv = await fs.readFile(csvPath, 'utf8');
const workbook = await Workbook.fromCSV(csv, { sheetName: '严格总榜' });
const sheet = workbook.worksheets.getItem('严格总榜');
const headers = sheet.getRange('A1:X1').values[0].map(value => String(value).replace(/^\uFEFF/, ''));
sheet.getRange('A1').values = [['strict_rank']];
const scoreNames = [
  'rb40_mature', 'nb201_e0', 'nb201_e5', 'nb201_e10', 'nb201_e25',
  'nb201_e50', 'darts_e0', 'darts_e100', 'nb201_mean',
  'Zero-cost mean', 'all_setting_mean',
];
for (const name of [...scoreNames, 'strict_rank', 'raw_rows', 'ok_finite_rows', 'required_rows']) {
  const column = headers.indexOf(name);
  if (column < 0) throw new Error(`Missing column: ${name}`);
  const range = sheet.getRangeByIndexes(1, column, 86, 1);
  const values = range.values.map(([value]) => [value === '' || value == null ? null : Number(value)]);
  if (values.some(([value]) => value !== null && !Number.isFinite(value))) throw new Error(`Nonfinite ${name}`);
  range.values = values;
  range.setNumberFormat(scoreNames.includes(name) ? '0.000' : '0');
}
if (typeof sheet.getRange('A2').values[0][0] !== 'number' ||
    typeof sheet.getRange('M2').values[0][0] !== 'number') {
  throw new Error('Rank and score cells must be numeric');
}
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getRange('A1:X87').format.font = { name: 'Aptos', size: 10 };
sheet.getRange('A1:X1').format.fill = '#163A63';
sheet.getRange('A1:X1').format.font = { name: 'Aptos', size: 10, bold: true, color: '#FFFFFF' };
sheet.getRange('A1:X1').format.rowHeight = 28;
sheet.getRange('A1:X87').format.verticalAlignment = 'center';
sheet.getRange('A1:X87').format.columnWidth = 13;
sheet.getRange('B1:B87').format.columnWidth = 38;
sheet.getRange('E1:E87').format.columnWidth = 19;
sheet.getRange('F1:F87').format.columnWidth = 31;
sheet.getRange('G1:G87').format.columnWidth = 48;
sheet.getRange('L1:L87').format.columnWidth = 42;
sheet.getRange('X1:X87').format.columnWidth = 39;
workbook.recalculate();
const check = await workbook.inspect({ kind: 'region', sheetId: '严格总榜', range: 'A1:D5', maxChars: 1500 });
console.log(check.ndjson);
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 30 }, maxChars: 1000 });
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: '严格总榜', range: 'A1:G8', scale: 1.5, format: 'png' });
await fs.writeFile(path.join(folder, 'teacher_table_preview.png'), new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
console.log(JSON.stringify({ xlsxPath, rows: 86, columns: headers.length }));
