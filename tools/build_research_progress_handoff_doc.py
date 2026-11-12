from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(r"D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric")
OUTPUT = ROOT / "docs" / "experiments" / "RobustMetric_研究进展跟进说明_20261016.docx"


def set_run_font(run, east_asia="宋体", latin="Arial", size=10.5, bold=None, color="000000"):
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=110, bottom=90, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def remove_paragraph_border(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is not None:
        p_pr.remove(p_bdr)


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def style_paragraph(paragraph, size=10.5, bold=False, color="000000", align=None):
    if align is not None:
        paragraph.alignment = align
    for run in paragraph.runs:
        set_run_font(run, size=size, bold=bold, color=color)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.15


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.2
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_run_font(lead, size=10.5, bold=True)
        rest = p.add_run(text[len(bold_lead):])
        set_run_font(rest, size=10.5)
    else:
        run = p.add_run(text)
        set_run_font(run, size=10.5)
    return p


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    run = p.add_run(text)
    set_run_font(run, size=10.2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    return p


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    p.paragraph_format.space_after = Pt(4)
    for run in p.runs:
        set_run_font(run, east_asia="微软雅黑", latin="Arial", size=14 if level == 1 else 11.5, bold=True)
    return p


def add_table(doc, headers, rows, widths_cm=None, font_size=9.0, center_cols=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for idx, label in enumerate(headers):
        cell = header.cells[idx]
        cell.text = str(label)
        set_cell_shading(cell, "1F4E78")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            set_run_font(run, east_asia="微软雅黑", size=font_size, bold=True, color="FFFFFF")
        if widths_cm:
            cell.width = Cm(widths_cm[idx])
    for ridx, data in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        for idx, value in enumerate(data):
            cell = row.cells[idx]
            cell.text = str(value)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ridx % 2 == 1:
                set_cell_shading(cell, "F3F6FA")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center_cols and idx in center_cols else WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                set_run_font(run, size=font_size)
            if widths_cm:
                cell.width = Cm(widths_cm[idx])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


doc = Document()
section = doc.sections[0]
section.page_width = Cm(21.0)
section.page_height = Cm(29.7)
section.top_margin = Cm(1.8)
section.bottom_margin = Cm(1.8)
section.left_margin = Cm(1.8)
section.right_margin = Cm(1.8)

styles = doc.styles
styles["Normal"].font.name = "Arial"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
styles["Normal"].font.size = Pt(10.5)
for style_name in ("Title", "Heading 1", "Heading 2"):
    style = styles[style_name]
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.name = "Arial"
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
title_style_ppr = styles["Title"]._element.get_or_add_pPr()
title_style_border = title_style_ppr.find(qn("w:pBdr"))
if title_style_border is not None:
    title_style_ppr.remove(title_style_border)

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(8)
remove_paragraph_border(title)
run = title.add_run("RobustMetric 研究进展跟进说明")
set_run_font(run, east_asia="微软雅黑", size=20, bold=True)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.paragraph_format.space_after = Pt(14)
run = subtitle.add_run("面向工作室内部跟进同学    更新日期 2026年10月16日")
set_run_font(run, east_asia="微软雅黑", size=10, color="555555")

add_body(doc, "这份说明用于帮助新接手的同学快速理解 RobustMetric 项目的研究目标、实验规范、当前可靠结论和后续工作。当前严格发布已经完成并通过独立审计：80 个指标进入跨空间总榜，5 个指标因定义或数值问题被排除；主榜数据可以用于汇报和论文整理，但 E37 新指标的跨模型泛化与机制解释仍需谨慎。")

add_heading(doc, "一 当前状态", 1)
add_table(
    doc,
    ["项目项", "当前状态", "说明"],
    [
        ["严格发布", "完成", "服务器状态为 COMPLETE，机器校验为 PASS"],
        ["原始数据", "47,600 行", "47,600 个唯一协议键，无重复键"],
        ["入榜指标", "80 项", "每项均有 560 条唯一、有限、status=ok 的原始记录"],
        ["排除指标", "5 项", "失败证据保留，不用局部结果替代"],
        ["新增实验", "E36 与 E37 已合并", "E36 为 4,480 行，E37 为 5,040 行"],
        ["RB40 覆盖", "40 个模型完整", "不存在把 29/40 模型当作最终结论的问题"],
        ["当前研究重点", "稳健性解释", "主榜已冻结，重点转向空间权重、留出验证和机制诊断"],
    ],
    widths_cm=[3.2, 3.0, 10.7],
    font_size=9.2,
    center_cols={1},
)

add_heading(doc, "二 研究目标与严格实验规范", 1)
add_body(doc, "项目希望评估无需完整训练或只需少量样本即可计算的鲁棒性代理指标，并检验这些指标能否在不同模型集合和 NAS 搜索空间中稳定反映真实鲁棒性。正式排名使用方向统一后的 Spearman 相关系数，数值越高表示指标排序与真实鲁棒性排序越一致。")
add_table(
    doc,
    ["空间", "设置", "每设置对象数", "贡献行数"],
    [
        ["RobustBench RB40", "mature", "40", "40"],
        ["NAS Bench 201", "e0 e5 e10 e25 e50", "64", "320"],
        ["DARTS", "e0 e100", "100", "200"],
        ["合计", "8 个设置", "", "560"],
    ],
    widths_cm=[4.0, 5.4, 3.4, 3.4],
    font_size=9.2,
    center_cols={1, 2, 3},
)
add_body(doc, "只有同时满足以下条件的指标才能进入 All setting mean：覆盖完整 560 行网格；协议键唯一；指标值有限；全部原始行为 status=ok；指标定义能跨三个空间成立。失败项不能填零、插补或删除后继续排名。八个设置等权，因此 NB201 的五个训练阶段在主榜中占五个设置权重。")

add_heading(doc, "三 研究推进过程", 1)
add_table(
    doc,
    ["阶段", "主要工作", "当前用途"],
    [
        ["E00 至 E25", "早期指标、RobustBench 与跨空间探索，包含老师已有工作及历史基线", "作为历史来源和指标出处，不直接替代严格发布"],
        ["E26 至 E35", "容量家族、去混杂、跨空间通用性及结构指标分析", "解释不同指标在哪些空间有效，以及为什么失效"],
        ["E36", "设计并完整运行 8 个新指标", "已以 4,480 条原始行并入严格发布"],
        ["E37", "预注册并完整运行 9 个多视图与相关性指标", "已以 5,040 条原始行并入严格发布，并完成冻结留出和混杂诊断"],
        ["严格发布", "统一合并既有指标、E36、E37，执行完整性门槛和方向统一排名", "当前唯一正式总榜"],
        ["2026年10月审计", "独立复算主榜、核验 manifest、标签、哈希、留出结果和计算成本", "确认统计发布与原始网格一致，并补充稳健性解释"],
    ],
    widths_cm=[2.8, 8.2, 5.8],
    font_size=8.8,
)

add_heading(doc, "四 严格主榜的主要结果", 1)
add_table(
    doc,
    ["排名", "指标", "来源", "All setting mean"],
    [
        ["1", "adversarial_loss", "已有方法", "0.614290"],
        ["2", "natural_loss", "已有方法", "0.591311"],
        ["3", "feature_measure_multiview_max_delta", "Ours E37", "0.524140"],
        ["4", "feature_measure_multiview_rms_delta", "Ours E37", "0.494958"],
        ["5", "fsp_delta", "已有方法", "0.479411"],
    ],
    widths_cm=[1.8, 8.0, 3.2, 3.8],
    font_size=9.2,
    center_cols={0, 2, 3},
)
add_body(doc, "E37 中表现最好的两个指标是 multiview max 和 multiview rms，它们在冻结的八设置主榜中位列第 3 和第 4。这个结果来自完整 560 行主实验，并非只使用 NB201 或只使用部分 RB40 模型。")

add_heading(doc, "五 E37 结果应如何解释", 1)
add_table(
    doc,
    ["指标", "主榜名次", "NB201 五阶段均值", "DARTS 两阶段均值", "三空间等权名次", "去 NB201 名次"],
    [
        ["multiview max", "3", "0.700925", "0.193497", "3", "19"],
        ["multiview rms", "4", "0.684606", "0.155558", "6", "34"],
    ],
    widths_cm=[3.6, 2.0, 3.4, 3.3, 2.8, 2.6],
    font_size=8.8,
    center_cols={1, 2, 3, 4, 5},
)
add_body(doc, "主榜优势主要来自 NB201。multiview max 在三空间等权下仍排第 3，但去掉 NB201 后降至第 19；multiview rms 在三空间等权下排第 6，去掉 NB201 后降至第 34。DARTS e100 上两项相关性接近零。因此，可以把它们描述为严格主网格中的高排名候选指标，但不能直接描述为已经证实的跨空间通用指标。")

add_heading(doc, "六 冻结留出与结构混杂诊断", 1)
add_body(doc, "E37 额外使用 11 个冻结 RB40 权重进行留出验证。共有 99 条唯一记录，其中 10 个有效模型对应 90 条成功记录；Rebuffi2021Fixing 70 16 cutmix ddpm 权重损坏，相关 9 条记录保持显式失败。没有填零或插补。")
add_table(
    doc,
    ["指标", "留出 rho", "95% bootstrap 区间", "解释"],
    [
        ["multiview max", "0.5394", "-0.0566 至 0.8765", "点估计为正，但区间跨零"],
        ["multiview rms", "-0.3939", "-0.9500 至 0.3585", "点估计为负，区间跨零"],
    ],
    widths_cm=[3.6, 2.6, 4.2, 6.5],
    font_size=9.0,
    center_cols={1, 2},
)
add_body(doc, "留出样本只有 10 个有效模型，区间较宽。结构混杂诊断还显示，控制操作组成后，E37 的部分相关性明显减弱。例如 multiview max 在 NB201 e0 中由 0.6857 降至 0.1282，在 DARTS e0 中由 0.3811 变为 -0.1420。现阶段应把结构组成视为重要混杂因素。")
add_body(doc, "对同一批 100 个 DARTS 架构，e0 与 e100 使用相同的最终鲁棒性标签排序，但两个指标自身的排序相关性分别为 -0.1970 和 -0.2434。该结果说明训练阶段会明显改变指标排序。当前可检验的工作假设是：早期信号较多反映架构组成，训练后表征漂移削弱这种代理关系。这个判断属于后验机制诊断，不改变冻结主榜。")

add_heading(doc, "七 排除指标与历史问题", 1)
add_table(
    doc,
    ["指标或问题", "原因", "处理结果"],
    [
        ["racl_arch_lipschitz_log_upper", "对固定 RB40 网络没有同一数学定义", "按空间特异指标排除"],
        ["racl_fixed_network_log_upper_proxy", "固定网络代理量与架构单元上界不是同一对象", "按空间特异指标排除"],
        ["三个 TRNAS 指标", "冻结定义下步长计算发生未解决的数值下溢", "float32 float64 和 CPU 隔离复现均失败，显式排除"],
        ["RB40 历史 CUDA OOM", "运行资源问题，不是缺少 11 个权重", "隔离续跑完成，最终 RB40 覆盖 40 模型"],
        ["早期候选榜遗漏 E36", "候选表没有合并 E36 原始行", "候选快照废弃，以严格发布为准"],
        ["留出集中一份损坏权重", "权重文件不可加载", "保留 9 条失败，仅用 10 个有效模型分析"],
    ],
    widths_cm=[5.0, 6.2, 5.6],
    font_size=8.7,
)

add_heading(doc, "八 计算成本", 1)
add_body(doc, "E37 的九个指标共享一次完整计算。第一个指标会完成攻击、三次前向和特征记录，其余指标读取缓存，因此 max 和 rms 行中的微秒级 runtime 不能视为独立端到端成本。历史记录显示，完整 E37 家族在单模型、64 样本、batch size 1、CUDA 条件下，每个 setting 的单模型中位耗时约为 3.08 至 5.43 秒。RB40 日志峰值显存约 4694 MiB，NB201 和 DARTS 的对应日志峰值较低。聚合公式本身开销很小，但它依赖的共享特征计算并不免费。")

add_heading(doc, "九 文件位置与使用顺序", 1)
add_table(
    doc,
    ["内容", "位置", "用途"],
    [
        ["本地项目", r"D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric", "代码、文档和本地同步材料"],
        ["服务器项目", r"wangjiahao:/workspace/robust_metric_lab_nas", "正式实验与原始输出"],
        ["严格发布", r"outputs/student_metrics/wangjiahao/strict_final_release_20261008T000000Z", "当前唯一正式总榜与验收文件"],
        ["最终报告", r"docs/experiments/E36_E37_严格跨空间实验最终报告.md", "主榜、协议、排除项和留出结果"],
        ["审计分析", r"docs/experiments/严格发布接手审计与E37稳健性分析_20261016.md", "独立复算、敏感性、成本和后续判断"],
        ["问题记录", r"docs/experiments/E36_E37_严格实验问题与排除说明.md", "历史问题和不应使用的文件"],
        ["老师用总表", r"桌面 RobustMetric项目资料_20261009", "汇报用 CSV 和简化版 Excel"],
    ],
    widths_cm=[3.2, 9.6, 4.0],
    font_size=8.2,
)
add_body(doc, "跟进时应先看 release_verification.json，再看 ranking/strict_ranking.csv 和 ranking/excluded_metrics.csv，最后结合中文报告解释结果。不要使用 strict_final_candidate 或早于 2026年10月8日的全部指标排行作为最终结论。")

add_heading(doc, "十 后续建议", 1)
add_bullet(doc, "论文和汇报继续使用冻结的 80 项严格主榜，不调整指标方向、空间权重或留出模型。")
add_bullet(doc, "将 E37 的结论写成高排名候选及其限制，明确说明优势主要来自 NB201，DARTS 后期和冻结留出证据不足。")
add_bullet(doc, "如果继续做实验，先预注册训练阶段稳定性机制验证，重点区分架构组成信号、表征漂移和真正的鲁棒性机制。")
add_bullet(doc, "只有论文需要分解攻击、特征提取、分量计算和聚合器时间时，才等待空闲 GPU 补做分段成本基准。")
add_bullet(doc, "任何新指标都必须先固定定义、方向、协议、完整 560 行网格和排除规则，再运行主实验；失败必须保留原始证据。")

add_heading(doc, "十一 可直接使用的汇报口径", 1)
add_body(doc, "严格发布已完成 47,600 条唯一原始记录的核验，共 80 个指标满足完整 560 行跨空间网格并进入总榜。E37 的 multiview max 和 rms 在冻结八设置主榜中排名第 3 和第 4，其中 max 在三空间等权下仍为第 3。两项指标的优势主要来自 NB201，在 DARTS 后期和 10 个有效留出模型上的证据较弱；结构控制后相关性也明显下降。因此，当前成果支持将它们作为值得继续验证的新指标，但尚不足以宣称已经获得稳定的跨空间通用机制。")

doc.core_properties.title = "RobustMetric 研究进展跟进说明"
doc.core_properties.subject = "工作室内部项目跟进"
doc.core_properties.author = "RobustMetric 项目组"
doc.core_properties.keywords = "RobustMetric, RobustNAS, 跨空间评估, E36, E37"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUTPUT)
print(OUTPUT)
