#!/usr/bin/env python3
"""生成 LangGraph 多智能体任务调优过程技术报告（Word）。"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn

OUT_PATH = "/Users/mingrun/resume-agent-system/docs/LangGraph多智能体任务调优报告.docx"


def set_cell_border(cell, **kwargs):
    """设置表格单元格边框。"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f"w:{edge}"
            element = tcPr.find(qn(tag))
            if element is None:
                element = docx.oxml.OxmlElement(tag)
                tcPr.append(element)
            for key in ["sz", "val", "color", "space"]:
                if key in edge_data:
                    element.set(qn(f"w:{key}"), str(edge_data[key]))


def add_heading(doc, text, level=1):
    """添加标题并设置中文字体。"""
    heading = doc.add_heading(level=level)
    run = heading.add_run(text)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if level == 1:
        run.font.size = Pt(18)
        run.font.color.rgb = RGBColor(0, 0, 128)
    elif level == 2:
        run.font.size = Pt(14)
    else:
        run.font.size = Pt(12)
    return heading


def add_paragraph(doc, text, bold=False, italic=False, size=10.5):
    """添加正文段落。"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(6)
    return p


def add_code_block(doc, code):
    """添加等宽代码块。"""
    p = doc.add_paragraph()
    run = p.add_run(code)
    run.font.name = "Consolas"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    run.font.size = Pt(9)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(6)
    return p


def add_table(doc, headers, rows):
    """添加表格。"""
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.name = "Microsoft YaHei"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    for row in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
            for paragraph in row_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Microsoft YaHei"
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    return table


def main():
    doc = Document()

    # 封面
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("LangGraph 多智能体文章写作任务\n调优过程技术报告")
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 0, 128)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("模型：Kimi k2p5-coding | 任务：D-B-E-C 四阶段智能体平台\n报告日期：2026-07-13")
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(12)

    doc.add_page_break()

    # 1. 背景与目标
    add_heading(doc, "1. 背景与目标", level=1)
    add_paragraph(doc,
        "本报告记录针对 LangGraph 多智能体文章写作系统生成任务的持续调优过程。"
        "该任务要求实现 researcher、writer、reviewer 三个角色，支持最多 3 次修改，"
        "reviewer 从 accuracy、completeness、readability、structure 四个维度评分。"
        "基线版本（v10）耗时 5.3 小时、得分 16.5、多次 600s 超时，几乎不可用。"
    )
    add_paragraph(doc,
        "调优目标围绕三个核心指标：总耗时压缩到 1 小时以内、消除 600s 超时、"
        "评估得分提升到 50+。报告将逐一说明每个版本的问题、策略、效果与失效原因，"
        "为后续更换模型提供明确的 prompt、架构与超参起点。"
    )

    # 2. 架构基线
    add_heading(doc, "2. 架构基线", level=1)
    add_paragraph(doc,
        "平台采用 D-B-E-C 四阶段工作流："
    )
    add_paragraph(doc, "D（Designer）：根据需求输出 file_plan、tech_stack、required_elements；")
    add_paragraph(doc, "B（Developer）：按 file_plan 生成代码，包含骨架、填充、import 干跑、Docker 测试；")
    add_paragraph(doc, "E（Evaluator）：对代码进行多维度评分；")
    add_paragraph(doc, "C（Critic/Rewriter）：根据评估反馈改写需求或 plan，进入下一轮。")
    add_paragraph(doc,
        "关键配置：macOS 使用 Celery solo 模式避免 fork+MPS 崩溃；Redis 做队列；"
        "Qdrant 本地文件模式做向量存储；Docker sandbox 用于测试。"
    )

    # 3. 版本演进与指标
    add_heading(doc, "3. 版本演进与关键指标", level=1)
    add_paragraph(doc, "下表汇总主要版本的耗时、得分、超时次数与核心问题：")
    add_table(doc,
        ["版本", "耗时", "得分", "超时", "核心问题/进展"],
        [
            ["v10", "5.3h", "16.5", "多次", "单响应输出 6000+ token，频繁 600s 超时；代码截断严重"],
            ["v11", "14.3min", "23.0", "0", "文件数降到 8 个，超时消失；chunk 拆分生效，但代码仍是占位符"],
            ["v12", "11.7min", "18.0", "0", "AST 校验通过，但生成内容语义偏离需求"],
            ["v13", "15.8min", "18.0", "0", "增加字段名/类名硬约束，效果不明显"],
            ["v14", "146s", "0", "0", "__init__.py 导入不存在的 create_article，流程快速失败"],
            ["v15", "~30min", "~15", "0", "agents.py 单文件 1699 行，graph.py 缺少 ArticleGraph"],
            ["v16", "~12min", "~15", "0", "拆分为 researcher.py/writer.py/reviewer.py，pytest 可运行，1/4 通过"],
        ]
    )

    # 4. 有效的策略
    add_heading(doc, "4. 被验证有效的策略", level=1)

    add_heading(doc, "4.1 文件级拆分：从单响应到每 group 1 个文件", level=2)
    add_paragraph(doc,
        "v10 的致命问题是把多个文件塞进一次 LLM 调用，输出 token 超过 6000，"
        "触发 600s 超时。将生成粒度改为"每 group 一个文件"后，单次输出控制在 "
        "2000-4000 token，耗时从 5.3 小时降到 15 分钟以内。"
    )

    add_heading(doc, "4.2 限制 file_plan 规模与禁止文件清单", level=2)
    add_paragraph(doc,
        "在 D 阶段 prompt 中加入上限：文件总数 ≤ 10，禁止 pyproject.toml、.env.example、"
        "conftest.py、src/__init__.py、src/config.py。减少无关文件后，上下文长度和 "
        "LLM 注意力分散问题明显改善。"
    )

    add_heading(doc, "4.3 import 干跑作为零成本拦截器", level=2)
    add_paragraph(doc,
        "在进 Docker 之前做 import 干跑，能在不花钱运行测试的情况下发现项目内部 "
        "缺失模块/符号。v11 之后几乎所有跨文件引用错误都在这一阶段被捕获。"
    )

    add_heading(doc, "4.4 __init__.py 规则化生成", level=2)
    add_paragraph(doc,
        "v14 证明让 LLM 写 __init__.py 会导入不存在的符号。改为用 AST 扫描 src/*.py "
        "中实际定义的顶层 Class/Function，再自动生成 import 语句后，import 阶段 "
        "错误基本消失。"
    )

    add_heading(doc, "4.5 单文件行数/类方法数上限", level=2)
    add_paragraph(doc,
        "v15 的 agents.py 1699 行仍被截断。v16 在 D prompt 中强制'每个文件不超过 "
        "200 行、每个类不超过 3 个方法'，并把三个 Agent 拆到 researcher.py / writer.py / "
        "reviewer.py，代码首次变得可导入、可运行。"
    )

    # 5. 无效或边际效应递减的策略
    add_heading(doc, "5. 无效或边际效应递减的策略", level=1)

    add_heading(doc, "5.1 单纯降低 max_tokens 无法解决截断", level=2)
    add_paragraph(doc,
        "把 complex 文件 max_tokens 从 16384 降到 8192，虽然减少了超时，"
        "但没有解决 LLM 在语义上'写不完'的问题。"
    )

    add_heading(doc, "5.2 chunk 拆分 + 续写合并并未显著提升功能正确性", level=2)
    add_paragraph(doc,
        "理论上把大文件拆成多个 chunk 可以降低单次输出长度，但实践中："
        "(1) LLM 容易在 chunk 中生成空方法体；"
        "(2) 拼接时容易出现缩进/try-except 块断裂；"
        "(3) 重试和回退逻辑增加了调用次数，但得分没有提升。"
        "更有效的做法是把文件本身拆小，而不是在一个大文件里做 chunk 拼接。"
    )

    add_heading(doc, "5.3 硬编码字段名/类名约束效果有限", level=2)
    add_paragraph(doc,
        "在 chunk prompt 中强制"评分字段必须是 accuracy/completeness/readability/structure"、"
        ""异常类名必须是 RevisionLimitExceeded"，对字段名有一定约束，但无法保证 "
        "LLM 正确实现状态流转、循环次数限制、异常抛出条件等功能逻辑。"
    )

    add_heading(doc, "5.4 空方法体检测导致过多重试", level=2)
    add_paragraph(doc,
        "最初用"方法体少于 5 行有效代码"作为 stub 检测标准，导致大量合法短方法被 "
        "误判，引发重试和回退。后续简化为只检测 pass / 空体 / return None 后，"
        "重试次数下降，但功能正确性问题仍未解决。"
    )

    # 6. 模型能力边界
    add_heading(doc, "6. 当前模型（Kimi k2p5-coding）的能力边界", level=1)
    add_paragraph(doc,
        "经过多轮调优，可以得出以下关于该模型在复杂代码生成任务上的能力边界："
    )
    add_paragraph(doc,
        "1. 单次输出上限：稳定生成约 3000-4000 token 的完整代码文件可行，"
        "超过 5000 token 后截断/语义不完整的概率显著上升。"
    )
    add_paragraph(doc,
        "2. 跨文件一致性：无法 reliably 维护多文件之间的接口约定，"
        "需要规则引擎（如 __init__.py 自动生成、依赖校验）兜底。"
    )
    add_paragraph(doc,
        "3. 状态机/循环逻辑：对"最多修改 3 次"、"满足阈值退出"、"
        ""抛出 RevisionLimitExceeded" 这类需要精确控制的状态流转，"
        "LLM 容易写成看似合理但实际条件错误的代码。"
    )
    add_paragraph(doc,
        "4. 测试驱动：无法根据隐藏测试意图生成完全正确的实现，"
        "只能生成"能通过自身测试"的代码，与评估器隐藏测试之间存在 gap。"
    )
    add_paragraph(doc,
        "5. 长上下文利用：当 prompt 中塞入多个文件摘要后，LLM 会丢失细节，"
        "表现为字段名错误、方法体变短、docstring 截断。"
    )

    # 7. 为下次更换模型提供的调优起点
    add_heading(doc, "7. 为下次更换模型提供的调优起点", level=1)
    add_paragraph(doc,
        "基于本次调优结论，建议新模型沿用以下已被验证的架构与 prompt，"
        "并重点解决当前模型无法克服的功能正确性问题。"
    )

    add_heading(doc, "7.1 必须保留的架构设定", level=2)
    add_paragraph(doc, "- 每 group 1 个文件，单次输出控制在 ≤4000 token")
    add_paragraph(doc, "- file_plan 文件总数 ≤ 10，禁止 pyproject.toml/.env.example/conftest.py/config.py/__init__.py")
    add_paragraph(doc, "- 每个文件 ≤ 200 行，每个类 ≤ 3 个方法")
    add_paragraph(doc, "- import 干跑在 Docker 测试前执行")
    add_paragraph(doc, "- __init__.py 由 AST 扫描规则生成")

    add_heading(doc, "7.2 建议保留的 prompt 约束", level=2)
    add_paragraph(doc, "- 明确列出每个元素的 name、type、methods、logic_requirements")
    add_paragraph(doc, "- 硬编码关键字段名和类名（accuracy/completeness/readability/structure、RevisionLimitExceeded）")
    add_paragraph(doc, "- 禁止 pass / return None / # TODO 占位")

    add_heading(doc, "7.3 需要重点攻坚的方向", level=2)
    add_paragraph(doc,
        "a) 状态机精确定义：把"revision_count 从 0 开始，每次 reviewer 评分后 +1，"
        "超过 MAX_REVISIONS 时抛出 RevisionLimitExceeded"写成伪代码塞进 prompt。"
    )
    add_paragraph(doc,
        "b) 显式测试用例作为生成目标：如果评估器隐藏测试可部分推断，"
        "把关键断言直接作为 required_elements 的逻辑要求。"
    )
    add_paragraph(doc,
        "c) RAG 引入 LangGraph 示例：从 keon/algorithms、psf/requests 等已索引仓库中 "
        "检索多智能体 / LangGraph 示例，作为 in-context 参考。"
    )
    add_paragraph(doc,
        "d) 方法级生成：如果新模型仍无法一次性写对复杂类，"
        "进一步把生成粒度从"文件"降到"方法"，每个方法单独生成并 AST 校验。"
    )
    add_paragraph(doc,
        "e) 评估反馈闭环：让 C 阶段拿到 evaluator 的具体失败项，"
        "针对性地改写 file_plan 或给 B 阶段发"修复任务"，而不是泛泛地"提高质量"。"
    )

    add_heading(doc, "7.4 建议的超参起点", level=2)
    add_code_block(doc, """generation:
  split_threshold_tokens: 2000      # 超过此阈值才拆分 chunk
  chunk_max_tokens: 3072            # 每个 chunk 输出上限
  max_completion_tokens_simple: 4096
  max_completion_tokens_normal: 4096
  max_completion_tokens_complex: 8192

orchestrator:
  max_iterations: 5
""")

    add_heading(doc, "7.5 建议的 D 阶段 prompt 核心约束", level=2)
    add_code_block(doc, """- file_plan 文件总数不得超过 10 个
- 禁止 pyproject.toml、.env.example、conftest.py、src/__init__.py、src/config.py
- 每个文件不得超过 200 行；每个类不得超过 3 个方法
- LangGraph 工作流类必须放在 src/graph.py
- 入口函数必须放在 src/main.py 或 src/graph.py
- required_elements 必须包含 name/type/methods/logic_requirements
""")

    # 8. 结论
    add_heading(doc, "8. 结论", level=1)
    add_paragraph(doc,
        "本次调优成功解决了 D-B-E-C 平台在复杂任务上的架构级问题："
        "5.3 小时 → 12 分钟、600s 超时归零、代码从无法导入变为可运行。"
        "但当前模型在功能正确性上遇到明显天花板，得分停留在 15-23 区间，"
        "pytest 仅 1/4 通过，核心状态机和 Agent 逻辑仍未实现正确。"
    )
    add_paragraph(doc,
        "下一步若要突破 50 分，需要更强的基础模型，或引入"测试用例驱动的 "
        "方法级生成 + RAG 示例 + 评估反馈闭环"。本报告所列的起点、约束与失败教训，"
        "可作为更换模型后快速复现当前最优状态并继续上攻的基准。"
    )

    doc.save(OUT_PATH)
    print(f"报告已保存到: {OUT_PATH}")


if __name__ == "__main__":
    main()
