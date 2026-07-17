"""多角色 SFT 数据采集器

从任务 state 中提取 instruction-output 对，分别服务 D/B/E/C 四个智能体。
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class SFTSample:
    """SFT 样本"""
    role: str
    instruction: str
    input_text: str
    output: str
    context: Dict[str, Any]
    quality_score: float = 0.0
    source_task: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
            "context": self.context,
            "quality_score": self.quality_score,
            "source_task": self.source_task,
        }


class BaseCollector(ABC):
    """采集器基类"""

    ROLE: str = ""

    @abstractmethod
    def can_collect(self, state: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        pass

    def _quality_score(self, state: Dict[str, Any]) -> float:
        evaluation = state.get("evaluation_result", {}) or {}
        return float(evaluation.get("overall_score", 0))


class DCollector(BaseCollector):
    """D 智能体采集器：需求 → 项目架构"""

    ROLE = "D"

    def can_collect(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("project")) and bool(state.get("requirements"))

    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        requirements = state.get("requirements", "")
        project = state.get("project", {})
        instruction = (
            "请根据以下需求，分析并给出最合适的项目架构（输出严格 JSON 格式）：\n\n"
            f"{requirements[:1500]}"
        )
        output = json.dumps(project, ensure_ascii=False, indent=2)
        return [SFTSample(
            role=self.ROLE,
            instruction=instruction,
            input_text="",
            output=output,
            context={"project_name": project.get("name", "")},
            quality_score=self._quality_score(state),
            source_task=state.get("task_id", ""),
        )]


class BCollector(BaseCollector):
    """B 智能体采集器：需求 + 架构 → 代码"""

    ROLE = "B"

    def can_collect(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("code_artifacts")) and bool(state.get("requirements"))

    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        requirements = state.get("requirements", "")
        project = state.get("project", {})
        artifacts = state.get("code_artifacts", [])

        instruction = (
            "请根据以下需求和项目架构，生成完整的项目代码。\n\n"
            f"需求：\n{requirements[:1500]}\n\n"
            f"项目架构：\n{json.dumps(project, ensure_ascii=False, indent=2)[:1000]}\n\n"
            "请使用标准格式输出代码（=== FILE: 路径 === ... === END ===）。"
        )

        parts = []
        for artifact in artifacts:
            path = artifact.get("file_path", "")
            content = artifact.get("content", "")
            parts.append(f"=== FILE: {path} ===\n{content}\n=== END ===")

        output = "\n\n".join(parts)

        return [SFTSample(
            role=self.ROLE,
            instruction=instruction,
            input_text="",
            output=output,
            context={"file_count": len(artifacts)},
            quality_score=self._quality_score(state),
            source_task=state.get("task_id", ""),
        )]


class ECollector(BaseCollector):
    """E 智能体采集器：代码 + 测试结果 → 评估报告"""

    ROLE = "E"

    def can_collect(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("evaluation_result")) and bool(state.get("static_metrics"))

    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        artifacts = state.get("code_artifacts", [])
        test_results = state.get("test_results", [])
        static_metrics = state.get("static_metrics", {})
        evaluation = state.get("evaluation_result", {})

        code_context = ""
        for artifact in artifacts:
            content = artifact.get("content", "")[:300]
            code_context += f"\n=== {artifact.get('file_path', '')} ===\n{content}\n"

        test_context = ""
        for tr in test_results:
            test_context += f"\n文件: {tr.get('file', '')} | 通过: {tr.get('passed', False)}"
            if not tr.get("passed", False):
                test_context += f"\n  错误: {tr.get('error', '')[:200]}"

        instruction = (
            "请根据以下代码、测试结果和静态分析摘要，给出代码质量评估报告（输出 JSON）。\n\n"
            f"代码摘要：\n{code_context}\n\n"
            f"测试结果：\n{test_context}\n\n"
            f"静态分析摘要：\n{json.dumps(static_metrics, ensure_ascii=False, indent=2)[:1000]}"
        )

        output = json.dumps(evaluation, ensure_ascii=False, indent=2)

        return [SFTSample(
            role=self.ROLE,
            instruction=instruction,
            input_text="",
            output=output,
            context={"overall_score": evaluation.get("overall_score")},
            quality_score=self._quality_score(state),
            source_task=state.get("task_id", ""),
        )]


class CCollector(BaseCollector):
    """C 智能体采集器：代码 → 训练脚本/适配器

    保留并增强原有 C.py 的 DatasetBuilder 逻辑。
    """

    ROLE = "C"

    INSTRUCTION_TEMPLATES = [
        "请编写一个Python函数，{description}",
        "实现以下功能：{description}",
        "编写代码完成：{description}",
        "用Python实现：{description}",
    ]

    def can_collect(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("code_artifacts"))

    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        requirements = state.get("requirements", "")
        artifacts = state.get("code_artifacts", [])
        samples = []

        for artifact in artifacts:
            if artifact.get("language") != "python":
                continue
            content = artifact.get("content", "")
            file_path = artifact.get("file_path", "")
            functions = self._extract_functions(content)

            for func in functions:
                instruction = self._generate_instruction(func, file_path, requirements)
                output = func["code"]
                samples.append(SFTSample(
                    role=self.ROLE,
                    instruction=instruction,
                    input_text="",
                    output=output,
                    context={"source_file": file_path, "function_name": func["name"]},
                    quality_score=self._quality_score(state),
                    source_task=state.get("task_id", ""),
                ))

        return samples

    def _extract_functions(self, code: str) -> List[Dict[str, str]]:
        """基于行解析提取顶层函数定义"""
        functions = []
        lines = code.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(r'^def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:', line)
            if match:
                name = match.group(1)
                func_lines = [line]
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith(' ') and not next_line.startswith('\t'):
                        break
                    func_lines.append(next_line)
                    i += 1
                func_code = '\n'.join(func_lines).rstrip('\n') + '\n'
                functions.append({
                    "name": name,
                    "code": func_code,
                    "description": f"实现 {name} 功能",
                })
                continue
            i += 1

        return functions

    def _generate_instruction(self, func: Dict[str, str], file_path: str, requirements: str) -> str:
        import random
        template = random.choice(self.INSTRUCTION_TEMPLATES)
        description = func.get("description", "")
        if not description and requirements:
            description = f"根据需求：{requirements[:100]}"
        if not description:
            description = f"在文件 {file_path} 中"
        return template.format(description=description)


class SFTCollectorPipeline:
    """SFT 采集流水线"""

    COLLECTORS = [DCollector, BCollector, ECollector, CCollector]

    def __init__(self, roles: List[str] = None):
        self.collectors = [cls() for cls in self.COLLECTORS]
        if roles:
            self.collectors = [c for c in self.collectors if c.ROLE in roles]

    def collect(self, state: Dict[str, Any]) -> List[SFTSample]:
        """从 state 中采集所有启用的角色样本"""
        samples = []
        for collector in self.collectors:
            try:
                if collector.can_collect(state):
                    samples.extend(collector.collect(state))
            except Exception as e:
                from src.core.logger import logger
                logger.warning(f"[SFT] {collector.ROLE} 采集失败: {e}")
        return samples
