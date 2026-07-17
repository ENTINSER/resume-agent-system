"""E - Evaluator（质量评估员）P1优化版

优化点：
- 强制JSON格式输出（更可靠解析）
- 多维度量化指标（代码质量、功能、性能、文档、测试）
- 结构化的问题和建议分类
- 降级评估更智能（基于代码特征）
- 评估历史对比
"""

import re
import json
import ast
from typing import List, Dict, Any, Optional

from src.core.shared_state import SharedState
from src.core.logger import logger
from src.core.llm import get_llm
from src.core.config import settings
from src.core.code_graph import CodeGraphAnalyzer
from src.vector_store.rag_retriever import RAGRetriever


# ===== 优化的系统提示词（强制JSON输出） =====
EVALUATOR_SYSTEM_PROMPT = """你是一位严格的质量评估员。你的任务是对代码进行全面评估。

## 评估维度与权重
1. 代码质量（30%）：PEP8规范、类型提示、文档字符串、注释质量、命名规范
2. 功能完整性（30%）：需求满足度、边界处理、错误处理、异常安全性
3. 性能（20%）：时间复杂度、空间复杂度、资源使用、算法效率
4. 文档质量（20%）：README完整性、API文档、使用示例、代码注释

## 输出格式（必须严格遵守JSON格式）

```json
{
  "overall_score": 85,
  "dimensions": {
    "code_quality": {
      "score": 80,
      "weight": 0.3,
      "details": [
        "遵循PEP8规范",
        "缺少部分类型提示"
      ]
    },
    "functionality": {
      "score": 90,
      "weight": 0.3,
      "details": [
        "满足核心需求",
        "边界处理完善"
      ]
    },
    "performance": {
      "score": 85,
      "weight": 0.2,
      "details": [
        "时间复杂度合理",
        "空间使用可优化"
      ]
    },
    "documentation": {
      "score": 85,
      "weight": 0.2,
      "details": [
        "README完整",
        "API文档需要补充"
      ]
    }
  },
  "passed": true,
  "pass_threshold": 80,
  "issues": [
    {
      "severity": "high|medium|low",
      "category": "syntax|functionality|performance|security|style",
      "file": "main.py",
      "line": 10,
      "description": "具体问题描述"
    }
  ],
  "recommendations": [
    {
      "priority": "high|medium|low",
      "category": "refactor|optimize|test|document",
      "description": "具体建议描述",
      "effort": "small|medium|large"
    }
  ]
}
```

## 评分标准
- 90-100: 优秀，代码质量高，可直接使用
- 80-89: 良好，满足需求，有少量改进空间
- 70-79: 合格，功能完整但质量待提升
- 60-69: 不合格，需要修改
- <60: 严重问题，需要重写

## 针对简单项目的评分调整
如果项目是一个简单的工具（如计算器、TODO列表、简单脚本），评分标准应适当放宽：
- 代码质量可适当降低要求（不需要太复杂的类型提示和文档字符串）
- 功能完整性只要满足基本需求即可
- 性能方面简单项目通常没有问题
- 文档质量只要有基本 README 即可

## 需求对齐硬规则（最高优先级）
- 必须将代码与用户需求进行对比
- 如果代码明显没有实现用户需求（例如：需求是“LangGraph 多智能体协作系统”，但代码是计算器、四则运算等无关内容），则 overall_score 必须为 0，passed 为 false
- 需求不匹配时，功能完整性（functionality）维度必须为 0 分

## 注意事项
- 评分必须客观公正，基于代码实际质量
- 问题必须具体（包含文件、行号、描述）
- 建议必须可操作（明确优先级和工作量）
- 必须通过JSON格式输出，不要包含其他内容
- 如果测试已通过，功能完整性不应低于 70 分
"""


class EvaluationParser:
    """评估结果解析器 - P1核心优化"""
    
    def __init__(self, response: str):
        self.response = response
    
    def parse(self) -> dict:
        """主解析入口"""
        # 策略1: 提取JSON代码块
        result = self._parse_json_block()
        if result:
            return self._validate_and_normalize(result)
        
        # 策略2: 提取整个JSON对象
        result = self._parse_json_raw()
        if result:
            return self._validate_and_normalize(result)
        
        # 策略3: 正则提取（降级）
        result = self._parse_regex_fallback()
        if result:
            return self._validate_and_normalize(result)
        
        # 最终降级
        return self._default_result()
    
    def _parse_json_block(self) -> Optional[dict]:
        """解析JSON代码块"""
        patterns = [
            r'```json\s*(\{[\s\S]*?\})\s*```',
            r'```\s*(\{[\s\S]*?\})\s*```',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None
    
    def _parse_json_raw(self) -> Optional[dict]:
        """解析原始JSON对象"""
        try:
            # 尝试找到JSON对象
            match = re.search(r'\{[\s\S]*"overall_score"[\s\S]*\}', self.response)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return None
    
    def _parse_regex_fallback(self) -> Optional[dict]:
        """正则提取降级方案"""
        result = {
            "overall_score": 0.0,
            "dimensions": {},
            "passed": False,
            "pass_threshold": 80,
            "issues": [],
            "recommendations": []
        }
        
        # 提取总体评分
        overall_match = re.search(r'总体评分[:\s]*(\d+(?:\.\d+)?)', self.response)
        if overall_match:
            result["overall_score"] = float(overall_match.group(1))
        
        # 提取各维度评分
        dim_patterns = {
            "code_quality": [r'代码质量[:\s]*(\d+(?:\.\d+)?)', r'quality[:\s]*(\d+(?:\.\d+)?)'],
            "functionality": [r'功能完整性[:\s]*(\d+(?:\.\d+)?)', r'functionality[:\s]*(\d+(?:\.\d+)?)'],
            "performance": [r'性能[:\s]*(\d+(?:\.\d+)?)', r'performance[:\s]*(\d+(?:\.\d+)?)'],
            "documentation": [r'文档质量[:\s]*(\d+(?:\.\d+)?)', r'documentation[:\s]*(\d+(?:\.\d+)?)']
        }
        
        for dim_name, patterns in dim_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, self.response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    weight = {"code_quality": 0.3, "functionality": 0.3, "performance": 0.2, "documentation": 0.2}.get(dim_name, 0.25)
                    result["dimensions"][dim_name] = {
                        "score": score,
                        "weight": weight,
                        "details": ["正则提取"]
                    }
                    break
        
        # 提取是否通过
        passed_match = re.search(r'(?:是否通过|passed)[:\s]*(是|否|true|false|yes|no)', self.response, re.IGNORECASE)
        if passed_match:
            val = passed_match.group(1).lower()
            result["passed"] = val in ('是', 'true', 'yes')
        else:
            result["passed"] = result["overall_score"] >= 80
        
        # 提取问题列表
        issues = self._extract_items(self.response, ["问题列表", "Issues", "问题"])
        result["issues"] = [{"severity": "medium", "category": "style", "description": issue} for issue in issues]
        
        # 提取建议列表
        recommendations = self._extract_items(self.response, ["改进建议", "Recommendations", "建议"])
        result["recommendations"] = [{"priority": "medium", "category": "refactor", "description": rec, "effort": "medium"} for rec in recommendations]
        
        return result if result["overall_score"] > 0 else None
    
    def _extract_items(self, text: str, headers: List[str]) -> List[str]:
        """提取列表项"""
        items = []
        for header in headers:
            # 找到header位置
            pattern = rf'{header}[：:]\s*\n'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start = match.end()
                # 提取后续列表项
                section = text[start:start+1000]
                for line in section.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') or line.startswith('* '):
                        items.append(line[2:].strip())
                    elif line and not line.startswith('#') and line[0].isdigit():
                        items.append(re.sub(r'^\d+[\.\)]\s*', '', line).strip())
                    elif line and line.startswith('**') and '**' in line[2:]:
                        items.append(line.strip())
                    elif not line:
                        break
        return items
    
    def _validate_and_normalize(self, result: dict) -> dict:
        """验证并规范化评估结果"""
        # 确保所有必需字段存在
        defaults = {
            "overall_score": 0.0,
            "dimensions": {},
            "passed": False,
            "pass_threshold": 80,
            "issues": [],
            "recommendations": []
        }
        
        for key, val in defaults.items():
            if key not in result or result[key] is None:
                result[key] = val
        
        # 确保各维度存在
        dim_defaults = {
            "code_quality": {"score": result["overall_score"] * 0.3 + 70, "weight": 0.3, "details": []},
            "functionality": {"score": result["overall_score"] * 0.3 + 70, "weight": 0.3, "details": []},
            "performance": {"score": result["overall_score"] * 0.2 + 80, "weight": 0.2, "details": []},
            "documentation": {"score": result["overall_score"] * 0.2 + 80, "weight": 0.2, "details": []}
        }
        
        for dim, default in dim_defaults.items():
            if dim not in result["dimensions"]:
                result["dimensions"][dim] = default
            else:
                dim_data = result["dimensions"][dim]
                if isinstance(dim_data, dict):
                    if "score" not in dim_data:
                        dim_data["score"] = result["overall_score"]
                    if "weight" not in dim_data:
                        dim_data["weight"] = default["weight"]
                    if "details" not in dim_data:
                        dim_data["details"] = []
                else:
                    # 如果维度是数字，转换为标准格式
                    result["dimensions"][dim] = {
                        "score": float(dim_data) if isinstance(dim_data, (int, float)) else result["overall_score"],
                        "weight": default["weight"],
                        "details": []
                    }
        
        # 计算加权总分（如果缺失）
        if result["overall_score"] == 0 and result["dimensions"]:
            weighted = sum(
                d["score"] * d["weight"] 
                for d in result["dimensions"].values() 
                if isinstance(d, dict)
            )
            result["overall_score"] = round(weighted, 1)
        
        # 规范化issues
        normalized_issues = []
        for issue in result["issues"]:
            if isinstance(issue, str):
                normalized_issues.append({
                    "severity": "medium",
                    "category": "style",
                    "file": "unknown",
                    "line": 0,
                    "description": issue
                })
            elif isinstance(issue, dict):
                normalized_issues.append({
                    "severity": issue.get("severity", "medium"),
                    "category": issue.get("category", "style"),
                    "file": issue.get("file", "unknown"),
                    "line": issue.get("line", 0),
                    "description": issue.get("description", str(issue))
                })
        result["issues"] = normalized_issues
        
        # 规范化recommendations
        normalized_recs = []
        for rec in result["recommendations"]:
            if isinstance(rec, str):
                normalized_recs.append({
                    "priority": "medium",
                    "category": "refactor",
                    "description": rec,
                    "effort": "medium"
                })
            elif isinstance(rec, dict):
                normalized_recs.append({
                    "priority": rec.get("priority", "medium"),
                    "category": rec.get("category", "refactor"),
                    "description": rec.get("description", str(rec)),
                    "effort": rec.get("effort", "medium")
                })
        result["recommendations"] = normalized_recs
        
        # 确保passed字段正确
        if result["overall_score"] < 0 or result["overall_score"] > 100:
            result["overall_score"] = max(0, min(100, result["overall_score"]))
        
        result["passed"] = result["overall_score"] >= result.get("pass_threshold", 80)
        
        return result
    
    def _default_result(self) -> dict:
        """默认结果"""
        return {
            "overall_score": 70.0,
            "dimensions": {
                "code_quality": {"score": 70, "weight": 0.3, "details": ["无法解析，使用默认值"]},
                "functionality": {"score": 70, "weight": 0.3, "details": ["无法解析，使用默认值"]},
                "performance": {"score": 70, "weight": 0.2, "details": ["无法解析，使用默认值"]},
                "documentation": {"score": 70, "weight": 0.2, "details": ["无法解析，使用默认值"]}
            },
            "passed": False,
            "pass_threshold": 80,
            "issues": [{"severity": "medium", "category": "style", "file": "unknown", "line": 0, "description": "评估结果解析失败"}],
            "recommendations": [{"priority": "high", "category": "document", "description": "检查评估输出格式", "effort": "small"}]
        }


class StaticAnalyzer:
    """静态代码分析器 - 不依赖LLM的评估"""
    
    @staticmethod
    def analyze_code(artifacts: List[dict]) -> dict:
        """对代码进行静态分析"""
        metrics = {
            "total_files": len(artifacts),
            "python_files": 0,
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "test_functions": 0,
            "docstring_count": 0,
            "type_hint_count": 0,
            "has_readme": False,
            "has_requirements": False,
            "has_tests": False,
            "syntax_errors": []
        }
        
        for artifact in artifacts:
            file_path = artifact.get("file_path", "")
            content = artifact.get("content", "")
            lines = content.split('\n')
            
            metrics["total_lines"] += len(lines)
            
            if file_path == "README.md":
                metrics["has_readme"] = True
            elif file_path == "requirements.txt":
                metrics["has_requirements"] = True
            elif file_path.startswith("tests/"):
                metrics["has_tests"] = True
            
            if file_path.endswith('.py'):
                metrics["python_files"] += 1
                
                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    
                    if not stripped:
                        metrics["blank_lines"] += 1
                    elif stripped.startswith('#'):
                        metrics["comment_lines"] += 1
                    else:
                        metrics["code_lines"] += 1
                        
                        if stripped.startswith('def ') and '(' in stripped:
                            metrics["functions"] += 1
                            if 'test_' in stripped:
                                metrics["test_functions"] += 1
                        elif stripped.startswith('class '):
                            metrics["classes"] += 1
                        elif stripped.startswith('import ') or stripped.startswith('from '):
                            metrics["imports"] += 1
                        
                        # 检测docstring
                        if '"""' in stripped or "'''" in stripped:
                            metrics["docstring_count"] += 1
                        
                        # 检测类型提示
                        if '-> ' in stripped or ': ' in stripped:
                            metrics["type_hint_count"] += 1
                
                # 语法检查
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    metrics["syntax_errors"].append(f"{file_path}:{e.lineno}: {e.msg}")
        
        # AST-based graph metrics (robust to syntax errors)
        graph_metrics = CodeGraphAnalyzer.analyze(artifacts)
        metrics["graph_avg_cyclomatic_complexity"] = graph_metrics["avg_cyclomatic_complexity"]
        metrics["graph_max_cyclomatic_complexity"] = graph_metrics["max_cyclomatic_complexity"]
        metrics["graph_unused_import_count"] = graph_metrics["unused_import_count"]
        metrics["graph_import_cycle_count"] = graph_metrics["import_cycle_count"]
        metrics["graph_test_to_code_function_ratio"] = graph_metrics["test_to_code_function_ratio"]

        return metrics
    
    @staticmethod
    def compute_dimension_scores(metrics: dict) -> dict:
        """基于静态分析计算维度分数"""
        scores = {}
        
        # 代码质量（基于PEP8、类型提示、文档等）
        code_quality = 50.0  # 基础分
        if metrics["python_files"] > 0:
            code_lines = metrics["code_lines"]
            if code_lines > 0:
                doc_ratio = metrics["docstring_count"] / max(code_lines, 1) * 100
                hint_ratio = metrics["type_hint_count"] / max(metrics["functions"], 1) * 100
                comment_ratio = metrics["comment_lines"] / max(code_lines, 1) * 100
                
                code_quality += min(20, doc_ratio * 2)
                code_quality += min(15, hint_ratio * 0.3)
                code_quality += min(10, comment_ratio * 2)
                code_quality -= len(metrics["syntax_errors"]) * 10
        
        if not metrics["syntax_errors"]:
            code_quality += 10
        
        scores["code_quality"] = min(100, max(0, code_quality))
        
        # 功能完整性（基于测试、结构）
        functionality = 50.0
        if metrics["has_tests"]:
            functionality += 20
        if metrics["test_functions"] > 0:
            functionality += min(20, metrics["test_functions"] * 5)
        if metrics["functions"] > 0:
            functionality += 10
        scores["functionality"] = min(100, functionality)
        
        # 性能（简单估计）
        performance = 70.0
        if metrics["python_files"] > 0:
            performance = 75.0
        scores["performance"] = performance
        
        # 文档质量
        documentation = 40.0
        if metrics["has_readme"]:
            documentation += 30
        if metrics["has_requirements"]:
            documentation += 10
        if metrics["docstring_count"] > 0:
            documentation += min(20, metrics["docstring_count"] * 2)
        scores["documentation"] = min(100, documentation)
        
        return scores


# ===== 主函数 =====

def evaluator_node(state: SharedState) -> SharedState:
    """评估节点 - P1优化版 + P0修复 + ML 自动化评估"""
    import os
    from src.ml.service import MLEvaluationService
    from src.ml.dataset import EvaluationDataset
    from src.ml.features import FeatureExtractor

    logger.info("[E] 开始质量评估")

    state["current_phase"] = "evaluation"
    state["status"] = "evaluating"

    # 先进行静态分析
    static_metrics = StaticAnalyzer.analyze_code(state.get("code_artifacts", []))
    state["static_metrics"] = static_metrics

    # 提取 ML 特征
    try:
        ml_features = FeatureExtractor.extract(state)
        state["ml_features"] = ml_features
    except Exception as e:
        logger.warning(f"[E] ML 特征提取失败: {e}")
        state["ml_features"] = None

    logger.info(f"[E] 静态分析: {static_metrics['total_files']} 文件, "
                f"{static_metrics['functions']} 函数, "
                f"{static_metrics['test_functions']} 测试")

    # 调用 LLM 评估
    try:
        result = _evaluate_with_llm(state, static_metrics)
    except Exception as e:
        logger.error(f"[E] LLM 评估失败: {e}")
        result = _evaluate_with_static_analysis(state, static_metrics)

    # 标记来源
    result["source"] = result.get("source", "llm")

    # ML 评估融合
    ml_enabled = settings.ml_eval_enabled
    ml_mode = settings.ml_eval_mode
    weight_ml = settings.ml_eval_weight_ml

    if ml_enabled:
        try:
            ml_service = MLEvaluationService(mode=ml_mode, weight_ml=weight_ml)
            result = ml_service.evaluate(state, result)
            state["ml_score"] = result.get("ml_score")
            state["ml_model_id"] = result.get("ml_model_id")
            state["ml_mode"] = result.get("ml_mode")
        except Exception as e:
            logger.warning(f"[E] ML 评估服务异常: {e}")

    # P0修复：根据测试是否通过调整 passed 状态
    test_results = state.get("test_results", [])
    all_tests_passed = all(r.get("passed", False) for r in test_results) if test_results else False

    # 如果测试全部通过，但评估分数低于阈值，考虑降低阈值
    if all_tests_passed and not result.get("passed", False):
        score = result.get("overall_score", 0)
        # 测试通过且分数 >= 50，认为是简单项目，通过
        if score >= 50:
            logger.info(f"[E] 测试全部通过但分数 {score} 低于阈值，调整为通过（简单项目）")
            result["passed"] = True
            result["pass_threshold"] = 50  # 记录实际使用的阈值

    # 如果测试失败，即使分数高也不通过
    if not all_tests_passed and result.get("passed", False):
        logger.warning("[E] 测试失败，强制标记为未通过")
        result["passed"] = False

    state["evaluation_result"] = result
    state["evaluation_source"] = result.get("source", "llm")
    state["evaluation_history"] = state.get("evaluation_history", []) + [{
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "result": result
    }]

    # 样本回流：将本次评估结果写入数据集
    try:
        dataset = EvaluationDataset()
        dataset.append(state, result)
        state["dataset_saved"] = True
    except Exception as e:
        logger.warning(f"[E] 评估样本回流失败: {e}")

    logger.info(f"[E] 评估完成: {result.get('overall_score', 0)}/100, "
                f"通过: {result.get('passed', False)}, "
                f"来源: {state.get('evaluation_source', 'unknown')}")

    return state


def _evaluate_with_llm(state: SharedState, static_metrics: dict) -> dict:
    """调用 LLM 评估代码质量 - P1优化版（强制JSON）+ MCP"""
    from src.mcp.bridge import create_mcp_client_for_state, ask_with_tools

    llm = get_llm()

    # 构建代码上下文（限制长度）
    code_context = ""
    for artifact in state.get("code_artifacts", []):
        content = artifact['content'][:300]  # 每个文件最多300字符
        code_context += f"\n=== {artifact['file_path']} ===\n{content}\n"

    # 构建测试结果（P0修复：让LLM知道测试是否通过）
    test_context = ""
    test_results = state.get("test_results", [])
    all_passed = all(r.get("passed", False) for r in test_results) if test_results else False
    for tr in test_results:
        test_context += f"\n测试: {tr.get('file', '')} | 通过: {tr.get('passed', False)} | "
        test_context += f"耗时: {tr.get('duration_ms', 0)}ms"
        if not tr.get("passed", False):
            err = tr.get("error", "")[:300]
            out = tr.get("output", "")[:300]
            test_context += f"\n  错误: {err}\n  输出: {out}"

    # 静态分析摘要
    static_summary = f"""
静态分析摘要:
- 文件数: {static_metrics['total_files']}
- Python文件: {static_metrics['python_files']}
- 代码行数: {static_metrics['code_lines']}
- 函数数: {static_metrics['functions']}
- 测试函数: {static_metrics['test_functions']}
- 文档字符串: {static_metrics['docstring_count']}
- 类型提示: {static_metrics['type_hint_count']}
- 语法错误: {len(static_metrics['syntax_errors'])}
- 包含README: {static_metrics['has_readme']}
- 包含requirements: {static_metrics['has_requirements']}
- 包含测试: {static_metrics['has_tests']}
- 测试是否全部通过: {all_passed}
"""

    # 判断是否为简单项目
    requirements = state.get("requirements", "")
    is_simple = any(kw in requirements.lower() for kw in [
        "计算器", "calculator", "todo", "增删改查", "crud", "简单", "simple",
        "hello world", "示例", "example"
    ])

    simple_note = "\n\n这是一个简单项目（如计算器、TODO列表等），评分标准可适当放宽。测试通过是最高优先级。" if is_simple else ""

    # RAG：检索相似评估报告
    rag_context = ""
    try:
        rag = RAGRetriever()
        similar_evals = rag.find_similar_evaluations(requirements, top_k=2)
        if similar_evals:
            rag_context = "\n\n相似历史项目评估参考（供评分一致性参考）：\n" + rag.format_context(similar_evals)
            logger.info(f"[E] RAG 检索到 {len(similar_evals)} 条相似评估")
    except Exception as e:
        logger.warning(f"[E] RAG 检索失败: {e}")

    user_prompt = f"""请评估以下代码质量。

需求：
{state.get('requirements', 'N/A')}

代码文件：
{code_context}

测试结果：
{test_context}

{static_summary}
{rag_context}

重要：请只输出JSON格式，不要包含其他内容。JSON必须包含所有字段（overall_score、dimensions、passed、issues、recommendations）。
{simple_note}
"""

    tool_calls = []
    tool_results = []
    try:
        with create_mcp_client_for_state(state) as mcp_client:
            response = ask_with_tools(
                llm=llm,
                mcp_client=mcp_client,
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=1.0,
                max_tokens=2048,
                max_rounds=2,
            )
            tool_calls = response.get("tool_calls", [])
            tool_results = response.get("tool_results", [])
    except Exception as e:
        logger.warning(f"[E] MCP 工具调用失败，回退普通 LLM: {e}")
        response = llm.chat(
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=1.0,
            max_tokens=2048
        )

    state["tool_calls"] = state.get("tool_calls", []) + tool_calls
    state["tool_results"] = state.get("tool_results", []) + tool_results

    # 记录token使用
    state["llm_usage"] = state.get("llm_usage", []) + [{
        "phase": "evaluation",
        "input_tokens": response.get("input_tokens", 0) if isinstance(response, dict) else 0,
        "output_tokens": response.get("output_tokens", 0) if isinstance(response, dict) else 0,
        "total_tokens": response.get("total_tokens", 0) if isinstance(response, dict) else 0,
        "cost_usd": response.get("cost_usd", 0.0) if isinstance(response, dict) else 0.0,
        "elapsed_ms": response.get("elapsed_ms", 0) if isinstance(response, dict) else 0
    }]

    # 使用解析器
    content = response.get("content", "") if isinstance(response, dict) else response
    parser = EvaluationParser(content)
    return parser.parse()


def _evaluate_with_static_analysis(state: SharedState, static_metrics: dict) -> dict:
    """基于静态分析的降级评估 - P0修复：考虑测试结果"""
    logger.info("[E] 使用静态分析进行降级评估")
    
    scores = StaticAnalyzer.compute_dimension_scores(static_metrics)
    
    overall = (
        scores["code_quality"] * 0.3 +
        scores["functionality"] * 0.3 +
        scores["performance"] * 0.2 +
        scores["documentation"] * 0.2
    )
    
    # 考虑测试结果
    test_results = state.get("test_results", [])
    all_tests_passed = all(r.get("passed", False) for r in test_results) if test_results else False
    
    issues = []
    if static_metrics["syntax_errors"]:
        for err in static_metrics["syntax_errors"]:
            issues.append({
                "severity": "high",
                "category": "syntax",
                "file": err.split(':')[0],
                "line": int(err.split(':')[1]) if ':' in err else 0,
                "description": err
            })
    if not static_metrics["has_tests"]:
        issues.append({
            "severity": "high",
            "category": "functionality",
            "file": "tests/",
            "line": 0,
            "description": "缺少测试文件"
        })
    if not all_tests_passed and test_results:
        issues.append({
            "severity": "high",
            "category": "functionality",
            "file": "tests/",
            "line": 0,
            "description": "测试未通过"
        })
    if not static_metrics["has_readme"]:
        issues.append({
            "severity": "medium",
            "category": "style",
            "file": "README.md",
            "line": 0,
            "description": "缺少README文档"
        })
    if not static_metrics["docstring_count"]:
        issues.append({
            "severity": "low",
            "category": "style",
            "file": "",
            "line": 0,
            "description": "缺少文档字符串"
        })
    
    recommendations = [
        {"priority": "high" if not static_metrics["has_tests"] else "medium", "category": "test", "description": "添加pytest测试用例", "effort": "medium"},
        {"priority": "medium", "category": "document", "description": "补充README和API文档", "effort": "small"},
        {"priority": "medium", "category": "refactor", "description": "添加类型提示", "effort": "small"}
    ]
    
    # P0修复：根据测试通过情况和项目复杂度调整阈值
    requirements = state.get("requirements", "")
    is_simple = any(kw in requirements.lower() for kw in [
        "计算器", "calculator", "todo", "增删改查", "crud", "简单", "simple",
        "hello world", "示例", "example"
    ])
    
    pass_threshold = 50 if (is_simple and all_tests_passed) else 80
    
    return {
        "overall_score": round(overall, 1),
        "dimensions": {
            "code_quality": {"score": scores["code_quality"], "weight": 0.3, "details": ["基于静态分析"]},
            "functionality": {"score": scores["functionality"], "weight": 0.3, "details": ["基于静态分析"]},
            "performance": {"score": scores["performance"], "weight": 0.2, "details": ["基于静态分析"]},
            "documentation": {"score": scores["documentation"], "weight": 0.2, "details": ["基于静态分析"]}
        },
        "passed": overall >= pass_threshold and all_tests_passed,
        "pass_threshold": pass_threshold,
        "issues": issues,
        "recommendations": recommendations,
        "source": "static_analysis"
    }


def generate_html_report(state: SharedState) -> str:
    """生成HTML评估报告 - P1优化版"""
    result = state.get("evaluation_result")
    if not result:
        return "<p>暂无评估结果</p>"
    
    project = state.get("project", {})
    project_name = project.get("name", "Unknown") if project else "Unknown"
    
    dimensions = result.get("dimensions", {})
    
    # 构建维度评分HTML
    dims_html = ""
    for dim_name, dim_data in dimensions.items():
        if isinstance(dim_data, dict):
            score = dim_data.get("score", 0)
            weight = dim_data.get("weight", 0.25) * 100
            details = dim_data.get("details", [])
            details_html = "<ul>" + "".join(f"<li>{d}</li>" for d in details[:3]) + "</ul>" if details else ""
            
            dims_html += f"""
            <div class="dimension">
                <h3>{dim_name} ({weight:.0f}%)</h3>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score}%">{score}/100</div>
                </div>
                {details_html}
            </div>
            """
    
    # 构建问题列表
    issues = result.get("issues", [])
    issues_html = ""
    severity_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}
    for issue in issues:
        if isinstance(issue, dict):
            sev = issue.get("severity", "medium")
            color = severity_colors.get(sev, "#f39c12")
            issues_html += f"""
            <li style="border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;">
                <strong>[{sev.upper()}]</strong> {issue.get("category", "general")}
                {f' - {issue.get("file", "")}:{issue.get("line", 0)}' if issue.get("file") != "unknown" else ""}
                <br>{issue.get("description", "")}
            </li>
            """
        else:
            issues_html += f"<li>{issue}</li>"
    
    # 构建建议列表
    recommendations = result.get("recommendations", [])
    recs_html = ""
    priority_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}
    for rec in recommendations:
        if isinstance(rec, dict):
            pri = rec.get("priority", "medium")
            color = priority_colors.get(pri, "#f39c12")
            effort = rec.get("effort", "medium")
            recs_html += f"""
            <li style="border-left: 4px solid {color}; padding-left: 10px; margin: 5px 0;">
                <strong>[{pri.upper()}]</strong> {rec.get("category", "general")}
                <span class="effort">({effort} effort)</span>
                <br>{rec.get("description", "")}
            </li>
            """
        else:
            recs_html += f"<li>{rec}</li>"
    
    passed = result.get("passed", False)
    score = result.get("overall_score", 0)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>评估报告 - {project_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .score-card {{ text-align: center; padding: 30px; margin: 20px 0; border-radius: 10px; }}
        .pass {{ background: #d4edda; color: #155724; }}
        .fail {{ background: #f8d7da; color: #721c24; }}
        .score {{ font-size: 64px; font-weight: bold; margin: 10px 0; }}
        .dimension {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .score-bar {{ width: 100%; height: 30px; background: #e9ecef; border-radius: 15px; overflow: hidden; }}
        .score-fill {{ height: 100%; background: #3498db; color: white; text-align: center; line-height: 30px; border-radius: 15px; transition: width 0.3s; }}
        .score-fill.high {{ background: #27ae60; }}
        .score-fill.medium {{ background: #f39c12; }}
        .score-fill.low {{ background: #e74c3c; }}
        ul {{ list-style: none; padding: 0; }}
        li {{ margin: 8px 0; padding: 8px; background: white; border-radius: 4px; }}
        .effort {{ color: #7f8c8d; font-size: 0.9em; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ padding: 15px; background: #e8f4f8; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>质量评估报告</h1>
        <div class="score-card {'pass' if passed else 'fail'}">
            <div class="score">{score}/100</div>
            <div style="font-size: 24px;">{'✅ 通过' if passed else '❌ 未通过'}</div>
            <div style="color: #666; margin-top: 10px;">
                阈值: {result.get('pass_threshold', 80)}分 | 
                来源: {result.get('source', 'llm')}
            </div>
        </div>
        
        <h2>维度评分</h2>
        {dims_html}
        
        <h2>问题列表</h2>
        <ul>{issues_html}</ul>
        
        <h2>改进建议</h2>
        <ul>{recs_html}</ul>
    </div>
</body>
</html>"""
    return html


def compare_evaluations(history: List[dict]) -> dict:
    """对比多次评估历史，分析改进趋势"""
    if len(history) < 2:
        return {"message": "需要至少2次评估才能对比"}
    
    comparisons = []
    for i in range(1, len(history)):
        prev = history[i-1].get("result", {})
        curr = history[i].get("result", {})
        
        prev_score = prev.get("overall_score", 0)
        curr_score = curr.get("overall_score", 0)
        delta = curr_score - prev_score
        
        comparison = {
            "iteration": i,
            "previous_score": prev_score,
            "current_score": curr_score,
            "delta": round(delta, 1),
            "improved": delta > 0,
            "passed_changed": prev.get("passed") != curr.get("passed")
        }
        comparisons.append(comparison)
    
    return {
        "comparisons": comparisons,
        "total_iterations": len(history),
        "best_score": max(h.get("result", {}).get("overall_score", 0) for h in history),
        "final_passed": history[-1].get("result", {}).get("passed", False)
    }
