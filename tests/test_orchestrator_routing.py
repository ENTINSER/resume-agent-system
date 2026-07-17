"""Orchestrator 路由单元测试

Mock B/E/C 节点，验证 D 的状态流转。
"""

import pytest
from unittest.mock import patch, Mock

from src.core.D import Orchestrator
from src.core.shared_state import create_initial_state


@pytest.fixture
def orchestrator():
    """返回一个 UI 模式关闭的编排器"""
    return Orchestrator(ui_mode=False)


@pytest.fixture
def requirements():
    return "开发一个 Python 计算器，支持加减乘除"


class TestAnalyzeNode:
    """测试需求分析节点"""

    def test_analyze_creates_project(self, orchestrator, requirements):
        state = create_initial_state(requirements)
        result = orchestrator._analyze_node(state)

        assert result["project"] is not None
        assert "id" in result["project"]
        assert "name" in result["project"]
        assert "tech_stack" in result["project"]
        assert "milestones" in result["project"]
        assert result["current_phase"] == "analysis"

    def test_route_after_analysis_error(self, orchestrator, requirements):
        state = create_initial_state(requirements)
        state["error"] = "some error"
        route = orchestrator._route_after_analysis(state)
        assert route == "human_review"

    def test_route_after_analysis_normal(self, orchestrator, requirements):
        state = create_initial_state(requirements)
        route = orchestrator._route_after_analysis(state)
        assert route == "develop"


class TestEvaluationRouting:
    """测试评估后的路由"""

    def test_route_after_evaluation_passed_no_training(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["iteration_count"] = 1
        state["evaluation_result"] = {"passed": True}
        route = orchestrator._route_after_evaluation(state)
        assert route == "human_review"

    def test_route_after_evaluation_passed_with_training_keyword(self, orchestrator):
        state = create_initial_state("微调一个模型")
        state["iteration_count"] = 1
        state["evaluation_result"] = {"passed": True}
        route = orchestrator._route_after_evaluation(state)
        assert route == "train"

    def test_route_after_evaluation_failed_under_max_iter(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["iteration_count"] = 1
        state["max_iterations"] = 10
        state["evaluation_result"] = {"passed": False}
        route = orchestrator._route_after_evaluation(state)
        assert route == "rewrite"

    def test_route_after_evaluation_failed_over_max_iter(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["iteration_count"] = 10
        state["max_iterations"] = 10
        state["evaluation_result"] = {"passed": False}
        route = orchestrator._route_after_evaluation(state)
        assert route == "end"

    def test_route_after_evaluation_no_result(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["evaluation_result"] = None
        route = orchestrator._route_after_evaluation(state)
        assert route == "rewrite"


class TestHumanReviewRouting:
    """测试人类审核后的路由"""

    def test_route_after_human_approve(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["human_approved"] = True
        route = orchestrator._route_after_human(state)
        assert route == "end"

    def test_route_after_human_train(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["human_approved"] = False
        state["human_feedback"] = "训练"
        route = orchestrator._route_after_human(state)
        assert route == "train"

    def test_route_after_human_rewrite(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["human_approved"] = False
        state["human_feedback"] = "rewrite"
        route = orchestrator._route_after_human(state)
        assert route == "continue"

    def test_route_after_human_evaluate(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["human_approved"] = False
        state["human_feedback"] = "evaluate"
        route = orchestrator._route_after_human(state)
        assert route == "evaluate"

    def test_route_after_human_default_continue(self, orchestrator):
        state = create_initial_state("开发一个计算器")
        state["human_approved"] = False
        state["human_feedback"] = ""
        route = orchestrator._route_after_human(state)
        assert route == "continue"


class TestWorkflowBuild:
    """测试工作流图构建"""

    def test_workflow_compiles(self, orchestrator):
        assert orchestrator.workflow is not None


class TestRunWithMockedNodes:
    """使用 Mock 节点验证完整流程"""

    def test_full_flow(self, requirements, monkeypatch):
        # Mock 开发节点：生成一个简单产物
        def mock_develop(s):
            s["code_artifacts"] = [{"file_path": "main.py", "content": "print(1)"}]
            s["test_results"] = [{"file": "docker_tests", "passed": True}]
            s["current_phase"] = "development"
            return s

        # Mock 评估节点：通过
        def mock_evaluate(s):
            s["evaluation_result"] = {"passed": True, "overall_score": 90}
            s["current_phase"] = "evaluation"
            return s

        # Mock 训练节点：跳过
        def mock_train(s):
            s["current_phase"] = "training"
            return s

        # Mock 人类审核：直接批准
        def mock_human_review(s):
            s["human_approved"] = True
            s["human_feedback"] = "approve"
            s["current_phase"] = "human_review"
            return s

        # 在 patch 上下文中新建 Orchestrator，确保图构建时拿到 Mock 函数
        with patch("src.core.B.developer_node", side_effect=mock_develop), \
             patch("src.core.E.evaluator_node", side_effect=mock_evaluate), \
             patch("src.core.C.trainer_node", side_effect=mock_train), \
             patch("src.core.D.Orchestrator._human_review_node", side_effect=mock_human_review), \
             patch("src.core.persistence.SessionStore.save_session"):

            orchestrator = Orchestrator(ui_mode=False)
            result = orchestrator.run(requirements)

        assert result["status"] in ("human_review", "done", "complete")
        assert result["evaluation_result"]["passed"] is True
