"""Prometheus 指标单元测试"""

from src.core.metrics import observe_task, observe_llm, observe_phase


class TestMetrics:
    """测试指标函数可正常调用，不抛异常"""

    def test_observe_task(self):
        observe_task(status="completed", duration_seconds=1.5)

    def test_observe_llm(self):
        observe_llm(phase="analysis", tokens=100, cost=0.001)

    def test_observe_phase(self):
        observe_phase(phase="development")
