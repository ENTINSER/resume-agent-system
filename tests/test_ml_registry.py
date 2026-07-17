"""ML 模型仓库单元测试"""

import tempfile

import pytest

from src.ml.registry import ModelRegistry, STAGE_SHADOW, STAGE_PRODUCTION, STAGE_STAGING


class TestModelRegistry:
    @pytest.fixture
    def registry(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        reg = ModelRegistry(registry_path=path)
        yield reg
        import os
        os.unlink(path)

    def test_register_and_list(self, registry):
        registry.register(
            experiment_id="exp-001",
            model_path="/tmp/model-001",
            metrics={"val": {"mae": 5.0}},
            stage=STAGE_STAGING,
        )
        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "exp-001"

    def test_promote_to_production(self, registry):
        registry.register("exp-001", "/tmp/m1", {}, stage=STAGE_PRODUCTION)
        registry.register("exp-002", "/tmp/m2", {}, stage=STAGE_STAGING)
        registry.promote("exp-002", STAGE_PRODUCTION)

        prod = registry.get_production_model()
        assert prod["id"] == "exp-002"

        old = registry.get_model("exp-001")
        assert old["stage"] == STAGE_STAGING

    def test_get_nonexistent(self, registry):
        assert registry.get_model("not-exist") is None
