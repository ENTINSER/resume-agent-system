"""B 阶段跨分片一致性检查测试"""

from src.core.B import (
    _extract_project_definitions,
    _find_test_mismatches,
    _fix_consistency_mismatches,
)


def test_extract_project_definitions_uses_dotted_modules():
    artifacts = [
        {"file_path": "src/agents.py", "content": "def run(): pass\n"},
        {"file_path": "src/graph.py", "content": "class Graph: pass\n"},
    ]
    defs = _extract_project_definitions(artifacts)
    assert "src.agents" in defs
    assert "src.graph" in defs
    assert "run" in defs["src.agents"]
    assert "Graph" in defs["src.graph"]


def test_package_imports_not_mismatched():
    """from src.agents import run 不应被报告为 module_missing"""
    artifacts = [
        {"file_path": "src/agents.py", "content": "def run(): pass\n"},
        {
            "file_path": "tests/test_agents.py",
            "content": "from src.agents import run\n\ndef test_run():\n    run()\n",
        },
    ]
    defs = _extract_project_definitions(artifacts)
    mismatches = _find_test_mismatches(artifacts, defs)
    assert not mismatches


def test_attribute_mismatch_still_detected():
    """from src.agents import missing_func 应报告 attribute 缺失"""
    artifacts = [
        {"file_path": "src/agents.py", "content": "def run(): pass\n"},
        {
            "file_path": "tests/test_agents.py",
            "content": "from src.agents import missing_func\n",
        },
    ]
    defs = _extract_project_definitions(artifacts)
    mismatches = _find_test_mismatches(artifacts, defs)
    assert len(mismatches) == 1
    assert mismatches[0]["type"] == "attribute"
    assert mismatches[0]["module"] == "src.agents"


def test_no_flattening_package_on_auto_rename():
    """自动重命名不应把 src/agents.py 拍平成 src.py"""
    artifacts = [
        {"file_path": "src/agents.py", "content": "def run(): pass\n"},
        {
            "file_path": "tests/test_agents.py",
            "content": "import src.agents\n\ndef test_run():\n    src.agents.run()\n",
        },
    ]
    defs = _extract_project_definitions(artifacts)
    mismatches = _find_test_mismatches(artifacts, defs)
    # 如果存在 module_missing，修复时也不应拍平
    fixed = _fix_consistency_mismatches(artifacts, mismatches, "req", "proj")
    paths = {a["file_path"] for a in fixed}
    assert "src/agents.py" in paths
    assert "src.py" not in paths
