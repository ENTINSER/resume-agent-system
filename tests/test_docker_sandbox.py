"""DockerSandbox 单元测试

使用 Mock 验证沙箱默认安全参数、run_tests 命令序列和结果判定逻辑。
不依赖真实 Docker 守护进程。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.docker_sandbox import DockerSandbox, check_docker_available


@pytest.fixture
def mock_docker_client():
    """构造一个 Mock Docker 客户端"""
    client = Mock()
    client.images.get.return_value = Mock()

    container = Mock()
    container.id = "container123"
    container.wait.return_value = {"StatusCode": 0}
    container.logs.side_effect = lambda stdout=True, stderr=False: (
        b"test passed" if stdout else b""
    )

    client.containers.run.return_value = container
    return client


class TestDockerSandboxInit:
    """测试沙箱初始化"""

    def test_default_network_disabled(self, mock_docker_client):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox()
            assert sandbox.network_disabled is True

    def test_custom_network_enabled(self, mock_docker_client):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox(network_disabled=False)
            assert sandbox.network_disabled is False


class TestRunTestsCommandSequence:
    """测试 run_tests 执行的命令序列"""

    def test_skips_pip_when_network_disabled(self, mock_docker_client, tmp_path):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox(network_disabled=True)
            sandbox.temp_dir = str(tmp_path)
            # 创建 requirements.txt，声明已预装的 pytest
            (tmp_path / "requirements.txt").write_text("pytest\n")

            sandbox.run_tests()

            calls = [call.kwargs.get("command", "") for call in mock_docker_client.containers.run.call_args_list]
            # 禁网模式下不应执行 pip install
            assert not any("pip install" in cmd for cmd in calls)
            # 应执行 pytest 命令
            assert any("pytest -v" in cmd for cmd in calls)

    def test_allows_pip_when_network_enabled(self, mock_docker_client, tmp_path):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox(network_disabled=False)
            sandbox.temp_dir = str(tmp_path)
            (tmp_path / "requirements.txt").write_text("pytest\n")
            sandbox.run_tests()

            calls = [call.kwargs.get("command", "") for call in mock_docker_client.containers.run.call_args_list]
            assert any("requirements.txt" in cmd for cmd in calls)

    def test_rejects_missing_preinstalled_dep_when_network_disabled(self, mock_docker_client, tmp_path):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox(network_disabled=True)
            sandbox.temp_dir = str(tmp_path)
            # 声明一个未预装的依赖
            (tmp_path / "requirements.txt").write_text("django\n")

            result = sandbox.run_tests()

            assert result["passed"] is False
            assert "django" in result["error"]
            # 不应启动容器
            mock_docker_client.containers.run.assert_not_called()

    def test_security_params_passed(self, mock_docker_client, tmp_path):
        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox()
            sandbox.temp_dir = str(tmp_path)
            sandbox.run_tests()

            kwargs = mock_docker_client.containers.run.call_args.kwargs
            assert kwargs["network_disabled"] is True
            assert kwargs["security_opt"] == ["no-new-privileges:true"]
            assert kwargs["cap_drop"] == ["ALL"]
            assert kwargs["read_only"] is True
            assert "/tmp" in kwargs["tmpfs"]


class TestRunTestsResultLogic:
    """测试 run_tests 结果判定"""

    def test_success_when_exit_code_zero(self, mock_docker_client, tmp_path):
        mock_docker_client.containers.run.return_value.wait.return_value = {"StatusCode": 0}
        mock_docker_client.containers.run.return_value.logs.side_effect = lambda stdout=True, stderr=False: (
            b"1 passed" if stdout else b""
        )

        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox()
            sandbox.temp_dir = str(tmp_path)
            result = sandbox.run_tests()

        assert result["success"] is True

    def test_failure_when_exit_code_nonzero(self, mock_docker_client, tmp_path):
        mock_docker_client.containers.run.return_value.wait.return_value = {"StatusCode": 1}
        mock_docker_client.containers.run.return_value.logs.side_effect = lambda stdout=True, stderr=False: (
            b"1 failed" if stdout else b"AssertionError"
        )

        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox()
            sandbox.temp_dir = str(tmp_path)
            result = sandbox.run_tests()

        assert result["success"] is False

    def test_failure_when_failed_keyword_present(self, mock_docker_client, tmp_path):
        # exit code 0 但日志里有 FAILED，应判定为失败
        mock_docker_client.containers.run.return_value.wait.return_value = {"StatusCode": 0}
        mock_docker_client.containers.run.return_value.logs.side_effect = lambda stdout=True, stderr=False: (
            b"1 FAILED" if stdout else b""
        )

        with patch("src.core.docker_sandbox.docker.from_env", return_value=mock_docker_client):
            sandbox = DockerSandbox()
            sandbox.temp_dir = str(tmp_path)
            result = sandbox.run_tests()

        assert result["success"] is False


class TestCheckDockerAvailable:
    """测试 Docker 可用性检查"""

    def test_available(self):
        client = Mock()
        client.ping.return_value = True
        with patch("src.core.docker_sandbox.docker.from_env", return_value=client):
            assert check_docker_available() is True

    def test_unavailable(self):
        with patch("src.core.docker_sandbox.docker.from_env", side_effect=Exception("no docker")):
            assert check_docker_available() is False
