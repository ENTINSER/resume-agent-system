"""Docker 沙箱工具 - B 开发工程师的代码执行环境

安全隔离：
- 代码在 Docker 容器内执行
- 限制 CPU/内存资源
- 默认禁止网络访问（network_disabled=True）：当禁网时，curl/wget 等命令无法连接外部网络
- 超时控制
- 执行后自动清理
"""

import os
import re
import tempfile
import shutil
import time
from typing import List, Dict, Optional
from pathlib import Path

import docker
from docker.errors import DockerException

from src.core.logger import logger


# 沙箱镜像预装的包（与 Dockerfile 保持一致）
# 禁网模式下，requirements.txt 只能声明此列表内的依赖，否则测试会明确报错
PREINSTALLED_PACKAGES = {
    "pytest", "pytest-cov", "pytest-asyncio", "black", "flake8", "mypy", "numpy",
    "requests", "beautifulsoup4", "bs4", "pandas", "pydantic", "typer",
    "pyyaml", "yaml", "python-dotenv", "dotenv",
    # LangGraph 生态（M9 项目 01 预装）
    "langgraph", "langchain", "langchain-core", "langchain-openai", "langchain-community",
    "openai", "langsmith",
}


class DockerSandbox:
    """Docker 沙箱 - 安全执行代码"""
    
    def __init__(self, 
                 image: str = "agent-sandbox:latest",
                 fallback_image: str = "python:3.11-slim",
                 cpu_limit: float = 1.0,
                 memory_limit: str = "512m",
                 timeout: int = 60,
                 network_disabled: bool = True):
        # 检查主镜像是否存在，不存在则使用 fallback
        self.image = self._resolve_image(image, fallback_image)
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        
        # Docker 客户端
        try:
            self.client = docker.from_env()
            logger.info("[Docker] Docker 客户端初始化成功")
        except DockerException as e:
            logger.error(f"[Docker] Docker 连接失败: {e}")
            self.client = None
        
        # 临时目录
        self.temp_dir: Optional[str] = None
        self.container_id: Optional[str] = None
    
    def _resolve_image(self, primary: str, fallback: str) -> str:
        """检查镜像是否存在，返回可用的镜像名。

        不使用 fallback 镜像：fallback 镜像缺少 pytest 等预装依赖，
        一旦回退只会导致测试直接失败并浪费迭代。如果主镜像不可用，
        直接报错让上层处理。
        """
        client = docker.from_env()
        for attempt in range(1, 4):
            try:
                # 注意：在 Docker Desktop 的 containerd 镜像存储模式下，
                # client.images.get(name) 可能返回 404，但 images.list(name=...)
                # 可以正确找到镜像。因此优先使用 list 检测。
                found = client.images.list(name=primary)
                if found:
                    logger.info(f"[Docker] 使用镜像: {primary}")
                    return primary
            except Exception as exc:
                logger.warning(
                    f"[Docker] 镜像 {primary} 检测失败（第 {attempt}/3 次）: {exc}"
                )
                if attempt < 3:
                    time.sleep(1.0)
        raise RuntimeError(
            f"Docker 镜像 {primary} 不可用，且已禁用 fallback。"
            f"请运行 `docker build -f Dockerfile.python -t agent-sandbox:latest .` 重新构建。"
        )
    
    def create_workspace(self) -> str:
        """创建临时工作目录"""
        self.temp_dir = tempfile.mkdtemp(prefix="agent_sandbox_")
        logger.info(f"[Docker] 创建工作目录: {self.temp_dir}")
        return self.temp_dir
    
    def write_files(self, artifacts: List[dict]) -> None:
        """将代码写入工作目录"""
        if not self.temp_dir:
            self.create_workspace()
        
        for artifact in artifacts:
            file_path = os.path.join(self.temp_dir, artifact.get("file_path", ""))
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(artifact.get("content", ""))
            
            logger.info(f"[Docker] 写入文件: {artifact.get('file_path', '')}")
    
    def run_command(self, command: str) -> Dict:
        """在 Docker 容器内执行命令 - P0修复：更详细的日志捕获"""
        if not self.client:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Docker 不可用",
                "duration_ms": 0
            }
        
        if not self.temp_dir:
            self.create_workspace()
        
        try:
            # 运行容器
            logger.info(f"[Docker] 执行命令: {command}")
            start_time = time.time()
            
            container = self.client.containers.run(
                image=self.image,
                command=f"sh -c 'cd /workspace && {command}'",
                volumes={self.temp_dir: {"bind": "/workspace", "mode": "rw"}},
                cpu_quota=int(self.cpu_limit * 100000),
                mem_limit=self.memory_limit,
                network_disabled=self.network_disabled,
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                read_only=True,
                tmpfs={"/tmp": "noexec,nosuid,size=100m"},
                detach=True,
                working_dir="/workspace"
            )
            
            self.container_id = container.id
            
            # 等待完成（带超时）
            try:
                result = container.wait(timeout=self.timeout)
                duration_ms = int((time.time() - start_time) * 1000)
                
                # P0修复：分别捕获 stdout 和 stderr
                stdout_logs = container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace')
                stderr_logs = container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace')
                
                # 合并用于兼容性
                all_logs = stdout_logs + ("\n" + stderr_logs if stderr_logs else "")
                
                status_code = result.get("StatusCode", -1)
                
                logger.debug(f"[Docker] 命令返回码: {status_code}")
                if stderr_logs and status_code != 0:
                    logger.debug(f"[Docker] stderr: {stderr_logs[:500]}")
                
                return {
                    "success": status_code == 0,
                    "returncode": status_code,
                    "stdout": all_logs,
                    "stderr": stderr_logs,
                    "duration_ms": duration_ms
                }
                
            except Exception as e:
                # 超时或其他错误
                logger.error(f"[Docker] 执行超时或错误: {e}")
                duration_ms = int((time.time() - start_time) * 1000)
                container.kill()
                return {
                    "success": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"执行超时或错误: {e}",
                    "duration_ms": duration_ms
                }
            
        except Exception as e:
            logger.error(f"[Docker] 容器运行失败: {e}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration_ms": 0
            }
    
    def _parse_requirements(self, req_path: str) -> set:
        """解析 requirements.txt，提取包名（忽略版本、注释和空行）"""
        packages = set()
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 处理 pkg==1.0、pkg>=1.0、pkg~=1.0 等格式
                    pkg = re.split(r'[=<>~!;\[\s]', line)[0].strip().lower()
                    if pkg:
                        packages.add(pkg)
        except Exception as e:
            logger.warning(f"[Docker] 解析 requirements.txt 失败: {e}")
        return packages

    def _check_preinstalled(self, req_path: str) -> list:
        """检查 requirements.txt 中的依赖是否都在沙箱预装列表中"""
        required = self._parse_requirements(req_path)
        missing = [pkg for pkg in required if pkg not in PREINSTALLED_PACKAGES]
        return missing

    def run_tests(self) -> Dict:
        """运行测试 - 禁网模式下依赖必须已预装，启网模式下允许 pip install"""
        req_path = os.path.join(self.temp_dir or "", "requirements.txt")

        if os.path.exists(req_path):
            if self.network_disabled:
                # 禁网模式：只检查依赖是否在预装列表中
                missing = self._check_preinstalled(req_path)
                if missing:
                    error_msg = (
                        f"以下依赖未在沙箱镜像中预装，且网络已禁用无法安装: {', '.join(missing)}"
                    )
                    logger.warning(f"[Docker] {error_msg}")
                    return {
                        "success": False,
                        "file": "docker_tests",
                        "passed": False,
                        "duration_ms": 0,
                        "output": error_msg,
                        "error": error_msg,
                        "sandbox": "docker",
                        "exit_code": -1,
                    }
                logger.info("[Docker] requirements.txt 中所有依赖均已预装")
            else:
                # 启网模式：允许 pip install（仅用于特殊调试场景）
                logger.info("[Docker] 安装项目依赖")
                req_install = self.run_command(
                    "pip install -r requirements.txt -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 || "
                    "pip install -r requirements.txt -q 2>&1"
                )
                if not req_install["success"]:
                    logger.warning(f"[Docker] 依赖安装可能失败: {req_install['stderr'][:300]}")
        
        # pytest 已预装在沙箱镜像中
        result = self.run_command("pytest -v --tb=short 2>&1")
        
        # 如果 pytest 命令不存在（返回码 127 或包含 command not found）
        if result["returncode"] == 127 or "command not found" in result.get("stdout", ""):
            result = self.run_command("python -m pytest -v --tb=short 2>&1")
        
        if result["returncode"] == 127 or "command not found" in result.get("stdout", ""):
            result = self.run_command("python3 -m pytest -v --tb=short 2>&1")
        
        # 判定结果：以 pytest 退出码为主，关键词为辅
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        combined = stdout + "\n" + stderr
        
        has_failure_keywords = (
            "FAILED" in combined
            or "ERROR" in combined
            or "failures" in combined.lower()
            or "errors" in combined.lower()
        )
        has_pass_keyword = "passed" in combined
        
        if result["returncode"] == 0:
            # pytest 返回 0 通常表示通过，但如果日志里明确有失败关键词，则降级
            result["success"] = not has_failure_keywords
        else:
            # pytest 返回非 0，但如果日志里没有失败关键词且包含 passed，可能是其他原因
            result["success"] = has_pass_keyword and not has_failure_keywords
        
        return result
    
    def run_code(self, entry_file: str = "main.py") -> Dict:
        """运行代码"""
        return self.run_command(f"python {entry_file}")
    
    def read_file(self, file_path: str) -> str:
        """读取沙箱内文件"""
        if not self.temp_dir:
            return ""
        
        full_path = os.path.join(self.temp_dir, file_path)
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except Exception:
            return ""
    
    def cleanup(self) -> None:
        """清理资源"""
        # 删除容器
        if self.container_id and self.client:
            try:
                container = self.client.containers.get(self.container_id)
                container.remove(force=True)
                logger.info(f"[Docker] 容器已删除: {self.container_id[:12]}")
            except Exception as e:
                logger.warning(f"[Docker] 容器删除失败: {e}")
        
        # 删除临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"[Docker] 工作目录已清理: {self.temp_dir}")
        
        self.temp_dir = None
        self.container_id = None
    
    def __enter__(self):
        self.create_workspace()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def check_docker_available() -> bool:
    """检查 Docker 是否可用"""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False
