# 4智能体平台 - Docker 沙箱镜像
# 预装 pytest 和常用工具，避免运行时重复安装

FROM python:3.11-slim

# 使用国内 Debian 镜像源（加速 apt）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list || true

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# 使用国内 PyPI 镜像（加速 pip）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip

# 预装测试工具与常用项目依赖（沙箱默认禁网，运行时不再 pip install）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    numpy>=1.26.0 \
    requests>=2.30.0 \
    beautifulsoup4>=4.12.0 \
    pandas>=2.0.0 \
    pydantic>=2.0.0 \
    typer>=0.9.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0

# 设置工作目录
WORKDIR /workspace

# 验证安装
RUN python3 --version && pip3 --version && pytest --version

CMD ["python3"]
