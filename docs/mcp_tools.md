# MCP 工具说明

平台使用 [Model Context Protocol](https://modelcontextprotocol.io/)（MCP）为智能体提供工具能力。

## 架构

```
┌─────────────┐     stdio      ┌─────────────────┐
│   D/B/E     │  ◄──────────►  │  MCP Server     │
│  (Client)   │                │  (FastMCP)      │
└─────────────┘                └─────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              filesystem         shell/docker           web/memory
```

- **Client**：`src/mcp/client.py`，每个任务一个实例，绑定到该任务的 `projects/{project_id}` 工作空间
- **Server**：`src/mcp/server.py`，通过 `src/mcp/run_server.py` 以独立进程启动
- **桥接层**：`src/mcp/bridge.py`，提供 `ask_with_tools()` 让 LLM 自动决定何时调用工具

## 内置工具

| 工具名 | 说明 | 典型使用场景 |
|--------|------|--------------|
| `file_read` | 读取工作空间内文件 | E 检查代码，B 读取已有文件 |
| `file_write` | 在工作空间内写入文件 | B 写入生成的代码 |
| `file_list` | 列出目录内容 | D 查看现有项目结构 |
| `shell_exec` | 执行 shell 命令 | B/E 运行测试、lint、安装依赖 |
| `web_fetch` | 只读获取网页内容 | D 做技术调研 |
| `vector_search` | 检索历史任务向量记忆 | D/B/E 查相似案例 |

## 安全策略

1. **路径隔离**：所有文件操作限制在当前任务的 `projects/{project_id}` 目录内
2. **命令安全**：`shell_exec` **必须**走 Docker 沙箱；Docker 不可用时默认拒绝执行，不再自动回退到本地 shell
3. **本地执行逃生口**：仅在开发调试时可通过 `MCP_SHELL_ALLOW_LOCAL=true` 开启本地执行，生产环境必须保持关闭
4. **网络只读**：`web_fetch` 仅使用 GET，禁止修改性请求

## 开关

环境变量：

```bash
MCP_ENABLED=true                # 启用（默认）
MCP_ENABLED=false               # 禁用
MCP_SHELL_ALLOW_LOCAL=false     # 是否允许 shell_exec 在 Docker 不可用时回退本地执行（默认 false，生产环境必须 false）
```

配置文件 `config/config.yaml`：

```yaml
mcp:
  enabled: true
  tools:
    file_read: true
    file_write: true
    file_list: true
    shell_exec: true
    web_fetch: true
    vector_search: true
```

## 独立测试 Server

```bash
source venv/bin/activate
python -m src.mcp.run_server --workspace projects/demo-project
```

## 添加自定义工具

要扩展工具，请修改 `src/mcp/server.py` 中的 `create_mcp_server` 函数：

```python
@mcp.tool()
async def my_tool(param: str) -> str:
    """工具描述"""
    return f"结果: {param}"
```

重启后 Client 会自动通过 `list_tools()` 发现新工具。
