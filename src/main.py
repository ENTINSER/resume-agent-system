#!/usr/bin/env python3
"""主入口 - 4智能体协作平台

使用方式：
    python src/main.py "开发一个LangGraph多智能体系统"
    python src/main.py "开发一个LangGraph多智能体系统" --ui  # 启动可视化界面

架构：
    D(编排器) → B(开发) → E(评估) → C(训练) → Human
"""

import os
import sys
import argparse
import threading

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.D import orchestrator
from src.core.B import save_project
from src.core.E import generate_html_report
from src.core.logger import logger


def handle_result(result: dict, output: str) -> None:
    """统一处理编排器结果：保存项目、生成报告、打印总结"""
    code_artifacts = result.get("code_artifacts", [])
    if code_artifacts:
        project_dir = save_project(result, output)
        print(f"\n📁 项目已保存: {project_dir}")

    evaluation_result = result.get("evaluation_result")
    if evaluation_result:
        report = generate_html_report(result)
        report_path = f"{output}/report.html"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📊 评估报告: {report_path}")

    print("\n" + "="*70)
    print("📋 执行总结")
    print("="*70)
    print(f"会话ID: {result.get('session_id', '')}")
    print(f"最终状态: {result.get('status', '')}")
    print(f"迭代次数: {result.get('iteration_count', 0)}")
    print(f"代码文件: {len(code_artifacts)}")
    print(f"评估次数: {len(result.get('evaluation_history', []))}")
    print(f"训练任务: {len(result.get('training_jobs', []))}")

    if result.get("error"):
        print(f"❌ 错误: {result['error']}")
    else:
        print("✅ 流程完成")

    print("="*70)


def run_orchestrator(requirements: str, output: str, ui_mode: bool = False):
    """运行编排器（在后台线程中）"""
    from src.core.D import Orchestrator

    orch = Orchestrator(ui_mode=ui_mode)
    result = orch.run(requirements)
    handle_result(result, output)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='4智能体协作平台')
    parser.add_argument('requirements', nargs='?', default='', help='项目需求描述')
    parser.add_argument('--output', '-o', default='projects', help='项目输出目录')
    parser.add_argument('--no-train', action='store_true', help='跳过训练阶段')
    parser.add_argument('--ui', action='store_true', help='启动可视化Web界面')
    parser.add_argument('--port', type=int, default=8080, help='UI端口 (默认8080)')
    parser.add_argument('--resume', type=str, default=None, help='从指定 session_id 断点续跑')
    parser.add_argument('--auto-approve', action='store_true', help='HITL 检查点自动批准（非交互式/自动化测试使用）')

    args = parser.parse_args()

    if args.auto_approve:
        from src.core.config import settings
        settings.auto_approve = True
        os.environ["AUTO_APPROVE"] = "true"

    if args.resume:
        # 断点续跑模式
        print("="*70)
        print("🔄 4智能体协作平台 - 断点续跑")
        print("="*70)
        print(f"会话ID: {args.resume}")
        print("="*70)
        from src.core.D import Orchestrator
        orch = Orchestrator()
        result = orch.resume_session(args.resume)
        handle_result(result, args.output)
        return
    
    if args.ui:
        # UI 模式：启动 FastAPI 服务器
        print("="*70)
        print("🌐 4智能体协作平台 - 可视化模式")
        print("="*70)
        print(f"控制面板: http://127.0.0.1:{args.port}")
        print("="*70)
        
        # 如果提供了需求，在后台启动编排器
        if args.requirements:
            print(f"🚀 后台启动编排器: {args.requirements[:50]}...")
            thread = threading.Thread(
                target=run_orchestrator,
                args=(args.requirements, args.output, True)
            )
            thread.daemon = True
            thread.start()
        
        # 启动 UI 服务器
        from src.ui.web_ui import start_ui_server
        start_ui_server(host="127.0.0.1", port=args.port)
    else:
        # CLI 模式
        if not args.requirements:
            print("❌ 请提供项目需求描述，或使用 --ui 启动可视化界面")
            parser.print_help()
            sys.exit(1)
        
        print("="*70)
        print("🤖 4智能体协作平台")
        print("="*70)
        print(f"需求: {args.requirements}")
        print(f"输出: {args.output}/")
        print("="*70)
        
        run_orchestrator(args.requirements, args.output, ui_mode=False)


if __name__ == "__main__":
    main()
