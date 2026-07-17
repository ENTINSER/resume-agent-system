#!/usr/bin/env python3
"""快速测试 - 验证P1/P2优化效果"""

import sys
sys.path.insert(0, '/Users/mingrun/resume-agent-system')

from src.core.llm import get_llm
from src.core.D import orchestrator
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("测试1: API连接 + 成本追踪")
print("=" * 60)

llm = get_llm()
result = llm.chat('You are helpful.', 'Say OK', max_tokens=10)
print(f"✅ API连接: {'成功' if result['success'] else '失败'}")
print(f"   Tokens: in={result['input_tokens']}, out={result['output_tokens']}")
print(f"   成本: ${result['cost_usd']:.6f}")
print(f"   耗时: {result['elapsed_ms']}ms")

print()
print("=" * 60)
print("测试2: 完整流程（简单需求）")
print("=" * 60)

# 运行一个简单测试
requirements = "创建一个Python函数，接收两个数字参数，返回它们的和。包含基本测试。"

final_state = orchestrator.run(requirements)

print()
print("=" * 60)
print("测试3: SQLite持久化验证")
print("=" * 60)

from src.core.persistence import get_session_store
store = get_session_store()

# 查询会话
session = store.get_session(final_state['session_id'])
if session:
    print(f"✅ 会话已保存: {session['session_id'][:8]}")
    print(f"   状态: {session['status']}")
    print(f"   评分: {session['overall_score']}")
    print(f"   通过: {'是' if session['passed'] else '否'}")
    print(f"   迭代: {session['iteration_count']}")
    print(f"   Tokens: {session['total_tokens']:,}")
    print(f"   成本: ${session['total_cost']:.4f}")
else:
    print("❌ 会话未保存")

# 性能统计
stats = store.get_performance_stats(final_state['session_id'])
if stats.get('llm_by_phase'):
    print(f"\n📊 LLM各阶段统计:")
    for phase in stats['llm_by_phase']:
        print(f"   {phase['phase']}: {phase['tokens']:,} tokens, ${phase['cost']:.4f}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
