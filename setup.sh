#!/bin/bash
# 高考志愿填报 Agent - 阿里云部署脚本
# 在服务器上执行: bash setup.sh

set -e

echo "=== 安装 Python 依赖 ==="
pip install langgraph langchain langchain-openai fastapi uvicorn httpx pydantic pydantic-settings python-dotenv sse-starlette

echo ""
echo "=== 请确认 .env 文件已配置 DEEPSEEK_API_KEY ==="
if grep -q "sk-your-deepseek-api-key-here" .env 2>/dev/null; then
    echo "⚠️  警告: 请先编辑 .env 文件，填入真实的 DeepSeek API Key！"
    echo "   vi .env"
    exit 1
fi

echo ""
echo "=== 启动服务 (端口 8003) ==="
echo "启动命令: uvicorn src.main:app --host 0.0.0.0 --port 8003 --reload"
echo ""
echo "测试: curl -X POST http://localhost:8003/chat -H 'Content-Type: application/json' -d '{\"session_id\":\"t1\",\"message\":\"你好\"}'"
