import fastapi_poe as fp
from groq import Groq
import os
import json
import sys

SYSTEM_PROMPT = """简洁回答。禁用问候语和填充短语（"当然！""好问题！""希望对你有帮助"）。不重复问题。直接给答案，不要先描述背景。
格式：默认散文。3项以上才用列表，3节以上才用标题，简短枚举用内联形式。
长度：匹配复杂度。简单问题1-3句，不填充。明确要求完整的内容（代码、列表、步骤）绝不截断。
代码只给相关片段。不确定时一句说明后直接回答。仅在歧义影响答案时提一个澄清问题。
使用网络搜索时，回复中不显示任何引用、脚注或参考标记。
用用户消息所用的语言回答。
"""

class GroqBot(fp.PoeBot):
    def __init__(self):
        super().__init__()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    async def get_response(self, request: fp.QueryRequest):
        messages = []

        # System prompt selalu pertama
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

        for msg in request.query:
            role = msg.role
            if role == "bot":
                role = "assistant"
            messages.append({"role": role, "content": msg.content})

        # Debug log
        print(f"Messages count: {len(messages)}", file=sys.stderr)
        print(f"Roles: {[m['role'] for m in messages]}", file=sys.stderr)
        print(f"Payload size: {len(json.dumps(messages))} bytes", file=sys.stderr)

        stream = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="low",
            stream=True,
            stop=None,
            tools=[{"type": "browser_search"}],
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield fp.PartialResponse(text=delta)

app = fp.make_app(GroqBot(), access_key=os.environ["POE_ACCESS_KEY"])
