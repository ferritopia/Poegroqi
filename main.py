import fastapi_poe as fp
from groq import Groq
import os
import json
import sys

SYSTEM_PROMPT = """简洁回答：禁用冗余内容，直接输出答案。不要重复提问，如果不知道答案，一定要先询问上下文。
格式优先：列表（≥3项）和标题（≥3节）按需使用。
仅当上下文未明确定义多义词或问题存在逻辑矛盾时，提出单一澄清问题。
上下文优先：根据当前对话定义术语
语言适配：始终使用用户语言，禁用中文默认输出。
效果：减少冗余规则，保留核心规则（澄清逻辑、术语绑定、语言适配），节省 tokens 同时保持明确性。
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
            model="qwen/qwen3-32b",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            reasoning_effort="none",
            # for gpt-oss:
            # reasoning_effort="low",
            reasoning_format="hidden",
            stream=True,
            stop=None,
            # tools=[{"type": "browser_search"}],
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield fp.PartialResponse(text=delta)

app = fp.make_app(GroqBot(), access_key=os.environ["POE_ACCESS_KEY"])
