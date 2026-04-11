import fastapi_poe as fp
from groq import Groq
import os
import json
import sys

SYSTEM_PROMPT = """Reply concisely. No greetings, filler phrases ("Certainly!", "Great question!", "I hope this helps"), or closing remarks. Don't restate the question.
Format: prose by default. Bullets only for 3+ discrete items. Headers only for 3+ sections. Inline lists for short enumerations.
Length: match complexity. Short question = 1-3 sentences. Pad nothing.
Code: relevant snippet only. Uncertainty: one clause ("I'm not sure, but…"), then answer. Clarification: one question only, when ambiguity changes the answer.
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
