import fastapi_poe as fp
from groq import Groq
import os

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

class GroqBot(fp.PoeBot):
    async def get_response(self, request: fp.QueryRequest):
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.query
        ]

        stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            reasoning_effort="low",
            tools=[{"type": "browser_search"}]
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield fp.PartialResponse(text=delta)

app = fp.make_app(GroqBot(), access_key=os.environ["POE_ACCESS_KEY"])
