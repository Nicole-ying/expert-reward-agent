"""Generic OpenAI-compatible API client (for proxy services like proaiapi.tech)."""
import json, os, time


class ProxyClient:
    def __init__(self, api_key, base_url):
        from openai import OpenAI
        if not api_key:
            raise RuntimeError("Missing API key")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def completion(self, max_empty_retries=1, max_connection_retries=5,
                   reasoning_effort=None, **kwargs):
        from openai import APIConnectionError, APITimeoutError

        base_max_tokens = kwargs.get("max_tokens")
        finish_reason = None
        for attempt in range(max_empty_retries + 1):
            request_kwargs = dict(kwargs)
            if base_max_tokens and attempt:
                request_kwargs["max_tokens"] = base_max_tokens * (attempt + 1)

            for connection_attempt in range(max_connection_retries + 1):
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                    break
                except (APIConnectionError, APITimeoutError) as exc:
                    if connection_attempt >= max_connection_retries:
                        raise
                    delay = min(30, 2 ** (connection_attempt + 1))
                    print(f"Proxy API connection failed; retry {connection_attempt+1} in {delay}s", flush=True)
                    time.sleep(delay)
            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message
            if (message.content or "").strip() or message.tool_calls:
                return response
        raise RuntimeError(f"Proxy API returned empty response; finish_reason={finish_reason!r}")

    def chat(self, model, system_prompt, user_prompt, temperature=0.2,
             max_tokens=4096, json_mode=False, reasoning_effort=None):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = self.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return r.choices[0].message.content

    def chat_json(self, *args, **kwargs):
        return json.loads(self.chat(*args, json_mode=True, **kwargs))
