import json, os
class DeepSeekClient:
    def __init__(self, api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com"):
        from openai import OpenAI
        # Fix SSL cert path for conda environments that have broken or missing cacert.pem
        ssl_cert = os.environ.get("SSL_CERT_FILE", "")
        if ssl_cert and not os.path.isfile(ssl_cert):
            try:
                import certifi
                os.environ["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                os.environ.pop("SSL_CERT_FILE", None)
        api_key=os.environ.get(api_key_env)
        if not api_key: raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        self.client=OpenAI(api_key=api_key, base_url=base_url)
    def chat(self, model, system_prompt, user_prompt, temperature=0.2, max_tokens=4096, json_mode=False):
        kwargs={}
        if json_mode: kwargs["response_format"]={"type":"json_object"}
        r=self.client.chat.completions.create(model=model,messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],temperature=temperature,max_tokens=max_tokens,**kwargs)
        return r.choices[0].message.content
    def chat_json(self,*args,**kwargs): return json.loads(self.chat(*args,json_mode=True,**kwargs))
