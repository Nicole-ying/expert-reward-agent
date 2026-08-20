"""LLM client factory — returns the right client based on config provider."""


def create_client(cfg: dict):
    """Create an LLM client from config's `llm` section.

    Reads ``provider``, ``api_key_env``, ``base_url`` from ``cfg["llm"]``.
    For ``proxy`` provider, reads ``api_key`` directly (not env var).
    """
    llm_cfg = cfg["llm"]
    provider = llm_cfg.get("provider", "deepseek")

    if provider == "openrouter":
        from .openrouter_client import OpenRouterClient
        api_key_env = llm_cfg["api_key_env"]
        base_url = llm_cfg["base_url"]
        return OpenRouterClient(api_key_env=api_key_env, base_url=base_url)

    elif provider == "proxy":
        from .proxy_client import ProxyClient
        api_key = llm_cfg.get("api_key", "")
        base_url = llm_cfg["base_url"]
        return ProxyClient(api_key=api_key, base_url=base_url)

    else:
        from .deepseek_client import DeepSeekClient
        api_key_env = llm_cfg["api_key_env"]
        base_url = llm_cfg["base_url"]
        return DeepSeekClient(api_key_env=api_key_env, base_url=base_url)
