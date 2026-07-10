#!/usr/bin/env python3
"""
DERES Framework Figure — OpenRouter GPT Image / Seedream / Gemini 生图

使用方法:
  export OPENROUTER_API_KEY='sk-or-...'
  python analysis/paper/generate_deres_figure_openrouter.py

支持的模型 (--model):
  openai/gpt-image-2          ← 指令遵循最强，文字最准
  bytedance/seedream-4.5      ← 国内模型，$0.04/张，18种比例
  google/gemini-3-pro-image   ← 擅长 infographics/diagrams，贵但好
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/images/generations"
OUT = Path(__file__).resolve().parent / "figures" / "framework"
OUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 精心设计的高质量 prompt — 参考了用户给的范例图风格
# ═══════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "main": """Create a professional, publication-quality scientific framework diagram for a top-tier machine learning conference paper (NeurIPS/ICML level). This is Figure 1: the method overview of DERES (Diagnosis-guided Expert Reward Evolution Search).

LAYOUT: A horizontal five-stage pipeline from left to right across the top half, with supporting infrastructure modules (Memory, Best Archive, Failure Recovery) arranged below.

STAGE 1 — ENVIRONMENT ABSTRACTION (far left):
A rounded-rectangle card labeled "Environment Card". Below it, a smaller connected box labeled "Expert Schema Prompting" with sub-items: "Task Profile", "Reward Roles", "Role-to-Signal Mapping", "Formula Operators". Color: dark gray (#4F5B66) with subtle gradient.

STAGE 2 — REWARD GENERATION:
A rounded-rectangle box with an LLM/neural network icon. Label: "Reward Generator (LLM Agent)". Subtitle: "role → signal → formula operator → reward code". Color: blue (#276FBF). Show a code snippet emerging from the right side.

STAGE 3 — PPO TRAINING & EVALUATION:
A rounded-rectangle box with a gear/training icon. Label: "PPO Training + Evaluator". Subtitle: "train with generated reward | evaluate with masked official reward". Color: warm amber/orange (#D4922E). Show a policy network icon inside.

STAGE 4 — STRUCTURED DIAGNOSIS (highlighted as core innovation):
A larger rounded-rectangle box with a magnifying glass / diagnostic icon. Label: "Diagnostic Reflection Agent". Sub-items inside the box (small text cards): "Component Statistics (mean, abs_mean, active_rate)", "External Score Analysis", "Failure Behavior Identification", "Intervention Level Selection: L1 Scale Repair | L2 Math Transform | L3 Structure Rebuild". This box should have a subtle red/rose highlight or glow to mark it as the core innovation. Color: rose/red (#B84A4A).

STAGE 5 — EVIDENCE-GUIDED REPAIR:
A rounded-rectangle box with a wrench/repair icon. Label: "Local Reward Repair". Subtitle: "modify ONE target component per iteration (attributable)". Color: violet/purple (#7B5EA7).

ARROWS AND FLOW:
- Thick horizontal arrows (dark gray) connecting stages 1→2→3→4→5.
- A prominent curved RED arrow looping back from Stage 5 (Repair) to Stage 3 (PPO Training). Label on the arrow: "Iterative Self-Evolution Loop: failed reward → diagnosis → repair → retrain".
- A dashed-line rounded-rectangle highlight region spanning Stage 4 and Stage 5, labeled "Core Innovation".

BOTTOM ROW (three modules, smaller, below the main pipeline):
1. "Best Reward Archive" (teal, #3A8A8A): icon of a trophy/star. Subtitle: "best reward.py + vecnormalize → warm-start".
2. "Reward Memory" (teal): icon of a database/history. Subtitle: "diagnosis history · repair records · scores".
3. "Failure Recovery" (gray): icon of a shield/refresh. Subtitle: "invalid code repair | duplicate detection | fresh restart".

Dotted vertical lines connecting PPO Training → Best Archive, Diagnosis → Memory, Repair → Memory. Curved dashed arrows from Memory → Diagnosis (read path) and Best Archive → Generator (warm-start).

STYLE REQUIREMENTS:
- Clean flat-vector illustration style with subtle gradients and soft shadows on boxes
- White or very light gray background
- Professional sans-serif font (like Arial/Helvetica), clear readable labels
- Boxes have rounded corners (8-12px radius) with subtle drop shadows
- Color scheme: restrained, professional — muted blues, warm amber, rose, violet, teal
- No 3D extrusions, no photorealistic elements, no decorative illustrations
- Icons should be minimal line-art style, not photo-realistic
- The overall look should feel like a polished conference paper figure, not a marketing graphic
- Aspect ratio: approximately 16:9, suitable for a full-width single-column figure

CRITICAL CONSTRAINTS:
- DO NOT include any Chinese characters. ALL text must be in English.
- DO NOT include any logos, watermarks, or institutional marks.
- DO NOT fabricate quantitative data, p-values, or experimental numbers.
- Keep all text labels short and legible — avoid long sentences inside boxes.
- The red loop arrow must be visually prominent — this is the key message of the figure.""",

    "variant_compact": """Create a professional scientific framework diagram in ACM/IEEE/AAAI double-column line-art schematic style for a machine learning paper.

Show a five-stage pipeline: Environment Card → Reward Generator (LLM) → PPO Training → Diagnostic Reflection Agent → Local Repair, with a prominent red feedback loop arrow returning from Repair to Training.

Style: White background, thin clean lines, limited muted colors (grays, one blue accent, one red accent for the loop), NO gradients, NO shadows, NO 3D effects, NO icons. Pure flat line-art schematic like a professionally hand-drawn figure. Arial/Helvetica labels, minimal text. All English.""",

    "variant_detail": """Create a detailed scientific mechanism diagram zooming into the "Diagnostic Reflection Agent" — the core module of the DERES framework.

Show a large central panel with the Diagnostic Reflection Agent processing structured training feedback (component-wise reward statistics, bar charts, external scores on the left) through four sequential steps:
1. "Identify Failure Behavior" (early crash, proxy hacking, sparse activation)
2. "Select Intervention Target" (which reward component?)
3. "Choose Level: L1 Scale | L2 Structure | L3 Rebuild"
4. "Generate Local Repair" (modify ONE component only)

On the right side, show output "Evidence Cards" that pair each reward component with a diagnosis and a repair action tag ([Fix], [Tune], [Keep], [Add]).

Professional flat-vector illustration style. Clean white background. Restrained blue/amber/rose/violet palette. Rounded boxes with subtle shadows. All English text. No fabricated data values. Aspect ratio 4:3."""
}


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("\n❌ OPENROUTER_API_KEY 未设置！\n")
        print("  1. 去 https://openrouter.ai/keys 注册并创建 API key")
        print("  2. 充值 $1 就够了（最低充值额）")
        print("  3. 运行:  export OPENROUTER_API_KEY='sk-or-...'")
        print(f"  4. 重新运行:  python {__file__}\n")
        raise SystemExit(1)
    return key


def call_api(prompt: str, api_key: str, model: str = "openai/gpt-image-2",
             aspect_ratio: str = "16:9", size: str = "1024x576",
             n: int = 1) -> dict:
    """Call OpenRouter Images API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"  Calling {model} ({size})...")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {e.code}: {body}") from e


def extract_and_save(response: dict, out_dir: Path, basename: str) -> list[Path]:
    """Extract image data from API response and save to disk."""
    saved = []
    data_items = response.get("data", [])
    for i, item in enumerate(data_items):
        name = f"{basename}_{i+1}" if len(data_items) > 1 else basename

        if "url" in item:
            url = item["url"]
            print(f"  Downloading from URL: {url[:80]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "ClaudeCode/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                img_data = resp.read()
            ext = ".png"
            path = out_dir / f"{name}{ext}"
            path.write_bytes(img_data)
            print(f"  ✓ {path} ({len(img_data)} bytes)")
            saved.append(path)

        elif "b64_json" in item:
            path = out_dir / f"{name}.png"
            path.write_bytes(base64.b64decode(item["b64_json"]))
            print(f"  ✓ {path}")
            saved.append(path)

        else:
            print(f"  ⚠ No URL or b64_json in response item {i}")
            # Save the raw response for debugging
            debug_path = out_dir / f"{name}_response.json"
            debug_path.write_text(json.dumps(response, indent=2, ensure_ascii=False))
            print(f"  Debug response saved to {debug_path}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="DERES Framework Figure via OpenRouter")
    parser.add_argument("--model", default="google/gemini-3-pro-image",
                        help="Image generation model (see openrouter.ai/collections/image-models)")
    parser.add_argument("--prompt", default="main",
                        choices=["main", "variant_compact", "variant_detail", "all"],
                        help="Which prompt to use")
    parser.add_argument("--aspect-ratio", default="16:9",
                        help="Aspect ratio (16:9, 4:3, 1:1)")
    parser.add_argument("--size", default="1024x576",
                        help="Output size e.g. 1024x576, 1792x1024")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only write prompts to files, don't call API")
    args = parser.parse_args()

    api_key = None if args.dry_run else load_api_key()

    prompts_to_run = ["main", "variant_compact", "variant_detail"] if args.prompt == "all" else [args.prompt]

    for prompt_key in prompts_to_run:
        prompt_text = PROMPTS[prompt_key]
        basename = f"deres_framework_{prompt_key}_{args.model.split('/')[-1]}"

        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_key} ({len(prompt_text)} chars)")
        print(f"Model: {args.model} | {args.aspect_ratio} | {args.size}")
        print(f"{'='*60}")

        # Save prompt
        prompt_path = OUT / f"{basename}_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        print(f"  Prompt saved: {prompt_path}")

        if args.dry_run:
            print(f"  [DRY RUN] Would generate: {basename}.png")
            continue

        try:
            response = call_api(prompt_text, api_key, model=args.model,
                               aspect_ratio=args.aspect_ratio, size=args.size)
        except RuntimeError as e:
            print(f"  ❌ {e}")
            continue

        # Save full response
        resp_path = OUT / f"{basename}_response.json"
        resp_path.write_text(json.dumps(response, indent=2, ensure_ascii=False))

        saved = extract_and_save(response, OUT, basename)

        if saved:
            print(f"\n  ✅ Generated {len(saved)} image(s) for '{prompt_key}'")
        else:
            print(f"\n  ❌ No images generated for '{prompt_key}'")

        # Rate limit: 15s between API calls
        if prompt_key != prompts_to_run[-1]:
            print("  Waiting 15s before next call...")
            time.sleep(15)

    print(f"\n{'='*60}")
    print(f"All done. Output directory: {OUT}")
    print(f"{'='*60}")

    # Print summary
    pngs = sorted(OUT.glob("*.png"))
    if pngs:
        print(f"\nGenerated images ({len(pngs)}):")
        for p in pngs:
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name} ({size_kb:.0f} KB)")
    else:
        print("\nNo images generated yet.")
        if args.dry_run:
            print("Run without --dry-run to generate images.")


if __name__ == "__main__":
    main()
