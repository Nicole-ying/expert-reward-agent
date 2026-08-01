import json, re
from pathlib import Path

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): items.append(json.loads(line))
    return items

def _compact_skeleton_catalog(skeleton_catalog):
    role_map = {
        "terminal_success_reward": ("task_objective", ["success_flag"], "sparse_reward_no_learning"),
        "terminal_failure_penalty": ("safety_cost", ["failure_flag or termination_reason"], "over_conservative_policy"),
        "time_penalty": ("efficiency_cost", ["time step"], "reckless_behavior"),
        "alive_bonus": ("survival_objective", ["not_done"], "hover_or_stand_still"),
        "distance_reward": ("learning_guidance", ["current_position_to_goal"], "goal_near_oscillation"),
        "progress_delta_reward": ("learning_guidance", ["current_distance", "next_distance"], "goal_near_oscillation"),
        "forward_progress_reward": ("locomotion_guidance", ["forward_velocity"], "fast_then_fail"),
        "event_reward": ("event_objective", ["event_flag"], "event_reward_farming"),
        "weighted_sum_reward": ("multi_objective", ["multiple_terms"], "weight_domination"),
        "gated_reward": ("safety_or_stage_gate", ["gate_condition"], "gate_too_strict"),
        "energy_penalty": ("action_cost", ["action or action_magnitude"], "agent_afraid_to_move"),
        "action_smoothness_penalty": ("action_regularization", ["previous_action/current_action"], "not_available_for_discrete_without_history"),
        "stability_penalty": ("safety_cost", ["velocity", "angle", "angular_velocity"], "over_conservative_policy"),
        "potential_based_shaping": ("learning_guidance", ["potential_current", "potential_next"], "potential_misspecification"),
        "intrinsic_exploration_reward": ("exploration", ["novelty/uncertainty"], "exploration_without_completion"),
        "dynamic_curriculum_reward": ("curriculum", ["training_progress"], "unstable_weights"),
        "learned_preference_reward": ("learned_reward", ["preference_model"], "reward_model_bias")
    }
    out = []
    for item in skeleton_catalog.get("skeletons", []):
        sid = item["skeleton_id"]
        role, requires, risk = role_map.get(sid, ("unknown", [], "unknown"))
        out.append({"skeleton_id": sid, "name_zh": item.get("name_zh", ""), "role": role, "requires": requires, "main_risk": risk})
    return out

def retrieve_generation_knowledge(environment_card, chunks_path, table_chunks_path, route_catalog_path, skeleton_catalog_path):
    chunks = load_jsonl(chunks_path); tables = load_jsonl(table_chunks_path)
    route_catalog = json.loads(Path(route_catalog_path).read_text(encoding="utf-8"))
    skeleton_catalog = json.loads(Path(skeleton_catalog_path).read_text(encoding="utf-8"))
    selected_route_id = environment_card.get("task_type_selection", {}).get("selected_route_id")
    valid_routes = {x["route_id"] for x in route_catalog["routes"]}
    if selected_route_id not in valid_routes: raise ValueError(f"Invalid selected_route_id: {selected_route_id}")
    route_context = [ch for ch in chunks if ch.get("source_file") == "03_任务类型到奖励骨架的决策树.md" and ch.get("route_id") == selected_route_id]
    route_recommended = []
    for route in route_catalog["routes"]:
        if route["route_id"] == selected_route_id: route_recommended = route.get("recommended_skeletons", [])
    skeleton_context = []
    for ch in chunks:
        if ch.get("source_file") == "02_奖励函数骨架库.md" and ch.get("skeleton_id") in route_recommended:
            item = dict(ch); item["retrieval_group"] = "route_recommended_detail"; skeleton_context.append(item)
    generation_table_context = []
    for t in tables:
        if t.get("source_file") == "02_奖励函数骨架库.md" and t.get("table_role") in ["quick_skeleton_selection_table", "skeleton_quick_selection"]:
            generation_table_context.append(t)
        elif t.get("source_file") == "03_任务类型到奖励骨架的决策树.md" and "反向路由" not in t.get("heading", ""):
            generation_table_context.append(t)
    return {"retrieval_stage": "generation", "retrieval_policy": "selected_route_id_to_03_plus_route_recommended_02_plus_17_compact_catalog", "allowed_sources": ["02_奖励函数骨架库.md", "03_任务类型到奖励骨架的决策树.md"], "forbidden_sources": ["00_奖励函数设计Agent工作流.md", "01_奖励函数设计总原则.md", "04_奖励函数失败模式库.md", "05_奖励黑客检查清单.md"], "selected_route_id": selected_route_id, "route_recommended_skeletons": route_recommended, "route_context": route_context, "skeleton_context": skeleton_context, "skeleton_catalog_17_compact": _compact_skeleton_catalog(skeleton_catalog), "table_context": generation_table_context}

def build_architect_generation_context(environment_card, interface_contract, retrieved):
    return {"context_type": "architect_generation_context", "selected_route_id": retrieved.get("selected_route_id"), "route_context": [{"route_id": ch.get("route_id"), "heading": ch.get("heading"), "recommended_skeletons": ch.get("recommended_skeletons", []), "raw_expert_text": ch.get("raw_text", "")} for ch in retrieved.get("route_context", [])], "route_recommended_skeletons": retrieved.get("route_recommended_skeletons", []), "skeleton_details_from_02": [{"skeleton_id": ch.get("skeleton_id"), "heading": ch.get("heading"), "raw_expert_text": ch.get("raw_text", "")} for ch in retrieved.get("skeleton_context", [])], "skeleton_catalog_17_compact": retrieved.get("skeleton_catalog_17_compact", []), "table_context": [{"source_file": t.get("source_file"), "heading": t.get("heading"), "table_role": t.get("table_role"), "raw_table": t.get("raw_table") or t.get("table_text", "")} for t in retrieved.get("table_context", [])], "selection_policy": {"do_not_mechanically_copy_route_recommendations": True, "check_interface_availability_first": True, "reward_v1_minimal_first": True, "environment_analyzer_did_not_select_v1_skeletons": True}, "interface_summary_for_selection": {"obs_indices": interface_contract.get("obs", {}).get("indices", {}), "action_values": interface_contract.get("action", {}).get("values", {}), "allowed_info_fields": interface_contract.get("info", {}).get("allowed_fields", []), "uncertain_info_fields": interface_contract.get("info", {}).get("uncertain_fields", []), "termination": interface_contract.get("termination", {})}}

def _score_text(text, terms):
    text = text.lower(); score = 0
    for term in terms:
        if term is None: continue
        t = str(term).lower().strip()
        if not t: continue
        if t in text: score += 5
        for part in re.split(r"[_\s/\-，。,:;()（）]+", t):
            if len(part) >= 3 and part in text: score += 1
    return score

def retrieve_reflection_knowledge(reflection_query, chunks_path, table_chunks_path, top_k=8, table_top_k=6):
    chunks = load_jsonl(chunks_path); tables = load_jsonl(table_chunks_path)
    terms = []
    if isinstance(reflection_query, dict):
        for key in ["observed_failure_modes", "suspected_reward_hacking", "behavior_symptoms", "reward_term_anomalies", "candidate_next_actions"]:
            value = reflection_query.get(key, [])
            terms.extend(value if isinstance(value, list) else [value])
        terms.append(reflection_query.get("free_text", ""))
    else: terms.append(str(reflection_query))
    scored_chunks = []
    for ch in chunks:
        if ch.get("source_file") not in ["04_奖励函数失败模式库.md", "05_奖励黑客检查清单.md"]: continue
        score = _score_text(" ".join([ch.get("source_file", ""), ch.get("heading", ""), ch.get("raw_text", "")]), terms)
        if score > 0:
            item = dict(ch); item["retrieval_score"] = score; item["retrieval_reason"] = "reflection_04_05_keyword_match"; scored_chunks.append(item)
    scored_tables = []
    for t in tables:
        if t.get("source_file") not in ["04_奖励函数失败模式库.md", "05_奖励黑客检查清单.md"]: continue
        score = _score_text(" ".join([t.get("source_file", ""), t.get("heading", ""), t.get("table_role", ""), t.get("raw_table", "")]), terms)
        if score > 0 or t.get("table_role") in ["quick_diagnosis_table", "failure_quick_diagnosis_table", "reward_hacking_test_table"]:
            item = dict(t); item["retrieval_score"] = score; item["retrieval_reason"] = "reflection_table_router"; scored_tables.append(item)
    scored_chunks.sort(key=lambda x: x["retrieval_score"], reverse=True); scored_tables.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return {"retrieval_stage": "reflection", "retrieval_policy": "04_05_only_failure_mode_and_hacking_check", "allowed_sources": ["04_奖励函数失败模式库.md", "05_奖励黑客检查清单.md"], "forbidden_sources": ["00_奖励函数设计Agent工作流.md", "01_奖励函数设计总原则.md", "02_奖励函数骨架库.md", "03_任务类型到奖励骨架的决策树.md"], "reflection_query": reflection_query, "matched_reflection_chunks": scored_chunks[:top_k], "matched_table_chunks": scored_tables[:table_top_k]}

def save_generation_knowledge(result, json_path, md_path):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Generation RAG Context\n\n")
        f.write(f"- retrieval_policy: {result.get('retrieval_policy')}\n- selected_route_id: {result.get('selected_route_id')}\n- allowed_sources: {', '.join(result.get('allowed_sources', []))}\n- forbidden_sources: {', '.join(result.get('forbidden_sources', []))}\n\n")
        f.write("# 03 Selected Route Context\n\n")
        for ch in result.get("route_context", []): f.write(f"## {ch.get('source_file')} / {ch.get('heading')}\n\n{ch.get('raw_text', '')}\n\n---\n\n")
        f.write("# 02 Route Recommended Skeleton Details\n\n")
        for ch in result.get("skeleton_context", []): f.write(f"## {ch.get('source_file')} / {ch.get('heading')}\n\n- skeleton_id: {ch.get('skeleton_id')}\n\n{ch.get('raw_text', '')}\n\n---\n\n")
        f.write("# 17 Skeleton Compact Catalog\n\n```json\n" + json.dumps(result.get("skeleton_catalog_17_compact", []), ensure_ascii=False, indent=2) + "\n```\n")

def save_reflection_knowledge(result, json_path, md_path):
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Reflection RAG Context: 04 + 05 only\n\n")
        f.write(f"- retrieval_policy: {result.get('retrieval_policy')}\n- allowed_sources: {', '.join(result.get('allowed_sources', []))}\n- forbidden_sources: {', '.join(result.get('forbidden_sources', []))}\n\n")
        f.write("# Matched 04/05 Reflection Chunks\n\n")
        for ch in result.get("matched_reflection_chunks", []): f.write(f"## {ch.get('source_file')} / {ch.get('heading')}\n\n- score: {ch.get('retrieval_score')}\n\n{ch.get('raw_text', '')}\n\n---\n\n")
        f.write("# Matched 04/05 Table Chunks\n\n")
        for t in result.get("matched_table_chunks", []): f.write(f"## {t.get('source_file')} / {t.get('heading')} / table {t.get('table_index')}\n\n- table_role: {t.get('table_role')}\n- score: {t.get('retrieval_score')}\n\n{t.get('raw_table', '')}\n\n---\n\n")
