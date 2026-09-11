import os
import json
import argparse
from typing import Dict, Any, List, Optional

# 导入步骤2的完整提示词模板
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

# 直接导入，不使用相对导入
import prompt_step_2
SOLUTION_PROMPT_RESUME_WITH_VERIFIER_V1 = prompt_step_2.SOLUTION_PROMPT_RESUME_WITH_VERIFIER_V1


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def pick_latest_feedback_by_puzzle(feedback_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    同一 puzzle_id 可能在一个 epoch 写多次，这里取 timestamp 最大的一条作为“最新反馈”。
    """
    best: Dict[str, Dict[str, Any]] = {}
    for it in feedback_items:
        pid = it.get("puzzle_id")
        if not pid:
            continue
        ts = float(it.get("timestamp", 0.0))
        if pid not in best or ts > float(best[pid].get("timestamp", 0.0)):
            best[pid] = it
    return best


def format_steps(steps: List[Dict[str, Any]], max_k: int) -> List[str]:
    """
    steps 是 step_verification 里收集的 dict 列表。
    输出字符串列表（供 prompt 使用）
    """
    out = []
    for s in steps[:max_k]:
        sid = s.get("sid") or ""
        dsl = s.get("dsl") or ""
        note = s.get("note") or ""
        if sid and dsl:
            out.append(f"{sid}: {dsl}. ({note})")
        elif dsl:
            out.append(f"{dsl}. ({note})")
        else:
            raw = (s.get("raw") or "").strip()
            if raw:
                out.append(raw)
    return out


def build_feedback_block(
    good_steps: List[Dict[str, Any]],
    bad_steps: List[Dict[str, Any]],
    unknown_steps: List[Dict[str, Any]],
    max_good: int = 15,
    max_bad: int = 15,
    max_unknown: int = 0,
) -> str:
    """
    构造你说的“GOOD/BAD reasoning steps”反馈块。
    """
    good_lines = format_steps(good_steps, max_good)
    bad_lines = format_steps(bad_steps, max_bad)
    unk_lines = format_steps(unknown_steps, max_unknown) if max_unknown > 0 else []

    block = []
    block.append("================================")
    block.append("Z3 FEEDBACK (from previous attempt)")
    block.append("================================")
    block.append("")
    block.append("Verified TRUE steps (you MAY use them as facts):")
    if good_lines:
        for x in good_lines:
            block.append(f"  - {x}")
    else:
        block.append("  - (none)")
    block.append("")
    block.append("Verified FALSE steps (you MUST NOT use them; they contradict the clues/domain):")
    if bad_lines:
        for x in bad_lines:
            block.append(f"  - {x}")
    else:
        block.append("  - (none)")
    block.append("")
    if unk_lines:
        block.append("Not-entailed steps (UNKNOWN: do NOT treat them as guaranteed facts):")
        for x in unk_lines:
            block.append(f"  - {x}")
        block.append("")
    block.append("Rules:")
    block.append("1) You must produce output strictly in the required 5-key JSON format inside <answer>...</answer>.")
    block.append("2) When writing parsed_reasoning, every step must be logically entailed by (parsed_clues + earlier VERIFIED TRUE steps).")
    block.append("3) Do NOT repeat any VERIFIED FALSE step; avoid making unstated assumptions.")
    block.append("")
    return "\n".join(block)


def build_step2_prompt(
    puzzle_text: str,
    step1_json: Dict[str, Any],
    verifier_feedback: Dict[str, Any]
) -> str:
    """
    使用 SOLUTION_PROMPT_RESUME_WITH_VERIFIER_V1 模板构建步骤2提示
    """
    step1_json_str = json.dumps(step1_json, ensure_ascii=False, indent=2)
    verifier_feedback_str = json.dumps(verifier_feedback, ensure_ascii=False, indent=2)
    
    prompt = SOLUTION_PROMPT_RESUME_WITH_VERIFIER_V1.format(
        PUZZLE_TEXT=puzzle_text or "",
        STEP1_JSON=step1_json_str,
        VERIFIER_FEEDBACK_JSON=verifier_feedback_str
    )
    
    return prompt


def build_parsed_only_user_content(
    n_houses: int,
    attribute_values: Dict[str, Any],
    parsed_clues: List[str],
    feedback_block: str,
) -> str:
    """
    不依赖原 puzzle 文本，直接给 domain + parsed_clues + feedback。
    适合你说的 Step-2：只 pass parsed input。
    """
    # 注意：这里不重复超长 system prompt，尽量短且强约束
    domain_json = json.dumps(
        {"n_houses": n_houses, "attribute_values": attribute_values, "parsed_clues": parsed_clues},
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are an expert logic puzzle solver.\n"
        "You are given the parsed domain and parsed clues in a DSL. You must generate:\n"
        "  - parsed_reasoning (list of step DSL lines)\n"
        "  - final solution table\n\n"
        "You MUST output ONLY one <answer>...</answer> JSON object with EXACTLY these 5 top-level keys:\n"
        "  n_houses, attribute_values, parsed_clues, parsed_reasoning, solution\n\n"
        "IMPORTANT:\n"
        "- Keep n_houses / attribute_values / parsed_clues identical to the given input.\n"
        "- In parsed_reasoning, each step must be a single sentence and end with a period.\n"
        "- Reasoning step DSL: S<k> [C<i>(+C<j>...)] set(H,Attr,Val). OR not(H,Attr,Val).\n"
        "- Every step must be logically entailed by parsed_clues + earlier VERIFIED TRUE steps.\n\n"
        "--------------------\n"
        "PARSED INPUT\n"
        "--------------------\n"
        f"{domain_json}\n\n"
        f"{feedback_block}\n\n"
        "Now generate the final <answer>...</answer> output.\n"
    )


def append_feedback_to_original_prompt(prompt_msgs: List[Dict[str, Any]], feedback_block: str) -> List[Dict[str, Any]]:
    """
    将 feedback 追加到最后一条 user content（最稳：不改系统指令）。
    """
    if not isinstance(prompt_msgs, list) or not prompt_msgs:
        return prompt_msgs

    # 找最后一个 role=user
    idx = None
    for i in range(len(prompt_msgs) - 1, -1, -1):
        if prompt_msgs[i].get("role") == "user":
            idx = i
            break
    if idx is None:
        # 没有 user，就直接 append
        prompt_msgs = list(prompt_msgs) + [{"role": "user", "content": feedback_block}]
        return prompt_msgs

    new_msgs = [dict(x) for x in prompt_msgs]
    old_content = new_msgs[idx].get("content", "")
    new_msgs[idx]["content"] = (old_content.rstrip() + "\n\n" + feedback_block).strip()
    return new_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback_jsonl", type=str, required=True, help="Step-1 输出的 reasoning_feedback.jsonl")
    ap.add_argument("--output_jsonl", type=str, required=True, help="输出 epoch2 数据集 jsonl")
    ap.add_argument("--epoch1_jsonl", type=str, default=None, help="(可选) epoch1 原始数据集 jsonl，用于 append 模式")
    ap.add_argument("--mode", type=str, choices=["append", "parsed_only"], default="parsed_only")
    ap.add_argument("--max_good", type=int, default=15)
    ap.add_argument("--max_bad", type=int, default=15)
    ap.add_argument("--max_unknown", type=int, default=0)
    args = ap.parse_args()

    feedback_items = read_jsonl(args.feedback_jsonl)
    latest = pick_latest_feedback_by_puzzle(feedback_items)

    epoch1_map: Dict[str, Dict[str, Any]] = {}
    if args.mode == "append":
        if not args.epoch1_jsonl:
            raise ValueError("mode=append 需要提供 --epoch1_jsonl（原始数据集，包含 prompt 字段）")
        epoch1_items = read_jsonl(args.epoch1_jsonl)
        for it in epoch1_items:
            pid = it.get("id") or it.get("puzzle_id") or (it.get("extra_info", {}) if isinstance(it.get("extra_info"), dict) else {}).get("id")
            if pid:
                epoch1_map[pid] = it

    out_items = []
    for pid, fb in latest.items():
        sv = fb.get("step_verification", {}) or {}
        good_steps = sv.get("good_steps", []) or []
        bad_steps = sv.get("bad_steps", []) or []
        unknown_steps = sv.get("unknown_steps", []) or []
        
        # 获取用于 Step-2 提示生成的所有必要信息
        puzzle_text = fb.get("puzzle_text", "")
        step1_json = fb.get("step1_json", {})
        verifier_feedback = fb.get("verifier_feedback", {})
        
        # 旧的 feedback_block 构建，用于兼容
        feedback_block = build_feedback_block(
            good_steps=good_steps,
            bad_steps=bad_steps,
            unknown_steps=unknown_steps,
            max_good=args.max_good,
            max_bad=args.max_bad,
            max_unknown=args.max_unknown,
        )

        if args.mode == "append":
            base = epoch1_map.get(pid)
            if not base:
                # 找不到就跳过（或你也可以选择 parsed_only fallback）
                continue
            prompt_msgs = base.get("prompt")
            if not isinstance(prompt_msgs, list):
                # 如果没有 prompt，就跳过
                continue

            new_prompt = append_feedback_to_original_prompt(prompt_msgs, feedback_block)

            new_item = dict(base)
            new_item["prompt"] = new_prompt
            # 标记 epoch2
            extra = new_item.get("extra_info", {}) if isinstance(new_item.get("extra_info"), dict) else {}
            extra["epoch"] = fb.get("epoch", 1) + 1  # 下一个 epoch
            extra["feedback_source"] = args.feedback_jsonl
            new_item["extra_info"] = extra
            out_items.append(new_item)

        else:
            # parsed_only：使用新的步骤2提示构建
            n_houses = fb.get("n_houses")
            attribute_values = fb.get("attribute_values")
            parsed_clues = fb.get("parsed_clues")

            if not (n_houses and isinstance(attribute_values, dict) and isinstance(parsed_clues, list)):
                continue
            
            # 使用新的步骤2提示模板构建用户内容
            if puzzle_text and step1_json and verifier_feedback:
                # 如果有完整的步骤2信息，使用新的提示模板
                user_content = build_step2_prompt(
                    puzzle_text=puzzle_text,
                    step1_json=step1_json,
                    verifier_feedback=verifier_feedback
                )
            else:
                # 否则使用旧的提示构建方式
                user_content = build_parsed_only_user_content(
                    n_houses=int(n_houses),
                    attribute_values=attribute_values,
                    parsed_clues=parsed_clues,
                    feedback_block=feedback_block,
                )

            # 输出结构：尽量与原框架相似（你可按需要补齐字段）
            new_item = {
                "id": pid,
                "ability": "logical_reasoning",
                "apply_chat_template": False,
                "prompt": [{"role": "user", "content": user_content}],
                "extra_info": {
                    "id": pid,
                    "epoch": fb.get("epoch", 1) + 1,  # 下一个 epoch
                    "feedback_source": args.feedback_jsonl,
                    "puzzle_text": puzzle_text,
                    "step1_json": step1_json,
                    "verifier_feedback": verifier_feedback
                }
            }
            out_items.append(new_item)

    write_jsonl(args.output_jsonl, out_items)
    print(f"[OK] Wrote epoch2 dataset: {args.output_jsonl} (count={len(out_items)})")


if __name__ == "__main__":
    main()