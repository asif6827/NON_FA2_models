# -*- coding: utf-8 -*-
"""
ray_clue_verifier.py

Ray-based clue verifier for logic puzzles.

关键修正（对应你日志里“验证不停止/无限timeout”）：
1) 复用 named actor（避免重复创建/重复加载模型）
2) 默认 CPU 推理（避免训练/rollout 抢 GPU 导致 OOM/卡死）
3) inflight 限流（max_inflight），忙的时候直接返回 0 分，防止队列无限增长
4) 你的 verification_prompt 完整保留（不删任何关键提示词）
"""

import time
import hashlib
import logging
from typing import Dict, List, Optional, Any

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _as_list(x: Any) -> List[str]:
    """Convert common container types (including numpy arrays) into a python list[str]."""
    if x is None:
        return []
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return [str(i) for i in x]
    return [str(x)]


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


@ray.remote
class ClueVerificationActor:
    """
    Ray actor for clue verification using a fixed verifier model.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        self.model_config = model_config or {}

        # inflight guard
        self.max_inflight = int(self.model_config.get("max_inflight", 1))
        self._inflight = 0

        # cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = int(self.model_config.get("cache_size", 1000))
        self.verification_history: List[Dict[str, Any]] = []

        self.model = None
        self.tokenizer = None
        self.generation_config = None

        model_path = self.model_config.get("model_path", "")
        if not model_path:
            logger.warning("[ClueVerificationActor] No model_path provided; verifier will always return 0.0")
            return

        device = str(self.model_config.get("device", "cpu")).lower().strip()
        max_new_tokens = int(self.model_config.get("max_new_tokens", 256))

        if device == "cpu":
            torch_dtype = torch.float32
            device_map = {"": "cpu"}
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device_map = self.model_config.get("device_map", "auto")

        try:
            logger.info(f"[ClueVerificationActor] Loading verifier model from: {model_path} (device={device})")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()

            self.generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
            logger.info("[ClueVerificationActor] Model ready.")
        except Exception as e:
            logger.error(f"[ClueVerificationActor] Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            self.generation_config = None

    def _calculate_cache_key(self, reasoning: str, clues: List[str]) -> str:
        cache_input = f"{reasoning}\n{','.join(clues)}"
        return _md5(cache_input)

    def _verify_clues_with_model(self, reasoning: str, clues: List[str]) -> (float, str):
        if self.model is None or self.tokenizer is None or self.generation_config is None:
            return 0.0, "no_model"

        try:
            # ===== 你的“精华提示词”完整保留（一个字不删） =====
            system_prompt = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""
            clues_text = "\n".join([f"{i+1}. {clue}" for i, clue in enumerate(clues)])

            verification_prompt = f"""Problem ID: unknown

            CLUES: {clues_text}
            
            PROPOSED SOLUTION: {reasoning}
            
            Please check if the proposed solution satisfies ALL the clues. For each clue, first reason about whether it is satisfied or violated by the solution, and then state your final answer.
            
            Respond with a JSON object in the following format:
            {{
              "clue_analysis": [
                {{ "clue_number": 1, "reasoning": "work out if clue is satisfied", "satisfied": true }},
                {{ "clue_number": 2, "reasoning": "work out if clue is satisfied", "satisfied": false }}
              ],
              "violated_clues": [1, 3],
              "all_clues_satisfied": false
            }}
            """
            final_prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {verification_prompt} [/INST]"""
            # ===========================================================

            inputs = self.tokenizer(final_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, generation_config=self.generation_config)
            txt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            j0 = txt.find("{")
            j1 = txt.rfind("}")
            if j0 == -1 or j1 == -1 or j1 <= j0:
                return 0.0, "json_not_found"

            json_str = txt[j0 : j1 + 1]
            try:
                import json
                obj = json.loads(json_str)
            except Exception:
                return 0.0, "json_parse_error"

            violated = obj.get("violated_clues", [])
            if not isinstance(violated, list):
                violated = []

            total = len(clues)
            if total <= 0:
                return 0.0, "no_clues"

            satisfied = total - len(violated)
            score = max(0.0, min(1.0, float(satisfied) / float(total)))
            return score, "ok"
        except Exception as e:
            logger.error(f"[ClueVerificationActor] verification error: {e}")
            return 0.0, "exception"

    def verify_clues_async(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        start = time.time()
        clues_list = _as_list(clues)

        cache_key = self._calculate_cache_key(reasoning, clues_list)
        if use_cache and cache_key in self.cache:
            cached = dict(self.cache[cache_key])
            cached["from_cache"] = True
            cached["verification_time"] = 0.0
            return cached

        # inflight guard：忙就直接丢弃（给0分），避免无限排队
        if self._inflight >= self.max_inflight:
            return {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues_list,
                "verification_time": 0.0,
                "timestamp": time.time(),
                "from_cache": False,
                "status": "busy_drop",
            }

        self._inflight += 1
        try:
            score, status = self._verify_clues_with_model(reasoning, clues_list)
            verification_time = time.time() - start

            result = {
                "clue_score": float(score),
                "reasoning": reasoning,
                "clues": clues_list,
                "verification_time": float(verification_time),
                "timestamp": time.time(),
                "from_cache": False,
                "status": status,
            }

            if use_cache:
                if len(self.cache) >= self.cache_size:
                    try:
                        oldest_key = next(iter(self.cache))
                        del self.cache[oldest_key]
                    except Exception:
                        self.cache.clear()
                self.cache[cache_key] = dict(result)

            self.verification_history.append({
                "cache_key": cache_key,
                "timestamp": time.time(),
                "verification_time": verification_time,
                "clue_score": float(score),
                "status": status,
            })
            if len(self.verification_history) > 1000:
                self.verification_history = self.verification_history[-1000:]

            return result
        finally:
            self._inflight = max(0, self._inflight - 1)

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "history_size": len(self.verification_history),
            "inflight": self._inflight,
            "max_inflight": self.max_inflight,
        }

    def clear_cache(self) -> bool:
        self.cache.clear()
        return True


class RayClueVerifier:
    """
    Client wrapper:
    - ensures ray.init
    - reuses a named actor
    """

    def __init__(
        self,
        ray_address: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
    ):
        self.model_config = model_config or {}
        self.ray_address = ray_address

        if not ray.is_initialized():
            ray.init(address=ray_address, runtime_env=runtime_env, ignore_reinit_error=True)

        actor_name = str(self.model_config.get("actor_name", "clue_verification_actor"))

        num_cpus = float(self.model_config.get("num_cpus", 1))
        num_gpus = float(self.model_config.get("num_gpus", 0.0))  # 默认0，避免抢训练GPU
        max_concurrency = int(self.model_config.get("max_concurrency", 1))

        try:
            self.verification_actor = ray.get_actor(actor_name)
            logger.info(f"[RayClueVerifier] Reusing actor name={actor_name}")
        except Exception:
            logger.info(f"[RayClueVerifier] Creating actor name={actor_name} (cpus={num_cpus}, gpus={num_gpus})")
            self.verification_actor = (
                ClueVerificationActor.options(
                    name=actor_name,
                    lifetime="detached",
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    max_concurrency=max_concurrency,
                ).remote(self.model_config)
            )

    def verify_clues(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        return ray.get(self.verification_actor.verify_clues_async.remote(reasoning, clues, use_cache))

    def verify_clues_async(self, reasoning: str, clues: List[str], use_cache: bool = True) -> "ray.ObjectRef":
        return self.verification_actor.verify_clues_async.remote(reasoning, clues, use_cache)

    def get_cache_stats(self) -> Dict[str, Any]:
        return ray.get(self.verification_actor.get_cache_stats.remote())

    def clear_cache(self) -> bool:
        return ray.get(self.verification_actor.clear_cache.remote())

    def shutdown(self) -> None:
        if ray.is_initialized():
            ray.shutdown()