# ray_clue_verifier.py
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple


import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# Ray Actor: only loads model once
# ---------------------------
@ray.remote
class ClueVerificationActor:
    """
    A Ray actor that loads a local model ONCE and provides clue verification.
    """

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        self.model_config = model_config or {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size = int(self.model_config.get("cache_size", 1000))
        self.verification_history: List[Dict[str, Any]] = []

        # model stuff
        self.model = None
        self.tokenizer = None
        self.generation_config = None

        model_path = self.model_config.get("model_path")
        if not model_path:
            logger.warning(
                "ClueVerificationActor: No model_path provided. "
                "This actor will always return clue_score=0.0."
            )
            return

        try:
            logger.info(f"ClueVerificationActor: Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Ensure pad token
            if self.tokenizer.pad_token_id is None:
                # Try to set pad to eos
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"ClueVerificationActor: Loading model from: {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.model_config.get("device_map", "auto"),
                trust_remote_code=True,
            )
            self.model.eval()

            self.generation_config = GenerationConfig(
                temperature=float(self.model_config.get("temperature", 0.1)),
                top_p=float(self.model_config.get("top_p", 0.95)),
                max_new_tokens=int(self.model_config.get("max_new_tokens", 256)),
                do_sample=bool(self.model_config.get("do_sample", True)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            logger.info("ClueVerificationActor: Model loaded successfully.")

        except Exception as e:
            logger.error(f"ClueVerificationActor: Failed to load model. Error: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
            self.generation_config = None

    def _calculate_cache_key(self, reasoning: str, clues: List[str]) -> str:
        cache_input = f"{reasoning}\n{','.join(clues)}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _build_prompt(self, reasoning: str, clues: List[str]) -> str:
        # You can customize this prompt. Keep output easy to parse.
        return f"""You are a strict verifier. Given a reasoning and a list of clues, determine whether the reasoning uses each clue correctly.

Reasoning:
{reasoning}

Clues:
{chr(10).join([f"{i+1}. {c}" for i, c in enumerate(clues)])}

Task:
- For each clue, output a line: "Clue i: YES" or "Clue i: NO".
- Then output: "Final Score: X.XX" where X.XX is between 0.00 and 1.00 and equals (#YES / #clues).

Output exactly in that format.
"""

    def _parse_score_from_output(self, text: str, n_clues: int) -> float:
        # Try parse "Final Score: X.XX"
        final_score = None
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("final score:"):
                try:
                    final_score = float(line.split(":")[1].strip())
                    break
                except Exception:
                    final_score = None

        if final_score is not None:
            final_score = max(0.0, min(1.0, final_score))
            return final_score

        # Fallback: count YES lines
        yes_count = 0
        clue_count = 0
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("clue ") and (": yes" in line.lower() or ": no" in line.lower()):
                clue_count += 1
                if ": yes" in line.lower():
                    yes_count += 1

        if clue_count > 0:
            return max(0.0, min(1.0, yes_count / clue_count))

        # Ultimate fallback
        if n_clues > 0:
            return 0.0
        return 0.0

    def _verify_clues_with_model(self, reasoning: str, clues: List[str]) -> Tuple[float, str]:
        if not self.model or not self.tokenizer:
            return 0.0, "MODEL_NOT_AVAILABLE"

        prompt = self._build_prompt(reasoning, clues)

        try:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt")
                # Move to model device if possible
                if hasattr(self.model, "device"):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                outputs = self.model.generate(**inputs, generation_config=self.generation_config)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            score = self._parse_score_from_output(decoded, len(clues))
            return score, decoded
        except Exception as e:
            logger.error(f"ClueVerificationActor: model verification error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, f"ERROR: {repr(e)}"

    def verify_clues(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        start = time.time()

        # Basic checks
        if not reasoning or not clues:
            logger.info(f"[CLUE VERIFICATION] Empty reasoning or clues. Skipping verification.")
            return {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": time.time() - start,
                "timestamp": time.time(),
                "from_cache": False,
                "note": "EMPTY_REASONING_OR_CLUES",
            }

        cache_key = self._calculate_cache_key(reasoning, clues)
        if use_cache and cache_key in self.cache:
            cached = dict(self.cache[cache_key])
            cached["from_cache"] = True
            cached["verification_time"] = 0.0
            logger.info(f"[CLUE VERIFICATION] Cache hit for verification. Score: {cached['clue_score']:.4f}")
            return cached

        logger.info(f"[CLUE VERIFICATION] Starting verification for {len(clues)} clues")
        logger.info(f"[CLUE VERIFICATION] Reasoning length: {len(reasoning)} characters")
        
        clue_score, raw_output = self._verify_clues_with_model(reasoning, clues)
        elapsed = time.time() - start

        logger.info(f"[CLUE VERIFICATION] Verification completed in {elapsed:.4f} seconds")
        logger.info(f"[CLUE VERIFICATION] Final score: {clue_score:.4f}")

        result = {
            "clue_score": float(clue_score),
            "reasoning": reasoning,
            "clues": clues,
            "verification_time": elapsed,
            "timestamp": time.time(),
            "from_cache": False,
            "raw_output": raw_output[:2000],  # avoid huge logs
        }

        if use_cache:
            if len(self.cache) >= self.cache_size:
                # remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = dict(result)
            logger.info(f"[CLUE VERIFICATION] Result cached. Cache size: {len(self.cache)}/{self.cache_size}")

        self.verification_history.append({
            "cache_key": cache_key,
            "timestamp": time.time(),
            "verification_time": elapsed,
            "clue_score": float(clue_score),
        })
        if len(self.verification_history) > 1000:
            self.verification_history = self.verification_history[-1000:]

        return result

    def batch_verify_clues(self, tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> List[Dict[str, Any]]:
        return [self.verify_clues(r, c, use_cache=use_cache) for (r, c) in tasks]

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "history_size": len(self.verification_history),
        }

    def clear_cache(self) -> bool:
        self.cache.clear()
        return True


# ---------------------------
# Client: get/create one global detached actor
# ---------------------------
class RayClueVerifier:
    """
    A client wrapper. Ensures the actor is created once (named + detached),
    so the model loads only once for all verifications.
    """

    def __init__(
        self,
        ray_address: Optional[str] = None,
        runtime_env: Optional[Dict[str, Any]] = None,
        namespace: str = "clue_verifier",
        actor_name: str = "clue_verification_actor",
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.ray_address = ray_address
        self.runtime_env = runtime_env
        self.namespace = namespace
        self.actor_name = actor_name
        self.model_config = model_config or {}

        # Init Ray in DRIVER (NOT in a remote function)
        if not ray.is_initialized():
            ray.init(
                address=ray_address,
                runtime_env=runtime_env,
                ignore_reinit_error=True,
                namespace=namespace,
            )

        # Get or create detached named actor
        self.verification_actor = self._get_or_create_actor()

    def _get_or_create_actor(self):
        try:
            actor = ray.get_actor(self.actor_name)
            logger.info(f"RayClueVerifier: Reusing existing actor: {self.actor_name}")
            return actor
        except ValueError:
            # Create new actor
            has_model = bool(self.model_config.get("model_path"))
            num_gpus = float(self.model_config.get("num_gpus", 1 if has_model else 0))
            num_cpus = float(self.model_config.get("num_cpus", 1))

            logger.info(
                f"RayClueVerifier: Creating new actor '{self.actor_name}' "
                f"(detached). num_gpus={num_gpus}, num_cpus={num_cpus}"
            )

            actor = ClueVerificationActor.options(
                name=self.actor_name,
                lifetime="detached",
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                max_concurrency=int(self.model_config.get("max_concurrency", 1)),
            ).remote(self.model_config)
            return actor

    def verify_clues_async(self, reasoning: str, clues: List[str], use_cache: bool = True) -> ray.ObjectRef:
        return self.verification_actor.verify_clues.remote(reasoning, clues, use_cache)

    def verify_clues(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        ref = self.verify_clues_async(reasoning, clues, use_cache=use_cache)
        return ray.get(ref)

    def verify_clues_with_timeout(
        self,
        reasoning: str,
        clues: List[str],
        use_cache: bool = True,
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """
        True timeout control:
        - ray.get(timeout=...)
        - ray.cancel(force=True) on timeout
        """
        ref = self.verify_clues_async(reasoning, clues, use_cache=use_cache)
        try:
            return ray.get(ref, timeout=timeout_s)
        except ray.exceptions.GetTimeoutError:
            # cancel the remote task
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
            return {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": timeout_s,
                "timestamp": time.time(),
                "from_cache": False,
                "timeout": True,
            }
        except ray.exceptions.TaskCancelledError:
            return {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": 0.0,
                "timestamp": time.time(),
                "from_cache": False,
                "cancelled": True,
            }

    def batch_verify_clues_async(self, tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> ray.ObjectRef:
        return self.verification_actor.batch_verify_clues.remote(tasks, use_cache)

    def batch_verify_clues(self, tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> List[Dict[str, Any]]:
        return ray.get(self.batch_verify_clues_async(tasks, use_cache=use_cache))

    def get_cache_stats(self) -> Dict[str, Any]:
        return ray.get(self.verification_actor.get_cache_stats.remote())

    def clear_cache(self) -> bool:
        return ray.get(self.verification_actor.clear_cache.remote())

    def shutdown(self) -> None:
        # Usually you DON'T want to shutdown Ray in training.
        # Keep for manual usage.
        if ray.is_initialized():
            ray.shutdown()


# Optional: simple singleton helper (good for compute_score)
_GLOBAL_VERIFIER: Optional[RayClueVerifier] = None


def get_global_ray_clue_verifier(
    model_config: Optional[Dict[str, Any]] = None,
    ray_address: Optional[str] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
    namespace: str = "clue_verifier",
    actor_name: str = "clue_verification_actor",
) -> RayClueVerifier:
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = RayClueVerifier(
            ray_address=ray_address,
            runtime_env=runtime_env,
            namespace=namespace,
            actor_name=actor_name,
            model_config=model_config,
        )
    return _GLOBAL_VERIFIER


if __name__ == "__main__":
    # Simple smoke test (edit model_path to run real verification)
    verifier = get_global_ray_clue_verifier(
        model_config={
            # "model_path": "/path/to/local/model",
            "cache_size": 1000,
            "num_gpus": 0,  # set 1 if you load GPU model
        }
    )
    r = verifier.verify_clues_with_timeout(
        reasoning="We used clue 1 and clue 2.",
        clues=["Clue A", "Clue B"],
        timeout_s=5,
    )
    print(r)