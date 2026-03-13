import ray
import time
import hashlib
import torch
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






# 初始化Ray
@ray.remote
def init_ray_cluster(address: Optional[str] = None, runtime_env: Optional[Dict[str, Any]] = None):
    """
    Initialize a Ray cluster connection.
    
    Args:
        address: Ray cluster address to connect to
        runtime_env: Runtime environment configuration
        
    Returns:
        True if initialization was successful
    """
    if not ray.is_initialized():
        ray.init(
            address=address,
            runtime_env=runtime_env,
            ignore_reinit_error=True
        )
    return True

@ray.remote
class ClueVerificationActor:
    """
    Ray actor for clue verification using external models.
    
    This actor provides asynchronous clue verification services,
    allowing the main training process to continue while verification is in progress.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the clue verification actor.
        
        Args:
            model_config: Configuration for the external model
        """
        self.model_config = model_config or {}
        self.cache = {}
        self.cache_size = self.model_config.get("cache_size", 1000)
        self.verification_history = []
        
        # 加载模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        try:
            # 从配置中获取模型路径
            model_path = self.model_config.get("model_path")
            
            if not model_path:
                logger.error("No model_path provided in model_config. Clue verification requires a model.")
                # 不加载模型，也不使用本地验证
                self.model = None
                self.tokenizer = None
                self.generation_config = None
                return
            
            logger.info(f"Loading model from: {model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.info("Tokenizer loaded successfully")
            
            # 加载模型，使用bfloat16精度和自动设备映射
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
            
            # 初始化生成配置
            self.generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.95,
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
            logger.info("Generation config initialized")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # 出错时不回退到本地验证
            self.model = None
            self.tokenizer = None
            self.generation_config = None
        
    def _calculate_cache_key(self, reasoning: str, clues: List[str]) -> str:
        """
        Calculate a cache key for the given reasoning and clues.
        
        Args:
            reasoning: The reasoning process to verify
            clues: The list of clues to check against
            
        Returns:
            A unique cache key string
        """
        cache_input = f"{reasoning}\n{','.join(clues)}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _verify_clues_with_model(self, reasoning: str, clues: List[str]) -> float:
        """
        Verify clues using the loaded model.
        
        Args:
            reasoning: The reasoning process to verify
            clues: The list of clues to check against
            
        Returns:
            Clue verification score (0.0-1.0)
        """
        if not self.model or not self.tokenizer:
            logger.error("No model called for lead verification")
            return 0.0
        
        try:
            # 硬编码的验证提示，基于prompts.py中的内容
            system_prompt = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""
            
            # 准备clues_text
            clues_text = "\n".join([f"{i+1}. {clue}" for i, clue in enumerate(clues)])
            
            # 构建验证提示
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

            # 组合系统提示和用户提示
            final_prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {verification_prompt} [/INST]"""
            
            # 构建模型输入
            inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.model.device)
            
            # 生成验证结果
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
            
            # 解码输出
            verification_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Verification result: {verification_result}")
            
            # 解析JSON结果
            import json
            try:
                # 提取JSON部分
                json_start = verification_result.find('{')
                json_end = verification_result.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = verification_result[json_start:json_end]
                    result = json.loads(json_str)
                    
                    # 计算得分
                    total_clues = len(clues)
                    violated = len(result.get("violated_clues", []))
                    satisfied = total_clues - violated
                    
                    if total_clues > 0:
                        final_score = satisfied / total_clues
                    else:
                        final_score = 0.0
                else:
                    logger.error("Failed to extract JSON from model output")
                    final_score = 0.0
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON result: {e}")
                final_score = 0.0
            
            logger.info(f"Calculated clue score: {final_score}")
            return final_score
            
        except Exception as e:
            logger.error(f"Error in _verify_clues_with_model: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回0.0
            return 0.0
    
    def verify_clues_async(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        """
        Asynchronously verify clues against reasoning.
        
        Args:
            reasoning: The reasoning process to verify
            clues: The list of clues to check against
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        
        # Check cache if enabled
        cache_key = self._calculate_cache_key(reasoning, clues)
        if use_cache and cache_key in self.cache:
            result = self.cache[cache_key]
            result["from_cache"] = True
            result["verification_time"] = 0.0
            return result
        
        # Check if model is available
        if not self.model or not self.tokenizer:
            verification_time = time.time() - start_time
            logger.error("No model called for lead verification")
            result = {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": verification_time,
                "timestamp": time.time(),
                "from_cache": False,
                "error": "No model called for lead verification"
            }
            return result
        
        # Perform actual verification
        try:
            clue_score = self._verify_clues_with_model(reasoning, clues)
            verification_time = time.time() - start_time
            
            result = {
                "clue_score": clue_score,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": verification_time,
                "timestamp": time.time(),
                "from_cache": False
            }
            
            # Update cache
            if use_cache:
                # Remove oldest items if cache is full
                if len(self.cache) >= self.cache_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                self.cache[cache_key] = result
            
            # Update history
            self.verification_history.append({
                "cache_key": cache_key,
                "timestamp": time.time(),
                "verification_time": verification_time,
                "clue_score": clue_score
            })
            
            # Keep history size manageable
            if len(self.verification_history) > 1000:
                self.verification_history = self.verification_history[-1000:]
            
            return result
            
        except Exception as e:
            return {
                "clue_score": 0.0,
                "reasoning": reasoning,
                "clues": clues,
                "verification_time": time.time() - start_time,
                "timestamp": time.time(),
                "from_cache": False,
                "error": str(e)
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "history_size": len(self.verification_history)
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the verification cache.
        
        Returns:
            True if cache was cleared successfully
        """
        self.cache.clear()
        return True
    
    def batch_verify_clues_async(self, verification_tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Batch process multiple clue verification tasks asynchronously.
        
        Args:
            verification_tasks: List of (reasoning, clues) tuples
            use_cache: Whether to use cached results
            
        Returns:
            List of verification results
        """
        results = []
        for reasoning, clues in verification_tasks:
            result = self.verify_clues_async(reasoning, clues, use_cache)
            results.append(result)
        return results

class RayClueVerifier:
    """
    Client class for Ray-based clue verification.
    
    This class provides a high-level interface for asynchronous clue verification,
    managing the Ray actor and providing methods for verification requests.
    """
    
    def __init__(self, ray_address: Optional[str] = None, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ray clue verifier.
        
        Args:
            ray_address: Ray cluster address to connect to
            model_config: Configuration for the external model
        """
        # Initialize Ray if not already initialized
        init_result = ray.get(init_ray_cluster.remote(ray_address))
        
        # Create the verification actor
        self.verification_actor = ClueVerificationActor.remote(model_config)
        self.model_config = model_config or {}
        self.ray_address = ray_address
        
    def verify_clues(self, reasoning: str, clues: List[str], use_cache: bool = True) -> Dict[str, Any]:
        """
        Verify clues against reasoning asynchronously.
        
        Args:
            reasoning: The reasoning process to verify
            clues: The list of clues to check against
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with verification results
        """
        return ray.get(self.verification_actor.verify_clues_async.remote(reasoning, clues, use_cache))
    
    def verify_clues_async(self, reasoning: str, clues: List[str], use_cache: bool = True) -> ray.ObjectRef:
        """
        Verify clues against reasoning asynchronously, returning an ObjectRef.
        
        Args:
            reasoning: The reasoning process to verify
            clues: The list of clues to check against
            use_cache: Whether to use cached results
            
        Returns:
            Ray ObjectRef that can be used to retrieve the result later
        """
        return self.verification_actor.verify_clues_async.remote(reasoning, clues, use_cache)
    
    def batch_verify_clues(self, verification_tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Batch process multiple clue verification tasks.
        
        Args:
            verification_tasks: List of (reasoning, clues) tuples
            use_cache: Whether to use cached results
            
        Returns:
            List of verification results
        """
        return ray.get(self.verification_actor.batch_verify_clues_async.remote(verification_tasks, use_cache))
    
    def batch_verify_clues_async(self, verification_tasks: List[Tuple[str, List[str]]], use_cache: bool = True) -> ray.ObjectRef:
        """
        Batch process multiple clue verification tasks asynchronously.
        
        Args:
            verification_tasks: List of (reasoning, clues) tuples
            use_cache: Whether to use cached results
            
        Returns:
            Ray ObjectRef that can be used to retrieve the results later
        """
        return self.verification_actor.batch_verify_clues_async.remote(verification_tasks, use_cache)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return ray.get(self.verification_actor.get_cache_stats.remote())
    
    def clear_cache(self) -> bool:
        """
        Clear the verification cache.
        
        Returns:
            True if cache was cleared successfully
        """
        return ray.get(self.verification_actor.clear_cache.remote())
    
    def shutdown(self) -> None:
        """
        Shutdown the Ray connection.
        """
        if ray.is_initialized():
            ray.shutdown()

# 简化的验证函数，用于直接调用
@ray.remote
def verify_clues_remote(reasoning: str, clues: List[str]) -> float:
    """
    Remote function for direct clue verification.
    
    This function requires a model to be loaded and configured separately.
    It will return 0.0 if model verification is not available.
    
    Args:
        reasoning: The reasoning process to verify
        clues: The list of clues to check against
        
    Returns:
        Clue verification score (0.0-1.0)
    """
    logger.warning("verify_clues_remote: This function requires a model to be loaded and configured separately.")
    logger.warning("verify_clues_remote: Currently returning 0.0 as no model is available.")
    logger.warning("verify_clues_remote: Please use the ClueVerificationActor with proper model configuration instead.")
    # 没有模型配置，返回0.0
    return 0.0



def main():
    """
    Example usage of the RayClueVerifier.
    """
    # Initialize the verifier
    verifier = RayClueVerifier()
    
    # Example reasoning and clues
    example_reasoning = "Based on the clues, Alice must be in house 1 with a cat, Bob in house 2 with a dog, and Charlie in house 3 with a bird."
    example_clues = [
        "Alice lives in the red house.",
        "Bob has a dog.",
        "Charlie lives in house 3.",
        "The cat owner lives next to Bob."
    ]
    
    print("=== Synchronous Verification ===")
    result = verifier.verify_clues(example_reasoning, example_clues)
    print(f"Verification Result: {result}")
    print(f"Clue Score: {result['clue_score']:.4f}")
    print(f"Verification Time: {result['verification_time']:.4f} seconds")
    
    print("\n=== Asynchronous Verification ===")
    obj_ref = verifier.verify_clues_async(example_reasoning, example_clues)
    # Do other work while verification is in progress
    print("Doing other work while verification is in progress...")
    time.sleep(0.5)
    # Get the result
    result_async = ray.get(obj_ref)
    print(f"Asynchronous Result: {result_async}")
    print(f"Clue Score: {result_async['clue_score']:.4f}")
    print(f"Verification Time: {result_async['verification_time']:.4f} seconds")
    
    print("\n=== Cache Test ===")
    # Verify again to test cache
    result_cache = verifier.verify_clues(example_reasoning, example_clues)
    print(f"Cached Result: {result_cache}")
    print(f"From Cache: {result_cache['from_cache']}")
    print(f"Verification Time: {result_cache['verification_time']:.4f} seconds")
    
    print("\n=== Batch Verification ===")
    batch_tasks = [
        (example_reasoning, example_clues),
        (example_reasoning, example_clues[:2]),
        ("Another reasoning process", example_clues)
    ]
    batch_results = verifier.batch_verify_clues(batch_tasks)
    for i, batch_result in enumerate(batch_results):
        print(f"Batch Result {i+1}: Clue Score = {batch_result['clue_score']:.4f}")
    
    # Get cache statistics
    cache_stats = verifier.get_cache_stats()
    print(f"\nCache Statistics: {cache_stats}")
    
    # Shutdown the verifier
    verifier.shutdown()
    
    # Test direct remote function call
    print("\n=== Direct Remote Function Call ===")
    ray.init(ignore_reinit_error=True)
    remote_result = ray.get(verify_clues_remote.remote(example_reasoning, example_clues))
    print(f"Remote Function Result: {remote_result:.4f}")
    
    ray.shutdown()

if __name__ == "__main__":
    main()
