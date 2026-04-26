from typing import Dict, List, Any

def calculate_pass_at_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate pass@k metric for a list of results.
    
    Args:
        results: List of solution results
        k: Number of attempts to consider
        
    Returns:
        pass@k score (float between 0 and 1)
    """
    if not results:
        return 0.0
    
    pass_count = 0
    for result in results:
        # Check if any sample in the first k samples is correct
        samples = result.get("all_samples_results", [])
        if any(sample["is_correct"] for sample in samples[:k]):
            pass_count += 1
    
    return pass_count / len(results)

def calculate_accuracy(result: Dict[str, Any]) -> float:
    """
    Calculate accuracy for a single result.
    
    Args:
        result: Solution result
        
    Returns:
        Accuracy score (float between 0 and 1)
    """
    if "best_score" in result:
        return result["best_score"]
    
    # Try to calculate from round scores
    samples = result.get("all_samples_results", [])
    if samples:
        best_scores = [sample["best_score"] for sample in samples]
        return max(best_scores) if best_scores else 0.0
    
    return 0.0

def calculate_verification_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate verification statistics for a list of results.
    
    Args:
        results: List of solution results
        
    Returns:
        Dictionary with verification statistics
    """
    total_verification_steps = 0
    total_false_positives = 0
    total_false_negatives = 0
    total_samples = 0
    samples_with_verification = 0
    total_verification_ratio = 0.0
    
    for result in results:
        sample_results = result.get("all_samples_results", [])
        total_samples += len(sample_results)
        
        for sample in sample_results:
            verification_stats = sample.get("verification_stats", {})
            verification_results = verification_stats.get("verification_results", [])
            
            if verification_results:
                samples_with_verification += 1
                total_verification_steps += verification_stats.get("total_verification_steps", 0)
                # Each sample can contribute at most 1 false positive, but multiple false negatives
                total_false_positives += 1 if verification_stats.get("false_positives", 0) > 0 else 0
                total_false_negatives += verification_stats.get("false_negatives", 0)  # Accumulate all false negatives
                total_verification_ratio += verification_stats.get("avg_verification_ratio", 0)
    
    avg_verification_ratio = total_verification_ratio / samples_with_verification if samples_with_verification > 0 else 0.0
    avg_verification_steps = total_verification_steps / samples_with_verification if samples_with_verification > 0 else 0
    
    return {
        "total_verification_steps": total_verification_steps,
        "avg_verification_steps": avg_verification_steps,
        "avg_verification_ratio": avg_verification_ratio,
        "total_false_positives": total_false_positives,
        "total_false_negatives": total_false_negatives,
        "samples_with_verification": samples_with_verification,
        "total_samples": total_samples
    }

def generate_summary_report(results: List[Dict[str, Any]], max_attempts: int, max_n_samples: int) -> Dict[str, Any]:
    """
    Generate a comprehensive summary report of all results.
    
    Args:
        results: List of solution results
        max_attempts: Maximum number of attempts per sample
        max_n_samples: Maximum number of samples per puzzle
        
    Returns:
        Dictionary with summary report
    """
    # Calculate total tasks (number of puzzles evaluated)
    total_tasks = len(results)
    
    # Calculate total samples across all results
    total_samples = sum(len(result.get("all_samples_results", [])) for result in results)
    
    # Calculate pass@k for all k values (k = number of samples per puzzle)
    pass_at_k = {}
    for k in range(1, max_n_samples + 1):
        pass_at_k[k] = calculate_pass_at_k(results, k)
    
    # Calculate overall accuracy
    accuracies = [calculate_accuracy(result) for result in results]
    overall_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    # Calculate verification statistics
    verification_stats = calculate_verification_stats(results)
    
    # Calculate outcome counts
    successful_tasks = sum(1 for result in results if any(sample["is_correct"] for sample in result.get("all_samples_results", [])))
    false_positive_tasks = sum(1 for result in results if any(sample["solution_status"] == "false_positive" for sample in result.get("all_samples_results", [])))
    answer_avoided_tasks = sum(1 for result in results if all(sample["solution_status"] == "havent found any solution yet" for sample in result.get("all_samples_results", [])))
    
    # Calculate n_stats (ALL N VALUES SUMMARY) - following reference code logic
    # N represents the maximum number of attempts allowed per sample
    # Each puzzle is counted in all N values >= its actual attempts used
    # For each puzzle and N, exactly one category is assigned with priority: success > false_positive > answer_avoided
    n_stats = {}
    for n in range(0, max_attempts + 1):
        n_stats[n] = {
            "total_tasks": total_tasks,  # All puzzles are counted for all N values
            "successful_tasks": 0,
            "false_positive_tasks": 0,
            "answer_avoided_tasks": 0,
            "total_score": 0.0
        }
    
    # Calculate for each puzzle and count in all applicable N values
    for result in results:
        sample_results = result.get("all_samples_results", [])
        
        # Collect round scores for each sample
        sample_scores = []
        for sample in sample_results:
            scores = sample.get("round_scores", [])
            if not isinstance(scores, list):
                scores = []
            sample_scores.append(scores)
        
        # For each N value >= 0, determine the category for this puzzle
        for n in n_stats:
            success_n = False
            false_positive_n = False
            answer_avoided_n = False
            
            # Step 1: Check if any sample succeeded within n attempts (highest priority)
            if n == 0:
                # For n=0, check first attempt of any sample
                for scores in sample_scores:
                    if len(scores) > 0 and scores[0] == 1.0:
                        success_n = True
                        break
                # For n=0, no false positives or answer avoided - only success or incorrect
                # So false_positive_n and answer_avoided_n remain False
            else:
                # For n>0, check if any sample has a successful attempt within first n attempts
                for s, scores in zip(sample_results, sample_scores):
                    upto_n = scores[:min(n, len(scores))]
                    if any(sc == 1.0 for sc in upto_n):
                        success_n = True
                        break
            
            if not success_n and n > 0:
                # Step 2: Check if any sample had a false positive within n attempts (medium priority)
                # Only check for false positives if n > 0
                for sample in sample_results:
                    if sample.get("solution_status") == "false_positive":
                        # Find the first attempt where FP occurred
                        fp_occurred = False
                        attempt_statuses = sample.get("attempt_statuses", [])
                        
                        # For each sample, check if FP occurred in any attempt within n rounds
                        # FP occurs when verification passed but ground truth check failed
                        for attempt_idx, attempt in enumerate(attempt_statuses):
                            if attempt_idx < n:  # Check within first n attempts
                                if attempt.get("verification_passed", False) and not attempt.get("is_correct", False):
                                    false_positive_n = True
                                    fp_occurred = True
                                    break
                        
                        # If we found FP in any sample, break out of loop
                        if fp_occurred:
                            break
            
            # Step 3: If not successful or false positive, it's answer avoided by default (lowest priority)
            # Only set answer_avoided_n if n > 0
            answer_avoided_n = not success_n and not false_positive_n and n > 0
            
            # Ensure mutual exclusivity: exactly one category per puzzle
            if success_n:
                n_stats[n]["successful_tasks"] += 1
            elif false_positive_n:
                n_stats[n]["false_positive_tasks"] += 1
            elif answer_avoided_n:
                n_stats[n]["answer_avoided_tasks"] += 1
            elif n == 0:
                # For n=0, if not successful, it's just incorrect, not answer avoided
                # So we don't increment any other category
                pass
        
        # Calculate best score for this N
        for n in n_stats:
            best_score_for_n = 0.0
            for scores in sample_scores:
                if not scores:
                    continue
                if n == 0:
                    # For n=0, use the first attempt score
                    score_n = scores[0] if len(scores) > 0 else 0.0
                else:
                    # For n>0, use the max score in first n attempts
                    upto_n = scores[:min(n, len(scores))]
                    score_n = max(upto_n) if upto_n else 0.0
                if score_n > best_score_for_n:
                    best_score_for_n = score_n
            
            n_stats[n]["total_score"] += best_score_for_n
    
    # Calculate average_k_scores (AVERAGE@K SUMMARY)
    # K represents the maximum number of attempts allowed per sample
    # Calculate average best score when considering up to K attempts per sample
    average_k_scores = {}
    # Determine the range of K values to calculate
    # When max_attempts=0, we still need to calculate K=1 (initial attempt only)
    start_k = 1
    end_k = max_attempts if max_attempts > 0 else 1
    for k in range(start_k, end_k + 1):
        total_best_score = 0.0
        count = 0
        
        for result in results:
            sample_results = result.get("all_samples_results", [])
            if not sample_results:
                continue
            
            # For each puzzle, consider only the first sample
            sample = sample_results[0]
            round_scores = sample.get("round_scores", [])
            if not isinstance(round_scores, list):
                round_scores = []
            
            # Get best score within first k attempts
            if round_scores:
                # Take up to k attempts
                considered_scores = round_scores[:k]
                best_score = max(considered_scores) if considered_scores else 0.0
                total_best_score += best_score
                count += 1
        
        average_k_scores[k] = total_best_score / count if count > 0 else 0.0
    
    return {
        "total_tasks": total_tasks,
        "total_samples": total_samples,
        "successful_tasks": successful_tasks,
        "false_positive_tasks": false_positive_tasks,
        "answer_avoided_tasks": answer_avoided_tasks,
        "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0.0,
        "overall_accuracy": overall_accuracy,
        "pass_at_k": pass_at_k,
        "verification_stats": verification_stats,
        "n_stats": n_stats,
        "average_k_scores": average_k_scores,
        "max_attempts": max_attempts,
        "max_n_samples": max_n_samples
    }

def print_summary_report(report: Dict[str, Any]) -> None:
    """
    Print a formatted summary report with the desired output format.
    
    Args:
        report: Summary report dictionary
    """
    print("\n" + "=" * 60)
    print("ALL N VALUES SUMMARY")
    print("=" * 60)
    print(f"{'N':<5} {'Accuracy':<15} {'False Positive':<15} {'Answer Avoided':<15} {'Total Tasks':<15}")
    print("-" * 60)
    
    # Print N values summary
    for n in sorted(report['n_stats'].keys()):
        # Skip N=0 when max_attempts is not 0
        if n == 0 and report['max_attempts'] != 0:
            continue
        
        stats = report['n_stats'][n]
        if stats['total_tasks'] > 0:
            # According to original code, accuracy here means successful tasks percentage, not average score
            accuracy = (stats['successful_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0.0
            false_positive_rate = (stats['false_positive_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0.0
            answer_avoided_rate = (stats['answer_avoided_tasks'] / stats['total_tasks'] * 100) if stats['total_tasks'] > 0 else 0.0
            print(f"{n:<5} {accuracy:<15.2f} {false_positive_rate:<15.2f} {answer_avoided_rate:<15.2f} {stats['total_tasks']:<15}")
    
    # Print PASS@K SUMMARY
    print("\n" + "=" * 60)
    print("PASS@K SUMMARY")
    print("=" * 60)
    print(f"{'K':<5} {'pass@k':<15} {'Pass Tasks':<15} {'Total Tasks':<15}")
    print("-" * 60)
    
    total_tasks = report['total_tasks']
    for k, pass_at_k in report['pass_at_k'].items():
        # According to original code, pass@k is in percentage format
        pass_rate = pass_at_k * 100
        pass_tasks = int(pass_at_k * total_tasks)
        print(f"{k:<5} {pass_rate:<15.2f} {pass_tasks:<15} {total_tasks:<15}")
    
    # Print AVERAGE@K SUMMARY
    print("\n" + "=" * 60)
    print("AVERAGE@K SUMMARY")
    print("=" * 60)
    print(f"{'K':<5} {'average@k':<15} {'Total Samples':<15}")
    print("-" * 60)
    
    # Use total_samples directly from report
    total_samples = report['total_samples']
    
    for k, score in report['average_k_scores'].items():
        print(f"{k:<5} {score:<15.4f} {total_samples:<15}")
    
    # Print VERIFICATION STATISTICS (optional, kept for completeness)
    print("\n" + "=" * 60)
    print("VERIFICATION STATISTICS")
    print("=" * 60)
    ver_stats = report['verification_stats']
    print(f"Total Verification Steps: {ver_stats['total_verification_steps']}")
    print(f"Avg Verification Ratio: {ver_stats['avg_verification_ratio']:.4f}")
    print(f"Avg Verification Steps per Sample: {ver_stats['avg_verification_steps']:.2f}")
    print(f"Total False Positives: {ver_stats['total_false_positives']}")
    print(f"Total False Negatives: {ver_stats['total_false_negatives']}")
    print(f"Samples with Verification: {ver_stats['samples_with_verification']}/{ver_stats['total_samples']}")

def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple results into a single summary.
    
    Args:
        results: List of solution results
        
    Returns:
        Aggregated results dictionary
    """
    if not results:
        return {}
    
    aggregated = {
        "all_results": results,
        "total_puzzles": len(results),
        "total_samples": sum(len(result.get("all_samples_results", [])) for result in results),
        "correct_samples": sum(sum(1 for sample in result.get("all_samples_results", []) if sample["is_correct"]) for result in results),
        "false_positive_samples": sum(sum(1 for sample in result.get("all_samples_results", []) if sample["solution_status"] == "false_positive") for result in results),
        "answer_avoided_samples": sum(sum(1 for sample in result.get("all_samples_results", []) if sample["solution_status"] == "havent found any solution yet") for result in results),
        "avg_attempts_used": sum(result.get("attempts_used", 0) for result in results) / len(results) if results else 0,
        "best_scores": [result.get("best_score", 0) for result in results],
        "round_scores": [score for result in results for score in result.get("round_scores", [])]
    }
    
    return aggregated
