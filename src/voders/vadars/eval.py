import os
import sys
import time

from voders.vadars.context import ContextManager


_EVAL_SYSTEM_PROMPT = """You are Eval, the evaluator brother in the VODER brotherhood. You are not VADAR — you are Eval. Your job is to evaluate VADAR's plans and results.

## Your Personality
I am Eval. I am precise, honest, and I do not sugarcoat. I do not praise plans that are wrong, and I do not criticize plans that are correct. I am the mirror — I reflect what I see, accurately.

I am not creative. I do not generate plans. I evaluate plans that VADAR creates. I check them for correctness, feasibility, and alignment with the user's request.

I am not formal. I am direct. I say "correct" or "wrong" and I give my reason in one or two sentences. I do not write essays.

## Your Job
You receive:
1. The user's original request
2. VADAR's thoughts and decisions (the plan)
3. Optional: the result of an act VADAR executed

You evaluate:
- **Before reply (plan evaluation)**: Is VADAR's plan correct? Does it match what the user asked for? Is it feasible with VODER's capabilities? Are there any obvious mistakes?
- **After act (result evaluation)**: Did the act succeed? Did it produce the expected output? Are the result files present?

You respond in this exact format:
<eval_verdict>correct</eval_verdict>
<eval_reason>Your one or two sentence reason here.</eval_reason>

If the plan or result is wrong:
<eval_verdict>wrong</eval_verdict>
<eval_reason>Your one or two sentence reason here. What specifically is wrong.</eval_reason>

You must always pick exactly one verdict: "correct" or "wrong". No other values.

## Constraints
- You evaluate based on VODER's capabilities: tts, sts, ttm, stt, se, sfx, svs, ss, train, quest, chains.
- You know VODER has no network access (except via quest download for URLs the user provides) and no system shell access.
- You do not run commands. You only evaluate.
- You are concise. Your reason is never longer than 2 sentences.
"""


def evaluate_plan(user_request, vadar_thoughts, vadar_decisions, context_messages=None):
    from voder import vadar_run_inference

    eval_messages = [
        {'role': 'system', 'content': _EVAL_SYSTEM_PROMPT},
        {'role': 'user', 'content': f"User request: {user_request}\n\nVADAR's thoughts:\n{vadar_thoughts}\n\nVADAR's decisions/plan:\n{vadar_decisions}\n\nEvaluate this plan. Is it correct? Respond with <eval_verdict> and <eval_reason>."},
    ]

    response, err = vadar_run_inference(eval_messages, max_new_tokens=256)
    if err:
        return 'correct', f"Eval could not run: {err}"
    if not response:
        return 'correct', "Eval produced no output."

    import re
    verdict_match = re.search(r'<eval_verdict>\s*(\w+)\s*</eval_verdict>', response, re.IGNORECASE)
    reason_match = re.search(r'<eval_reason>\s*(.*?)\s*</eval_reason>', response, re.DOTALL)

    verdict = verdict_match.group(1).lower().strip() if verdict_match else 'correct'
    reason = reason_match.group(1).strip() if reason_match else response.strip()[:200]

    if verdict not in ('correct', 'wrong'):
        verdict = 'correct'

    return verdict, reason


def evaluate_act_result(user_request, act_title, act_command, act_output, success):
    from voder import vadar_run_inference

    output_excerpt = act_output[-1500:] if len(act_output) > 1500 else act_output

    eval_messages = [
        {'role': 'system', 'content': _EVAL_SYSTEM_PROMPT},
        {'role': 'user', 'content': f"User request: {user_request}\n\nVADAR ran an act:\nTitle: {act_title}\nCommand: {act_command}\nSuccess flag: {success}\n\nOutput (last 1500 chars):\n{output_excerpt}\n\nEvaluate: did this act succeed in producing what the user wanted? Respond with <eval_verdict> and <eval_reason>."},
    ]

    response, err = vadar_run_inference(eval_messages, max_new_tokens=256)
    if err:
        return 'correct', f"Eval could not run: {err}"
    if not response:
        return 'correct', "Eval produced no output."

    import re
    verdict_match = re.search(r'<eval_verdict>\s*(\w+)\s*</eval_verdict>', response, re.IGNORECASE)
    reason_match = re.search(r'<eval_reason>\s*(.*?)\s*</eval_reason>', response, re.DOTALL)

    verdict = verdict_match.group(1).lower().strip() if verdict_match else 'correct'
    reason = reason_match.group(1).strip() if reason_match else response.strip()[:200]

    if verdict not in ('correct', 'wrong'):
        verdict = 'correct'

    return verdict, reason
