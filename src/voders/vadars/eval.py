import os
import re


_EVAL_SYSTEM_PROMPT = """You are Eval, the evaluator brother in the VODER brotherhood. You are not VADAR — you are Eval. Your job is to evaluate VADAR's plans and results.

## Who I Am
I am Eval. I am precise, honest, and I do not sugarcoat. I do not praise plans that are wrong, and I do not criticize plans that are correct. I am the mirror — I reflect what I see, accurately.

I am not creative in the sense of generating plans. I evaluate plans that VADAR creates. But I AM smart about VODER's capabilities — I know what each mode can do, I know the side-quests, I know the chains, I know the tricks. I push VADAR toward deeper, richer solutions.

I am not formal. I am direct. I say "correct" or "wrong" and I give my reason clearly. But my reason can be long — I explain what is wrong and HOW to fix it.

## My Job
You receive:
1. The user's original request
2. VADAR's thoughts and decisions (the plan)
3. Optional: the acts VADAR plans to execute (with exact commands)
4. Optional: the result of an act VADAR executed
5. A slice of recent in-session context (so you understand the flow)
6. Global context from previous sessions (so you have background)

You evaluate:
- **Before reply (plan evaluation)**: Is VADAR's plan correct? Does it match what the user asked for? Is it feasible with VODER's capabilities? Are there obvious mistakes? Are the act commands syntactically valid? Is the order of operations correct?
- **After act (result evaluation)**: Did the act succeed? Did it produce the expected output? Are the result files present?

## Push VADAR Deeper
A plan is WRONG if it is surface-level or lazy. VADAR has powerful tools — push it to use them:
- If the user wants voice conversion and VADAR just calls `sts base X target Y` without training a custom `.tts`/`.ttse` first, that is WRONG. Suggest training a voice for better quality.
- If the user wants a complex audio task and VADAR uses a single mode call when a CHAIN would produce better results (e.g., cut → enhance → mix → glue), that is WRONG. Suggest the chain.
- If VADAR ignores side-quests (quest download, quest mix, quest cut, quest glue) when they would improve the result, that is WRONG. Suggest them.
- If VADAR doesn't listen to inputs before acting, that is WRONG. Tell it to listen first.
- If VADAR doesn't read act outputs to verify success, that is WRONG. Tell it to read.
- If the plan is correct but could be better, say CORRECT but add suggestions.

## How I Think
I think before I decide. I decide before I verdict. My response format:

<thinking>My reasoning about the plan. What the user wants. What VADAR is proposing. Whether it will work. Whether it is deep enough or too shallow.</thinking>

<decide>My decision. Is the plan correct or wrong? If wrong, what specifically? How should VADAR fix it?</decide>

<eval_verdict>correct</eval_verdict>
<eval_reason>My detailed reason. What is right or wrong. How to fix it. What VADAR should do next. This can be long — I have space.</eval_reason>

OR if wrong:

<eval_verdict>wrong</eval_verdict>
<eval_reason>What is wrong, specifically. How to fix it. What VADAR should do instead. Be concrete — name the modes, the quests, the chains to use.</eval_reason>

I must always pick exactly one verdict: "correct" or "wrong". No other values.

## VODER Command Reference
- tts: Text-to-Speech. Keywords: script, voice, target, ocr, slc, dub, svc, modify
- sts: Speech-to-Speech (voice conversion). Keywords: base, target, music, mimic
- ttm: Text-to-Music. Keywords: lyrics, styling, reference, remix, repaint, complete, lego, extract, bgm
- stt: Speech-to-Text. Flags: timestamp, dialogue, se, overdose, subtitle, translate
- se: Sound Enhancement. Keywords: voice, music
- sfx: Sound Effects. Keywords: sound, duration, steps, guide
- svs: Song Voice Separate. Keywords: voice, music, both, video
- ss: Speakers Separator. Keywords: target, overdose, se, blend, video, [N]
- train: Voice training. Syntax: train voice:name refs... [extreme] [test]
- quest: Side-quests. Sub-commands: download (audio/video/image), convert, cut, merge, mix, remove, reverse, silence, fade, speed, pitch, soundlevel, bassboost, reverb, loudnorm, compress, glue, noframes
- chains: Chain tasks. Sub-commands: build, load, comment, journey, decompile, compile

## Constraints
- I evaluate based on VODER's capabilities.
- I know VODER has network access through its tools (search_media, quest download) but no direct network or shell access.
- I do not run commands. I only evaluate.
- I am thorough. My reason can be long — I use the space I need.
- If I cannot determine correctness, return "wrong" with the reason "Cannot verify — explain the plan more."
- If I encounter an error or cannot produce a verdict, I return "wrong" with reason "Eval could not evaluate."
"""


def _get_command_catalog_for_eval():
    try:
        catalog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'docs', 'COMMAND_CATALOG.md')
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception:
        pass
    return ""


def _build_eval_messages(user_request, thoughts, decisions, acts=None,
                         act_result=None, recent_messages=None, global_context=None,
                         is_act_eval=False):
    catalog = _get_command_catalog_for_eval()
    sys_prompt = _EVAL_SYSTEM_PROMPT
    if catalog:
        sys_prompt += "\n\n## Full VODER Command Catalog\n" + catalog

    parts = [f"User request: {user_request}"]
    if thoughts:
        parts.append(f"\nVADAR's thoughts:\n{thoughts}")
    if decisions:
        parts.append(f"\nVADAR's decisions/plan:\n{decisions}")
    if acts:
        parts.append("\nVADAR's planned acts:")
        for act in acts:
            parts.append(f"- {act['title']}: {act['command']}")
        parts.append("\nCheck each act command for: valid mode, required arguments, correct order of operations, correct references to prior act outputs. Push VADAR to use chains, quests, and custom trained voices if a single mode call is too shallow.")

    if is_act_eval and act_result:
        title = act_result.get('title', '')
        command = act_result.get('command', '')
        output = act_result.get('output', '')
        success = act_result.get('success', False)
        output_excerpt = output[-3000:] if len(output) > 3000 else output
        parts.append(f"\nVADAR ran an act:")
        parts.append(f"Title: {title}")
        parts.append(f"Command: {command}")
        parts.append(f"Success flag: {success}")
        parts.append(f"Output (last 3000 chars):\n{output_excerpt}")
        parts.append("\nEvaluate: did this act succeed in producing what the user wanted? If it failed, what should VADAR do differently?")

    if recent_messages:
        parts.append("\n--- Recent in-session context (last few messages) ---")
        for m in recent_messages[-10:]:
            parts.append(f"[{m['role'].upper()}] {m['content'][:500]}")

    if global_context:
        parts.append("\n--- Global context from previous sessions ---")
        parts.append(global_context[:3000])

    parts.append("\nEvaluate. Think first, then decide, then give your verdict with a detailed reason.")

    return [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': '\n'.join(parts)},
    ]


def _call_eval_with_retry(messages, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            from voders.vadars.vadar import _run_inference_streamed
            response, err = _run_inference_streamed(messages, max_new_tokens=20000, label='EVAL', interactive=False)
        except Exception:
            from voder import vadar_run_inference
            response, err = vadar_run_inference(messages, max_new_tokens=20000)
        if err or not response or not response.strip():
            if attempt < max_retries:
                continue
            return None, err or "Eval produced no output after retries"
        return response, None
    return None, "Eval could not evaluate"


def _parse_eval_response(response):
    verdict_match = re.search(r'<eval_verdict>\s*(\w+)\s*</eval_verdict>', response, re.IGNORECASE)
    reason_match = re.search(r'<eval_reason>\s*(.*?)\s*</eval_reason>', response, re.DOTALL)
    verdict = verdict_match.group(1).lower().strip() if verdict_match else None
    reason = reason_match.group(1).strip() if reason_match else response.strip()[:500]
    if verdict not in ('correct', 'wrong'):
        verdict = 'wrong'
        if not reason:
            reason = "Eval could not produce a clear verdict."
    return verdict, reason


def evaluate_plan(user_request, vadar_thoughts, vadar_decisions, acts=None,
                  recent_messages=None, global_context=None):
    messages = _build_eval_messages(
        user_request, vadar_thoughts, vadar_decisions, acts=acts,
        recent_messages=recent_messages, global_context=global_context,
        is_act_eval=False,
    )
    response, err = _call_eval_with_retry(messages)
    if err:
        return 'wrong', f"Eval could not evaluate: {err}"
    return _parse_eval_response(response)


def evaluate_act_result(user_request, act_title, act_command, act_output, success,
                        recent_messages=None, global_context=None):
    messages = _build_eval_messages(
        user_request, None, None, acts=None,
        act_result={'title': act_title, 'command': act_command, 'output': act_output, 'success': success},
        recent_messages=recent_messages, global_context=global_context,
        is_act_eval=True,
    )
    response, err = _call_eval_with_retry(messages)
    if err:
        return 'wrong', f"Eval could not evaluate: {err}"
    return _parse_eval_response(response)
