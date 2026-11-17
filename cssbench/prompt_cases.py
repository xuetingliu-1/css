
from .prompts import sp_cot, sc, self_refine as sr, debate, css


def cases():
    result = {}
    for n in (4, 5):
        q = 'A patient has symptom α. Which option is best?'
        opts = {chr(65+i): 'Option ' + str(i+1) for i in range(n)}
        prefix = str(n) + 'op/'
        for mode in ('sp', 'cot'):
            for system in (False, True):
                result[prefix+mode+'/system='+str(system)] = sp_cot.build_messages(q, opts, mode, system)
            result[prefix+'sc/'+mode] = sc.build_messages(q, sc.format_options_parentheses(opts)[0], mode)
        inp = sr.build_input_text(q, sr.format_options_medins(opts))
        result[prefix+'self_refine/init'] = sr.build_messages(sr.sr_init_instruction(), inp)
        result[prefix+'self_refine/feedback'] = sr.build_messages(sr.sr_feedback_instruction(), f'{inp}\n\nProposed answer:\nA\n\nFeedback:\n')
        result[prefix+'self_refine/refine'] = sr.build_messages(sr.sr_refine_instruction(), f'{inp}\n\nProposed answer:\nA\n\nFeedback:\nCheck B.\nStop: NO\n\nFinal answer:\n')
        for i in range(5):
            result[prefix+'debate/persona'+str(i)] = debate.build_messages(debate.build_system_for_agent('persona', i), inp)
        result[prefix+'debate/sampling'] = debate.build_messages(debate.build_system_for_agent('sampling', 0), inp)
        for style in ('short', 'long'):
            result[prefix+'debate/'+style] = debate.build_messages(debate.build_system_for_agent('persona', 0), debate.build_debate_query(inp, 'Answer: A', ['Answer: B', 'Answer: C'], style))
        result[prefix+'debate/judge'] = debate.build_messages(debate.JUDGE_SYSTEM, debate.build_judge_query(inp, ['Answer: B', 'Answer: A']))
        block = css.format_options(opts)
        for letter in opts:
            result[prefix+'css/agent'+letter] = [{'role':'system','content':css.build_agent_system(letter)}, {'role':'user','content':css.build_agent_user(q,block,letter)}]
        scenarios = {k:{'rationale':'Evidence supporting '+k} for k in opts}
        for reverse in (False, True):
            order = list(opts)[::-1] if reverse else list(opts)
            result[prefix+'css/judge/reversed='+str(reverse)] = [{'role':'system','content':css.build_judge_system()}, {'role':'user','content':css.build_judge_user(q,block,scenarios,order)}]
    return result
