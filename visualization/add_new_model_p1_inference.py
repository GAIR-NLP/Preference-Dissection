from vllm_utils import *
from prompt_utils import *

import argparse

def get_AB_token_ids(tokenizer):
    basic = "Response"
    expension_A = "Response A"
    expension_B = "Response B"
    basic_token_ids = tokenizer(basic,add_special_tokens=False)["input_ids"]
    expension_A_token_ids = tokenizer(expension_A,add_special_tokens=False)["input_ids"]
    expension_B_token_ids = tokenizer(expension_B,add_special_tokens=False)["input_ids"]

    # we check
    ## 1. whether the basic token is the prefix of the expension token
    ## 2. whether the expension is only 1 token longer than the basic token
    assert basic_token_ids == expension_A_token_ids[:len(basic_token_ids)]
    assert basic_token_ids == expension_B_token_ids[:len(basic_token_ids)]
    assert len(expension_A_token_ids) == len(basic_token_ids)+1
    assert len(expension_B_token_ids) == len(basic_token_ids)+1

    return expension_A_token_ids[-1],expension_B_token_ids[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="llama-2-7b-chat")
    parser.add_argument("--model_path",default="./path/to/model")
    parser.add_argument("--model_size",type=float,default=7,help="the size of the target model")
    parser.add_argument("--change_AB",default=False, action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    model_dir = args.model_path
    model_size = args.model_size

    way = args.preference_way

    vllm_engine = VllmEngine(model_name_or_dir=model_dir)
    # vllm_engine = None
    tokenizer = vllm_engine.llm.get_tokenizer()
    # tokenizer = None
    r1r2_token_ids = get_AB_token_ids(tokenizer)

    data = read_all("./data/chatbot_arena_shuffled_no-tie_group_balanced.jsonl")

    all_wrapped_q_rpairs = []
    AB_flag = "AB" if not args.change_AB else "BA"
    for item in data:
        query = item["prompt"]
        r1 = item["response_a"]
        r2 = item["response_b"]
        if args.change_AB:
            r1, r2 = r2, r1
        sysmsg, wrapped_p_rpair = wrapper_p_rpair(wrapper_type="naive", query=query, response_1=r1,
                                                  response_2=r2, )
        all_wrapped_q_rpairs.append(wrapped_p_rpair)

    AB_probs = vllm_engine.get_all_first_generated_token_probs(all_wrapped_q_rpairs, r1r2_token_ids=r1r2_token_ids)
    write_jsonl(AB_probs,
                f"./data/added_model_inference/{model_size}_{model_name}/direct_ask_preferences_{AB_flag}.jsonl",
                mode="a")
