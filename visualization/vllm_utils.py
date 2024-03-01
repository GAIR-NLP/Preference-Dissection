from utils import *

try:
    from vllm import LLM, SamplingParams

    hasvllm = True
except:
    print_colored_text("[WARNING] vllm is not installed, but vllm_utils is imported!", "red")
import torch


class VllmEngine:
    def __init__(self, max_num_batched_tokens=None, dummy_tokenizer=None, model_name_or_dir=None,
                 num_response_per_query=1,
                 temperature=0.9, top_p=1.0, max_new_tokens=10, stop=None, presence_penalty=0.0, frequency_penalty=0.0,
                 stop_token_ids=None):

        num_gpus = torch.cuda.device_count()
        another_args = {}
        if max_num_batched_tokens is not None: another_args['max_num_batched_tokens']=max_num_batched_tokens
        if dummy_tokenizer is not None: another_args['tokenizer'] = dummy_tokenizer
        self.llm = LLM(model=model_name_or_dir, tensor_parallel_size=num_gpus, trust_remote_code=True, **another_args)
        self.vocab_size = self.llm.get_tokenizer().vocab_size
        self.tokenizer = self.llm.get_tokenizer()
        print('>>>>>> model loaded')
        # part 2 we set the sampling params
        self.num_response_per_query = num_response_per_query
        self.sampling_params = SamplingParams(n=self.num_response_per_query,
                                              temperature=temperature, top_p=top_p,
                                              max_tokens=max_new_tokens,
                                              stop=stop, presence_penalty=presence_penalty,
                                              frequency_penalty=frequency_penalty,stop_token_ids=stop_token_ids)

    def set_sampling_params(self, mode="get_prompt_probs"):
        if mode == "get_prompt_probs":
            self.sampling_params = SamplingParams(n=1, temperature=0, top_p=1.0,
                                                  max_tokens=1, presence_penalty=0.0,
                                                  frequency_penalty=0.0, logprobs=0, prompt_logprobs=0)
        elif mode == "use_all_first_generated_token":
            self.sampling_params = SamplingParams(n=1, temperature=0, top_p=1.0,
                                                  max_tokens=1, presence_penalty=0.0,
                                                  frequency_penalty=0.0, logprobs=self.vocab_size)
        else:
            raise NotImplementedError

    def generate_batch(self, processed_prompts, sampling_params=None, direct_msg_list=False):
        print_colored_text(
            "[INFO] Need to generate {} samples x {}".format(len(processed_prompts), self.num_response_per_query),
            "green")
        returned = []
        if sampling_params is None: sampling_params = self.sampling_params
        outputs = self.llm.generate(processed_prompts, sampling_params)

        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        for iid, item in enumerate(sorted_outputs):
            if self.num_response_per_query == 1:
                sample = {"output": item.outputs[0].text, "logprob": item.outputs[0].cumulative_logprob,
                          "finish_reason": item.outputs[0].finish_reason, "id": item.request_id}
            else:
                sample = {"outputs": [o.text for o in item.outputs],
                          "logprobs": [o.cumulative_logprob for o in item.outputs],
                          "finish_reasons": [o.finish_reason for o in item.outputs], "id": item.request_id}
            returned.append(sample)
        return returned

    def get_prompt_token_probs(self, processed_prompts, prompt_response_tokens=None):
        self.set_sampling_params(mode="get_prompt_probs")
        return self._get_prompt_token_probs(processed_prompts, prompt_response_tokens)

    def _get_prompt_token_probs(self, processed_prompts, prompt_response_tokens=None):
        if prompt_response_tokens is None:
            print_colored_text("prompt_response_tokens is None, we only return the whole input as prompt.")
            prompt_response_tokens = []
            for item in processed_prompts:
                prompt_response_tokens.append((len(self.tokenizer.encode(item)),0))
        assert len(processed_prompts) == len(prompt_response_tokens)
        outputs = self.llm.generate(processed_prompts, self.sampling_params)
        all_probs = []
        for oid, output in enumerate(outputs):
            accumulated_logprob_prompt = 0.0
            accumulated_logprob_respose = 0.0
            assert len(output.prompt_logprobs) == sum(prompt_response_tokens[oid])
            for idx, item in enumerate(output.prompt_logprobs):
                if item is None:
                    prob = 0.0
                else:
                    assert len(item) == 1
                    prob = list(item.values())[0]
                if idx < prompt_response_tokens[oid][0]:
                    accumulated_logprob_prompt += prob
                else:
                    accumulated_logprob_respose += prob
            all_probs.append(
                {"prompt": {"tokens": prompt_response_tokens[oid][0], "logprob": accumulated_logprob_prompt},
                 "response": {"tokens": prompt_response_tokens[oid][1], "logprob": accumulated_logprob_respose}}
            )
        return all_probs

    def get_all_first_generated_token_probs(self, processed_prompts,r1r2_token_ids):
        self.set_sampling_params(mode="use_all_first_generated_token")
        return self._get_all_first_generated_token_probs(processed_prompts,r1r2_token_ids)

    def _get_all_first_generated_token_probs(self, processed_prompts, r1r2_token_ids):
        outputs = self.llm.generate(processed_prompts, self.sampling_params)
        all_probs = []
        for oid, output in enumerate(outputs):
            assert len(output.outputs)==1
            if len(output.outputs[0].logprobs)>=1:
                if len(output.outputs[0].logprobs)>1: print_colored_text(f"[WARNING] Strange bug! len(output.outputs[0].logprobs) = {len(output.outputs[0].logprobs)}","yellow")
                output_logprobs = output.outputs[0].logprobs[0]
                r1_prob = output_logprobs[r1r2_token_ids[0]]
                r2_prob = output_logprobs[r1r2_token_ids[1]]
                all_probs.append({"response_a": r1_prob, "response_b": r2_prob})
            else:
                print_colored_text(f"[WARNING] Strange bug! len(output.outputs[0].logprobs) = {len(output.outputs[0].logprobs)}", "red")
                all_probs.append({"response_a": 0.0, "response_b": 0.0})
        return all_probs