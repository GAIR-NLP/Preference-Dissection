from utils import *
from api_utils import *

openai.api_base = "xxxxx"
openai_api_key = "xxxxx"

openai_model = 'gpt-4-1106-preview'
temp = 0.0
max_tokens = 4096
output_type = "text"

if __name__ == '__main__':
    engine = OpenAIChat(api_key=openai_api_key, model=openai_model,
                        temperature=temp, max_tokens=max_tokens, top_p=1.0,
                        frequency_penalty=0, presence_penalty=0, request_timeout=600,
                        type=output_type, seed=42)

    data = read_all("./raw_data/sample_unannotated.jsonl")
    of = "./raw_data/gpt4turbo-references.jsonl"

    end = 2
    batchsize = 20
    total_cost = 0.0

    batched_inputs = []

    for xid in range(end):
        item = data[xid]
        prompt = item["prompt"]
        batched_inputs.append({"usermsg": prompt})

    already_have = len(read_jsonl(of))

    batched_generate_with_write(engine, batched_inputs, of, batch_size=batchsize, already_have=already_have)
