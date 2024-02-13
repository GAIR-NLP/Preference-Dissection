"""
This file is to fix any failed openai completion.
Note: We always assume that "query_info" is successfully generated.
"""
from utils import *
from api_utils import *

openai.api_base = "xxxxx"
openai_api_key = "xxxxx"


openai_model = 'gpt-4-1106-preview'
temp = 0.0
max_tokens = 4096
output_type = "json_object"

if __name__ == '__main__':
    engine = OpenAIChat(api_key=openai_api_key, model=openai_model,
                        temperature=temp, max_tokens=max_tokens, top_p=1.0,
                        frequency_penalty=0, presence_penalty=0, request_timeout=60,
                        type=output_type, seed=42)


    def fix_something(adict, time=0):
        if adict.get("output", "") == "Failed!":
            newoutput = engine.generate_single({"usermsg": adict["usermsg"]})
            adict["output"] = newoutput["output"]
            adict["cost"] = newoutput["cost"]
            adict["finish_reason"] = newoutput["finish_reason"]

            if newoutput["output"] != "Failed!":
                print_colored_text(f"Fixed one `Failed!` output in {time + 1} times", "green")
                return True
            else:
                print_colored_text("Failed to fix one `Failed!` output, we do it again", "red")
                fix_something(adict, time + 1)


    other_files = ["accuracy-features.jsonl",
                   "query-aware-once-features-by-pair.jsonl",
                   "query-aware-twice-features-by-pair.jsonl",
                   "query-free-features-by-pair.jsonl",
                   "query_info.jsonl"]

    for filename in other_files:
        fn = f"./annotation_results/{filename}"
        data = read_all(fn)
        print_colored_text(f"Fixing {fn}", "yellow")
        for item in data:
            if fix_something(item):
                write_jsonl(data, fn)

    print_colored_text("Fix all done!", "cyan")
