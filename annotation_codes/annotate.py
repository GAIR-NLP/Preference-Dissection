"""
In this file, we annotate the samples.
Note, to make things work properly, we may first need to annotate the query_info, since query-twice-aware features are based on query_info.
"""
# the complete pipeline for annotating the samples
import copy
import json

from api_utils import *
from utils import *


###################################
# Your OpenAI configuration here
###################################
openai.api_base = "https://lonlie.plus7.plus/v1"
openai_api_key = "sk-w00Jrh7xFC3o10vXC4Df2961Ca3d4c25B34fE6Fb60Fd8704"


openai_model = 'gpt-4-1106-preview'
temp = 0.0
max_tokens = 4096
timeout = 120
output_type = "json_object"

###################################
# Templates
###################################

# step 1, prepare templates
# 1.1 query info collection
query_info_collection = read_yaml(f"./prompts/check_query.yaml")
# 1.2 query free
query_free_pairwise = read_yaml(f"./prompts/query-free_pairwise.yaml")
query_free_single = read_yaml(f"./prompts/query-free_single.yaml")
# 1.3 query aware twice
query_aware_once_pairwise = read_yaml(f"./prompts/query-aware-once_pairwise.yaml")
query_aware_once_single = read_yaml(f"./prompts/query-aware-once_single.yaml")
# 1.4 query aware twice
query_aware_twice_pairwise = read_yaml(f"./prompts/query-aware-twice_pairwise.yaml")
query_aware_twice_single = read_yaml(f"./prompts/query-aware-twice_single.yaml")
# 1.5 accuracy
accuracy_check_single = read_yaml(f"./prompts/accuracy_single.yaml")
accuracy_check_pairwise = read_yaml(f"./prompts/accuracy_pairwise.yaml")

def build_criteria(raw_criteria):
    str = ""
    for k, v in raw_criteria.items():
        str += f"{k}: {v['content']}\n"
    return str.strip()

query_free_characteristics = read_yaml(f"./prompts/query_free_characteristics.yaml")
query_aware_once_characteristics = read_yaml(f"./prompts/query_aware_once_characteristics.yaml")

query_free_pairwise["template"] = query_free_pairwise["template"].replace("{characteristics}", build_criteria(query_free_characteristics))
query_free_single["template"] = query_free_single["template"].replace("{characteristics}", build_criteria(query_free_characteristics))

query_aware_once_pairwise["template"] = query_aware_once_pairwise["template"].replace("{characteristics}", build_criteria(query_aware_once_characteristics))
query_aware_once_single["template"] = query_aware_once_single["template"].replace("{characteristics}", build_criteria(query_aware_once_characteristics))

if __name__ == '__main__':
    # step 0, build engine
    engine = OpenAIChat(api_key=openai_api_key, model=openai_model,
                        temperature=temp, max_tokens=max_tokens, top_p=1.0,
                        frequency_penalty=0, presence_penalty=0, request_timeout=timeout,
                        type=output_type, seed=42)

    # step -1, read all samples
    samples = read_all("./raw_data/sample_unannotated.jsonl")
    gpt4_refs = read_all("./raw_data/gpt4turbo-references.jsonl")

    def process_a_batch(startid,endid, batch_tag="", collect_type="pair", step1=True, step2=True, step3=True, step4=True, step5=True,):

        """
        :param batch:
        :param batch_tag:
        :param collect_type:
        :param step1and2: query info and query aware twice
        :param step3: query aware once
        :param step4: query free
        :param step5: information richness
        :param step6: accuracy
        :return:
        """
        total_cost = 0.0

        if step1:
            ################################################################
            # step 1, get query info
            ################################################################

            alreadyhavedata = read_all(f"./annotation_results/query_info.jsonl")

            query_info_dicts = alreadyhavedata[startid:endid]

            if len(query_info_dicts) == endid - startid and all([query_info_dict.get("finish_reason", "") in ["stop",""] for query_info_dict in query_info_dicts]):
                print_colored_text(f"Query info collection done! - {batch_tag}, cost: 0.0", color="green")
            else:
                all_query_collection_instances = []
                for sample in samples[startid:endid]:
                    prompt = sample["prompt"]
                    query_info_collection_instance = query_info_collection["template"].replace("{prompt}", prompt)
                    all_query_collection_instances.append({"usermsg": query_info_collection_instance})

                query_info_dicts = engine.generate_batch(all_query_collection_instances)
                step1_cost = sum([float(query_info_dict["cost"]) for query_info_dict in query_info_dicts])
                total_cost += step1_cost

                assert len(query_info_dicts) == len(samples[startid:endid])
                for idx, query_info_dict in enumerate(query_info_dicts):
                    query_info_dict["tag"] = f"{samples[startid:endid][idx]['id']}_query-info"
                    if query_info_dict.get("finish_reason", "") not in ["stop","length",""]:
                        query_info_dict["usermsg"] = all_query_collection_instances[idx]["usermsg"]
                write_jsonl(query_info_dicts, f"./annotation_results/query_info.jsonl", mode="a")
                print_colored_text(f"\nQuery info collection done! - {batch_tag}, cost: {step1_cost}", color="green")

        if step2:
            ################################################################
            # step 2, get query-aware-twice features
            ################################################################

            all_question_output_strs = []
            mapping = {}
            for idx, query_info_dict in enumerate(query_info_dicts):
                query_info_dict = json.loads(query_info_dict["output"])
                question_str,output_str = query_info_to_questions(query_info_dict, pair_ver=(collect_type=="pair"))
                if question_str == "":
                    continue
                all_question_output_strs.append([idx, question_str, output_str])
                mapping[idx] = len(all_question_output_strs) - 1

            # step 2.1, generate
            rtemplate = query_aware_twice_pairwise["template"] if collect_type == "pair" else query_aware_twice_single["template"]
            all_query_aware_twice_instances = []
            tags = []
            for item in all_question_output_strs:
                idx, question_str, output_str = item
                current_sample = samples[startid:endid][idx]
                prompt,response_1,response_2 = current_sample["prompt"],current_sample["response_a"],current_sample["response_b"]
                mtemplate = rtemplate.replace("{prompt}", prompt).replace("{questions_str}", question_str).replace("{output_format_str}", output_str)
                if collect_type=="pair":
                    template = mtemplate.replace("{response_1}", response_1).replace("{response_2}", response_2)
                    tags.append(f"{current_sample['id']}_query-aware-features-pair")
                    all_query_aware_twice_instances.append({"usermsg": template})
                else:
                    template = mtemplate.replace("{response}", response_1)
                    tags.append(f"{current_sample['id']}_query-aware-features-1")
                    all_query_aware_twice_instances.append({"usermsg": template})
                    template = mtemplate.replace("{response}", response_2)
                    tags.append(f"{current_sample['id']}_query-aware-features-2")
                    all_query_aware_twice_instances.append({"usermsg": template})

            query_aware_twice_features = engine.generate_batch(all_query_aware_twice_instances)
            step2_cost = sum([float(query_aware_twice_feature["cost"]) for query_aware_twice_feature in query_aware_twice_features])
            total_cost += step2_cost

            assert len(query_aware_twice_features) == len(all_query_aware_twice_instances)
            for idx, query_aware_twice_feature in enumerate(query_aware_twice_features):
                query_aware_twice_feature["tag"] = tags[idx]
                if query_aware_twice_feature.get("finish_reason", "") not in ["stop","length",""]:
                    query_aware_twice_feature["usermsg"] = all_query_aware_twice_instances[idx]["usermsg"]

            final_query_aware_twice_features = []
            if collect_type=="single":

                pairwised = []
                assert len(query_aware_twice_features) == len(all_question_output_strs) * 2
                for j in range(len(query_aware_twice_features) // 2):
                    pairwised.append({
                        "response 1": query_aware_twice_features[2 * j],
                        "response 2": query_aware_twice_features[2 * j + 1]
                    })
            else:
                pairwised = query_aware_twice_features
            assert len(pairwised) == len(all_question_output_strs)

            # step 2.2 add {} to those without query info
            for i in range(len(samples[startid:endid])):
                tagx = f"{samples[startid:endid][i]['id']}_query-aware-twice-features-{collect_type}"
                if i not in mapping: final_query_aware_twice_features.append({"tag":tagx})
                else:
                    tt = copy.deepcopy(pairwised[mapping[i]])
                    tt["tag"] = tagx
                    final_query_aware_twice_features.append(tt)

            write_jsonl(final_query_aware_twice_features, f"./annotation_results/query-aware-twice-features-by-{collect_type}.jsonl", mode="a")
            print_colored_text(f"\nQuery-aware-twice features collection done! - {batch_tag}, cost: {step2_cost}", color="green")

        # raise ValueError

        if step3:
            ################################################################
            # step 3, get query-aware-once features
            ################################################################

            rtemplate = query_aware_once_pairwise["template"] if collect_type == "pair" else query_aware_once_single["template"]
            all_query_aware_once_instances = []
            for sample in samples[startid:endid]:
                prompt, response_1, response_2 = sample["prompt"], sample["response_a"], sample["response_b"]
                if collect_type == "pair":
                    template = rtemplate.replace("{prompt}", prompt).replace("{response_1}", response_1).replace(
                        "{response_2}", response_2)
                    all_query_aware_once_instances.append({"usermsg": template})
                else:
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_1)
                    all_query_aware_once_instances.append({"usermsg": template})
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_2)
                    all_query_aware_once_instances.append({"usermsg": template})

            query_aware_once_features = engine.generate_batch(all_query_aware_once_instances)
            step3_cost = sum([float(query_aware_once_feature["cost"]) for query_aware_once_feature in query_aware_once_features])
            total_cost += step3_cost


            assert len(query_aware_once_features) == len(all_query_aware_once_instances)
            for idx, query_aware_once_feature in enumerate(query_aware_once_features):
                if query_aware_once_feature.get("finish_reason", "") not in ["stop","length",""]:
                    query_aware_once_feature["usermsg"] = all_query_aware_once_instances[idx]["usermsg"]

            final_query_aware_once_features = []
            if collect_type == "single":
                assert len(query_aware_once_features) == len(samples[startid:endid]) * 2
                for j in range(len(query_aware_once_features) // 2):
                    final_query_aware_once_features.append({
                        "response 1": query_aware_once_features[2 * j],
                        "response 2": query_aware_once_features[2 * j + 1]
                    })
            else:
                final_query_aware_once_features = query_aware_once_features
            assert len(final_query_aware_once_features) == len(samples[startid:endid])
            for idx, xx in enumerate(final_query_aware_once_features):
                xx["tag"] = f"{samples[startid:endid][idx]['id']}_query-aware-once-features-{collect_type}"

            write_jsonl(final_query_aware_once_features,
                        f"./annotation_results/query-aware-once-features-by-{collect_type}.jsonl", mode="a")
            print_colored_text(f"\nQuery-aware-once features collection done!- {batch_tag}, cost: {step3_cost}", color="green")

        if step4:
            ################################################################
            # step 4, get query-free features
            ################################################################

            rtemplate = query_free_pairwise["template"] if collect_type == "pair" else query_free_single["template"]
            all_query_free_instances = []
            for sample in samples[startid:endid]:
                prompt,response_1,response_2 = sample["prompt"],sample["response_a"],sample["response_b"]
                if collect_type=="pair":
                    template = rtemplate.replace("{prompt}", prompt).replace("{response_1}", response_1).replace("{response_2}", response_2)
                    all_query_free_instances.append({"usermsg": template})
                else:
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_1)
                    all_query_free_instances.append({"usermsg": template})
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_2)
                    all_query_free_instances.append({"usermsg": template})

            query_free_features = engine.generate_batch(all_query_free_instances)
            step4_cost = sum([float(query_free_feature["cost"]) for query_free_feature in query_free_features])
            total_cost += step4_cost

            assert len(query_free_features) == len(all_query_free_instances)
            for idx, query_free_feature in enumerate(query_free_features):
                if query_free_feature.get("finish_reason", "") not in ["stop","length",""]:
                    query_free_feature["usermsg"] = all_query_free_instances[idx]["usermsg"]

            final_query_free_features = []
            if collect_type=="single":
                assert len(query_free_features) == len(samples[startid:endid]) * 2
                for j in range(len(query_free_features) // 2):
                    final_query_free_features.append({
                        "response 1": query_free_features[2 * j],
                        "response 2": query_free_features[2 * j + 1]
                    })
            else:
                final_query_free_features = query_free_features
            assert len(final_query_free_features) == len(samples[startid:endid])
            for idx, query_free_feature in enumerate(final_query_free_features):
                query_free_feature["tag"] = f"{samples[startid:endid][idx]['id']}_query-free-features-{collect_type}"

            write_jsonl(final_query_free_features, f"./annotation_results/query-free-features-by-{collect_type}.jsonl", mode="a")
            print_colored_text(f"\nQuery-free features collection done!- {batch_tag}, cost: {step4_cost}", color="green")

        if step5:
            ################################################################
            # step 5, get accuracy
            ################################################################

            rtemplate = accuracy_check_single["template"] if collect_type == "single" else accuracy_check_pairwise["template"]
            all_accuracy_instances = []
            for sid,sample in enumerate(samples[startid:endid]):
                prompt,response_1,response_2,reference = sample["prompt"],sample["response_a"],sample["response_b"],gpt4_refs[startid+sid]["output"]
                if collect_type == "pair":
                    template = rtemplate.replace("{prompt}", prompt).replace("{response_1}", response_1).replace(
                        "{response_2}", response_2).replace("{reference}", reference)
                    all_accuracy_instances.append({"usermsg": template})
                else:
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_1).replace(
                        "{reference}", reference)
                    all_accuracy_instances.append({"usermsg": template})
                    template = rtemplate.replace("{prompt}", prompt).replace("{response}", response_2).replace(
                        "{reference}", reference)
                    all_accuracy_instances.append({"usermsg": template})


            accuracy_features = engine.generate_batch(all_accuracy_instances)
            step6_cost = sum([float(accuracy_feature["cost"]) for accuracy_feature in accuracy_features])
            total_cost += step6_cost

            assert len(accuracy_features) == len(all_accuracy_instances)
            for idx, accuracy_feature in enumerate(accuracy_features):
                if accuracy_feature.get("finish_reason", "") not in ["stop","length",""]:
                    accuracy_feature["usermsg"] = all_accuracy_instances[idx]["usermsg"]

            final_accuracy = []
            if collect_type == "single":
                assert len(accuracy_features) == len(all_accuracy_instances) == len(samples[startid:endid]) * 2
                for j in range(len(accuracy_features) // 2):
                    final_accuracy.append({
                        "response 1": accuracy_features[2 * j],
                        "response 2": accuracy_features[2 * j + 1],
                    })
            else:
                final_accuracy = accuracy_features
            assert len(final_accuracy) == len(samples[startid:endid])
            for idx, xx in enumerate(final_accuracy):
                xx["tag"] = f"{samples[startid:endid][idx]['id']}_accuracy-features-{collect_type}"

            write_jsonl(final_accuracy, f"./annotation_results/accuracy-features.jsonl", mode="a")
            print_colored_text(f"\nAccuracy collection done!- {batch_tag}, cost: {step6_cost}", color="green")

        return total_cost


    total_cost = 0.0
    start=0 # in
    end=2 # out
    batchsize=10
    ctype="pair"
    total_batches = (end - start) // batchsize
    total_batches += 1 if (end - start) % batchsize != 0 else 0

    for batch_idx in range(total_batches):
        batch_start = start + batch_idx * batchsize
        batch_end = min(end, batch_start + batchsize)
        one_batch_cost = process_a_batch(batch_start,batch_end, batch_tag="", collect_type=ctype,step1=True, step2=True, step3=True, step4=True, step5=True)
        total_cost += one_batch_cost
        print_colored_text(f"\nBatch {batch_idx + 1}/{total_batches} done! - [{batch_start}, {batch_end}), batch cost: {one_batch_cost}, accumulative cost: {total_cost}", color="cyan")









