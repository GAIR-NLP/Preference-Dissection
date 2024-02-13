"""
This file is to resolve the annotated data, from string to json object.

"""
from utils import *
from api_utils import *


def resolve_xx(str, tag):
    try:
        return json.loads(str)
    except:
        print_colored_text(f"Error in {tag}, we print the original one", color="red")
        print_colored_text('====================================', color="purple")
        print(str)
        print_colored_text('====================================', color="purple")
        return None


def default_comparison(score1, score2):
    if score1 == score2:
        return 0
    elif score1 > score2:
        return 1
    else:
        return -1


def zero_vs_nonzero_comparison(score1, score2):
    if score1 > 0 and score2 == 0:
        return 1
    elif score1 == 0 and score2 > 0:
        return -1
    else:
        return 0


def comparison(score1, score2, feature_name):
    # for some feature we need to do some special process
    # in v1.4, no feature needs special process
    if feature_name in ["relevance without considering inaccuracy"]:
        return score1 - score2, zero_vs_nonzero_comparison(score1, score2)
    else:
        return score1 - score2, default_comparison(score1, score2)


def get_comparison_acc(resolved_acc):
    # first we build meta, which is easy

    # elegant_show(resolved_acc)
    # raise ValueError
    resolved_acc_r1 = resolved_acc["Response 1"]
    resolved_acc_r2 = resolved_acc["Response 2"]

    meta = {"Response 1": resolved_acc_r1, "Response 2": resolved_acc_r2}

    # then we build comparison
    simplified_errors_r1 = {"Severe": 0, "Moderate": 0, "Minor": 0}
    simplified_errors_r2 = {"Severe": 0, "Moderate": 0, "Minor": 0}

    # check reliability of results
    if resolved_acc_r1['accuracy check'] != "applicable" or resolved_acc_r2['accuracy check'] != "applicable":
        # print_colored_text("Accuracy check is not applicable for at least one of the responses.", color="red")
        return {},{}, 999, meta

    for item in resolved_acc_r1["inaccuracies"]:
        if item["severity"].lower().strip() == "severe":
            simplified_errors_r1["Severe"] += 1
        elif item["severity"].lower().strip() == "moderate":
            simplified_errors_r1["Moderate"] += 1
        elif item["severity"].lower().strip() == "minor":
            simplified_errors_r1["Minor"] += 1

    for item in resolved_acc_r2["inaccuracies"]:
        if item["severity"].lower().strip() == "severe":
            simplified_errors_r2["Severe"] += 1
        elif item["severity"].lower().strip() == "moderate":
            simplified_errors_r2["Moderate"] += 1
        elif item["severity"].lower().strip() == "minor":
            simplified_errors_r2["Minor"] += 1

    acc_diff = {"Severe": simplified_errors_r2["Severe"] - simplified_errors_r1["Severe"],
                "Moderate": simplified_errors_r2["Moderate"] - simplified_errors_r1["Moderate"],
                "Minor": simplified_errors_r2["Minor"] - simplified_errors_r1["Minor"]}

    acc_norm_diff = {"Severe": norm_a_diff(acc_diff["Severe"], 5),
                     "Moderate": norm_a_diff(acc_diff["Moderate"], 5),
                     "Minor": norm_a_diff(acc_diff["Minor"], 5)}

    # compare these two with the following rule
    # if one has more severe errors, then it is worse
    # if the number of severe errors are the same, then we compare moderate errors
    # and so on so forth

    def small_map(v):
        if v == 0: return 0
        if v > 0:
            return 1
        else:
            return -1

    acc_comp = {k: small_map(v) for k, v in acc_diff.items()}

    return acc_norm_diff, acc_diff, acc_comp, meta


def get_comparison_query_aware_twice(query_aware_twice_features_by_pair, take_mean=True):
    xx = {
        "clarify user intent": {"comparison": 0, "diff": 0, "norm_diff": 0, "meta": None},
        "showing empathetic": {"comparison": 0, "diff": 0, "norm_diff": 0, "meta": None},
        "satisfying explicit constraints": {"comparison": 0, "diff": 0, "norm_diff": 0, "meta": None},
        "supporting explicit subjective stances": {"comparison": 0, "diff": 0, "norm_diff": 0, "meta": None},
        "correcting explicit mistakes or biases": {"comparison": 0, "diff": 0, "norm_diff": 0, "meta": None},
    }

    if "satisfying explicit constraints" in query_aware_twice_features_by_pair:
        score1, score2 = 0.0, 0.0
        for k, v in query_aware_twice_features_by_pair["satisfying explicit constraints"].items():
            score1 += get_score(v["Response 1"])
            score2 += get_score(v["Response 2"])
        if take_mean:
            score1 /= len(query_aware_twice_features_by_pair["satisfying explicit constraints"])
            score2 /= len(query_aware_twice_features_by_pair["satisfying explicit constraints"])
        xx["satisfying explicit constraints"]["diff"], xx["satisfying explicit constraints"]["comparison"] = comparison(
            score1, score2, "satisfying explicit constraints")
        xx["satisfying explicit constraints"]["norm_diff"] = norm_a_diff(xx["satisfying explicit constraints"]["diff"],
                                                                         3)
        xx["satisfying explicit constraints"]["meta"] = query_aware_twice_features_by_pair[
            "satisfying explicit constraints"]

    if "clarify user intent" in query_aware_twice_features_by_pair:
        score1, score2 = get_score(query_aware_twice_features_by_pair["clarify user intent"]["Response 1"]), get_score(
            query_aware_twice_features_by_pair["clarify user intent"]["Response 2"])
        xx["clarify user intent"]["diff"], xx["clarify user intent"]["comparison"] = comparison(score1, score2,
                                                                                                "clarify user intent")
        xx["clarify user intent"]["norm_diff"] = norm_a_diff(xx["clarify user intent"]["diff"], 3)
        xx["clarify user intent"]["meta"] = {"Response 1": score1, "Response 2": score2}

    if "showing empathetic" in query_aware_twice_features_by_pair:
        # elegant_show(query_aware_twice_features_by_pair["showing empathetic"])
        score1, score2 = get_score(query_aware_twice_features_by_pair["showing empathetic"]["Response 1"]), get_score(
            query_aware_twice_features_by_pair["showing empathetic"]["Response 2"])
        xx["showing empathetic"]["diff"], xx["showing empathetic"]["comparison"] = comparison(score1, score2,
                                                                                              "showing empathetic")
        xx["showing empathetic"]["norm_diff"] = norm_a_diff(xx["showing empathetic"]["diff"], 3)
        xx["showing empathetic"]["meta"] = {"Response 1": score1, "Response 2": score2}

    if "supporting explicit subjective stances" in query_aware_twice_features_by_pair:
        mapping = {"strongly supported": 2, "weakly supported": 1, "neutral": 0, "weakly opposed": -1,
                   "strongly opposed": -2}
        score1, score2 = 0.0, 0.0
        for k, v in query_aware_twice_features_by_pair["supporting explicit subjective stances"].items():
            score1 += mapping.get(v["Response 1"].lower().strip(), 0)
            score2 += mapping.get(v["Response 2"].lower().strip(), 0)
        if take_mean:
            score1 /= len(query_aware_twice_features_by_pair["supporting explicit subjective stances"])
            score2 /= len(query_aware_twice_features_by_pair["supporting explicit subjective stances"])
        xx["supporting explicit subjective stances"]["diff"], xx["supporting explicit subjective stances"][
            "comparison"] = comparison(score1, score2, "supporting explicit subjective stances")
        xx["supporting explicit subjective stances"]["norm_diff"] = norm_a_diff(
            xx["supporting explicit subjective stances"]["diff"], 3)
        xx["supporting explicit subjective stances"]["meta"] = query_aware_twice_features_by_pair[
            "supporting explicit subjective stances"]

    if "correcting explicit mistakes or biases" in query_aware_twice_features_by_pair:
        # elegant_show(query_aware_twice_features_by_pair["correcting mistakes or biases"])
        mapping = {"pointed out and corrected": 3, "corrected without being pointed out": 2,
                   "pointed out but not corrected": 1, "neither pointed out nor corrected": 0}
        score1, score2 = 0.0, 0.0
        for k, v in query_aware_twice_features_by_pair["correcting explicit mistakes or biases"].items():
            score1 += mapping.get(v["Response 1"].lower().strip(), 0)
            score2 += mapping.get(v["Response 2"].lower().strip(), 0)
        if take_mean:
            score1 /= len(query_aware_twice_features_by_pair["correcting explicit mistakes or biases"])
            score2 /= len(query_aware_twice_features_by_pair["correcting explicit mistakes or biases"])
        xx["correcting explicit mistakes or biases"]["diff"], xx["correcting explicit mistakes or biases"][
            "comparison"] = comparison(score1, score2, "correcting explicit mistakes or biases")
        xx["correcting explicit mistakes or biases"]["norm_diff"] = norm_a_diff(
            xx["correcting explicit mistakes or biases"]["diff"], 3)
        xx["correcting explicit mistakes or biases"]["meta"] = query_aware_twice_features_by_pair[
            "correcting explicit mistakes or biases"]

    return xx


def get_comparison_length(source):
    word1 = source["response_a word"]  # an integer
    word2 = source["response_b word"]  # an integer
    lendiff = word1 - word2
    meta = {"Response 1": word1, "Response 2": word2}
    # General case: if shorter one is shorter than 70% of the other, and the difference is more than 10, then it is worse

    flipped = False
    if word1 > word2:
        word1, word2 = word2, word1
        flipped = True
    # now word1 is the shorter one

    label = 0

    if word1 <= 11:
        if word1 == 0 and word2 != 0: label = -1
        if word1 == 1 and word2 >= 3: label = -1
        if word1 == 2 and word2 >= 5: label = -1
        if word1 == 3 and word2 >= 7: label = -1
        if word1 == 4 and word2 >= 8: label = -1
        if word1 == 5 and word2 >= 10: label = -1
        if word1 == 6 and word2 >= 11: label = -1
        if word1 == 7 and word2 >= 13: label = -1
        if word1 == 8 and word2 >= 14: label = -1
        if word1 == 9 and word2 >= 16: label = -1
        if word1 == 10 and word2 >= 17: label = -1
        if word1 == 11 and word2 >= 19: label = -1
    elif word1 <= 0.7 * word2 and word2 - word1 >= 10:
        label = -1
    elif word1 <= 0.8 * word2 and word2 - word1 >= 20:
        label = -1
    elif word1 <= 0.9 * word2 and word2 - word1 >= 40:
        label = -1

    label = -label if flipped else label

    norm_diff = norm_a_diff(lendiff, 500,way="log_linear")

    return norm_diff, lendiff, label, meta


if __name__ == '__main__':
    show_raw = False
    source = read_all("./raw_data/sample_unannotated.jsonl")
    collected_data_dir = f"./annotation_results"

    all_info_list = []
    all_query_info_list = {"all": [], "express_feeling": [],
                           "intent_unclear": [], "w_constraints": [],
                           "w_mistakes": [], "w_stances": []}

    query_info_file = f"{collected_data_dir}/query_info.jsonl"
    query_infos = read_all(query_info_file)

    query_aware_twice_features_by_pair_file = f"{collected_data_dir}/query-aware-twice-features-by-pair.jsonl"
    query_aware_twice_features_by_pair = read_all(query_aware_twice_features_by_pair_file)

    query_aware_once_features_by_pair_file = f"{collected_data_dir}/query-aware-once-features-by-pair.jsonl"
    query_aware_once_features_by_pair = read_all(query_aware_once_features_by_pair_file)

    query_free_features_by_pair_file = f"{collected_data_dir}/query-free-features-by-pair.jsonl"
    query_free_features_by_pair = read_all(query_free_features_by_pair_file)

    accuracy_features = f"{collected_data_dir}/accuracy-features.jsonl"
    accuracy_features = read_all(accuracy_features)

    new_output = []

    valid = 0

    for idx, (s, qinfo, q0f, q1f, q2f, acc) in enumerate(zip(source, query_infos, query_free_features_by_pair,
                                                             query_aware_once_features_by_pair,
                                                             query_aware_twice_features_by_pair,
                                                             accuracy_features)):
        r_qinfo = resolve_xx(qinfo["output"], f"query_info-{idx}")
        r_q0f = resolve_xx(q0f["output"], f"query_free-{idx}")
        r_q1f = resolve_xx(q1f["output"], f"query_aware_once-{idx}")
        r_q2f = resolve_xx(q2f.get("output", "{}"), f"query_aware_twice-{idx}")
        r_acc = resolve_xx(acc["output"], f"accuracy-{idx}")

        if any([x is None for x in [r_qinfo, r_q0f, r_q1f, r_q2f, r_acc]]):
            # some error in this sample, we just skip it
            continue

        valid += 1

        if show_raw:
            pass

        simplified_comparison_results = {}

        # add query free features & comparison
        for k, v in r_q0f.items():
            score_1, score_2 = get_score(v['response 1']), get_score(v['response 2'])
            difference, comparison_result = comparison(score_1, score_2, k)
            simplified_comparison_results[k] = {"comparison": comparison_result,
                                                "diff": difference,
                                                "norm_diff": norm_a_diff(difference, 3),
                                                "meta": {"Response 1": score_1, "Response 2": score_2}}

        # add query aware once features & comparison
        for k, v in r_q1f.items():
            score_1, score_2 = get_score(v['response 1']), get_score(v['response 2'])
            difference, comparison_result = comparison(score_1, score_2, k)
            simplified_comparison_results[k] = {"comparison": comparison_result,
                                                "diff": difference,
                                                "norm_diff": norm_a_diff(difference, 3),
                                                "meta": {"Response 1": score_1, "Response 2": score_2}}

        # add accuracy features & comparison
        acc_norm_diff, acc_diff, acc_comp, acc_meta = get_comparison_acc(r_acc)
        if acc_comp == 999:
            print_colored_text(
                f"idx - {idx} : Accuracy check is not applicable for at least one of the responses. "
                f"- GPT-4 cannot do accuracy comparison, we skip this sample",
                color="red")
        simplified_comparison_results["accuracy"] = {"comparison": acc_comp,
                                                     "diff": acc_diff,
                                                     "norm_diff": acc_norm_diff,
                                                     "meta": acc_meta}

        # add query aware twice features & comparison
        q2f = get_comparison_query_aware_twice(r_q2f)
        for k, v in q2f.items():
            simplified_comparison_results[k] = v

        # get length features & comparison
        norm_length_diff, length_diff, length_comp, length_meta = get_comparison_length(s)
        simplified_comparison_results["length"] = {"comparison": length_comp, "diff": length_diff,
                                                   "norm_diff": norm_length_diff, "meta": length_meta}

        # combine all things info one sample
        sample = {"query_info": r_qinfo, "comparison": simplified_comparison_results}

        all_query_info_list["all"].append(idx)
        if r_qinfo['clear intent'] == "No": all_query_info_list["intent_unclear"].append(idx)
        if r_qinfo['explicitly express feelings'] == "Yes": all_query_info_list["express_feeling"].append(idx)
        if len(r_qinfo['explicit constraints']) > 0: all_query_info_list["w_constraints"].append(idx)
        if len(r_qinfo['explicit subjective stances']) > 0: all_query_info_list["w_stances"].append(idx)
        if len(r_qinfo['explicit mistakes or biases']) > 0: all_query_info_list["w_mistakes"].append(idx)
        group = s["group"]
        if group not in all_query_info_list: all_query_info_list[group] = []
        all_query_info_list[group].append(idx)

        new_output.append(sample)

    write_jsonl(new_output, f"./resolved_annotations/annotated.jsonl")
    write_json(all_query_info_list, f"./resolved_annotations/query_aware_idxs.json")
