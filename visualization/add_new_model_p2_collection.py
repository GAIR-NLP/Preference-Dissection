import os

from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="Test-Model")
    parser.add_argument("--model_size",type=float,default=6.5,help="the size of the target model")
    args = parser.parse_args()

    model_name = args.model_name
    model_size = args.model_size

    dir = f"./data/added_model_inference/{model_size}_{model_name}/"
    filelist = os.listdir(dir)

    # you need to have the two inference results to proceed
    assert "direct_ask_preferences_AB.jsonl" in filelist
    assert "direct_ask_preferences_BA.jsonl" in filelist

    AB_results = read_all(dir+"direct_ask_preferences_AB.jsonl")
    BA_results = read_all(dir+"direct_ask_preferences_BA.jsonl")
    pair_len = len(read_all("./data/chatbot_arena_shuffled_no-tie_group_balanced.jsonl"))
    assert len(AB_results) == len(BA_results) == pair_len

    ss = []
    for ar,br in zip(AB_results,BA_results):
        a_prob = ar['response_a']+br['response_b']
        b_prob = ar['response_b']+br['response_a']
        if a_prob > b_prob:
            ss.append('A')
        else:
            ss.append('B')
    write_json(ss,dir+"direct_ask_preferences.json")