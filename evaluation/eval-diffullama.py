import torch
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from attention_patch import replace_attention_mask

replace_attention_mask()
from tqdm import tqdm
from llamafactory.train.ddm.trainer import eval_forward, generate_samples, generate_samples_v2
from model import DiscreteDiffusionModel
from argparse import ArgumentParser

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

import torch.distributions as dists
import torch.nn.functional as F
from f1 import compute_f1, normalize_answer
from evaluation.ifeval import test_instruction_following_strict, test_instruction_following_loose, load_ifeval_prompts
from datasets import load_dataset
from evaluation.codex_evaluation import evaluate_functional_correctness, write_jsonl

# These examplars are from the Table 20 of CoT paper (https://arxiv.org/pdf/2201.11903.pdf).
GSM_EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8",
    },
]


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)
    
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    # import pdb; pdb.set_trace();
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def eval_Lambada(model, tokenizer, args):
    total_cnt = 0
    cor = 0
    with open('evaluation/lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
        for line in file:
            total_cnt += 1
            line = line.strip()
            # import pdb; pdb.set_trace();
            x0 = tokenizer.encode(line)
            prefix = tokenizer.encode(' '.join(line.split()[:-1]))
            # attention_mask = get_anneal_attn_mask(len(xt), 1, dtype=model.lm_head.weight.dtype, device=model.device, attn_mask_ratio=1.0)
            # masked_nums = len(xt)-len(inputs)
            # xt[-masked_nums:] = [tokenizer.mask_token_id] * masked_nums
            # xt = torch.tensor([xt]).to(model.device)
            # logits = model(xt, attention_mask=attention_mask)
            # filter_logits = top_p_logits(logits/0.8, p=0.8)
            # scores = torch.log_softmax(filter_logits, dim=-1)
            # # x0_scores, x0 = scores.max(-1)
            # x0 = dists.Categorical(logits=scores).sample()
            # pred = tokenizer.decode(x0.tolist()[0][-masked_nums-1:-1])

            masked_nums = len(x0)-len(prefix)
            src_mask = [1]*len(prefix)+[0]*masked_nums
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            args.diffusion_steps = masked_nums
            args.logits_temp = 1.0
            res = generate_samples(model, args, tokenizer, inputs, eval=True)
            pred = tokenizer.decode(res.tolist()[0][-masked_nums:])
            # import pdb; pdb.set_trace();
            if pred.strip() == line.split()[-1].strip():
                cor += 1
            # print(total_cnt, cor/total_cnt)

            if pred.strip() == line.split()[-1].strip():
                cor += 1
    print('acc:', cor/total_cnt)

import re
import numpy as np

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def eval_hellaswag(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split='validation')

    total_cnt = 0
    cor = 0

    for doc in tqdm(ds):
        total_cnt += 1
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])

        score_list = []
        prefix = tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  


def eval_wino(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code=True)

    total_cnt = 0
    cor = 0

    for doc in tqdm(ds):
        total_cnt += 1
        
        idx = doc["sentence"].index("_")
        
        options = [doc["option1"], doc["option2"]]

        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]

        score_list = []
        
        for opt in options:
            target = opt 
            suffix = doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target, add_special_tokens=False)
            suffix_id = tokenizer.encode(suffix, add_special_tokens=False)
            prefix = doc["sentence"][:idx]
            prefix_id = tokenizer.encode(prefix, add_special_tokens=False)

            x0 = prefix_id + target_id + suffix_id
            src_mask = [1]*len(prefix_id)+[0]*(len(x0)-len(prefix_id))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  

def eval_piqa(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split='validation', trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in tqdm(ds):
        total_cnt += 1
        
        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        gold = doc["label"]

        score_list = []
        prefix = tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(" " + choice)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)  

def eval_siqa(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("allenai/social_i_qa", split='validation', trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in tqdm(ds):
        total_cnt += 1
        
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc["label"]) - 1

        score_list = []
        prefix = tokenizer.encode(query, add_special_tokens=False)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice, add_special_tokens=False)
            src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix))
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
            score = eval_forward(model, inputs, args, tokenizer)
            # import pdb; pdb.set_trace();
            score_list.append(score.tolist())
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        
    print('acc:', cor/total_cnt)

import csv, json
import evaluate

def eval_infilling(model, tokenizer, args):
    problems = []
    with open(f"evaluation/cloze_test_val__spring2016.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
            # sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
            problems.append(sents)
    
    samples = []
    total_cnt = 0
    gens = []
    refs = []

    for stories in problems:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        prompt = stories[0] + " " + stories[1]
        suffix = stories[3] + " " + stories[4]
        middle = stories[2]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle, add_special_tokens=False) + suff
        src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
        samples.append(dict(pred=pred, label=middle, prefix=prompt, suffix=suffix))
        gens.append(pred)
        refs.append(middle)

        if total_cnt == 1000:
            break

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")


    with open(f'ROCInfill_medium_t{args.diffusion_steps}_tmp{args.logits_temp}.jsonl', 'w') as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + '\n')

def humaneval_infill(model, tokenizer, args):
    from human_eval_infilling.data import write_jsonl, read_problems

    subtasks = "single-line"
    problems = read_problems(benchmark_name=subtasks)
    samples = []
    for task_id in problems:
        # import pdb; pdb.set_trace();
        prompt = problems[task_id]["prompt"]
        suffix = problems[task_id]["suffix"]
        middle = problems[task_id]["canonical_solution"]

        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suff = tokenizer.encode(suffix, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle, add_special_tokens=False) + suff
        src_mask = [1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)
        if len(x0) > 1000:
            print(task_id)
            continue
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
        samples.append(dict(task_id=task_id, completion=pred))

    write_jsonl(f"humaneval_medium_samplingv1_{subtasks}.jsonl", samples)
    
def eval_triva(model, tokenizer, args):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    # ds = load_dataset("rajpurkar/squad", split='validation')
    gens = []
    refs = []
    total_cnt = 0
    cor = 0
    triviaqa_shots = [
      "Which American-born Sinclair won the Nobel Prize for Literature in 1930?\n\n(Harry) Sinclair Lewis",
      "Where in England was Dame Judi Dench born?\n\nYork, England",
    ]
    for doc in tqdm(ds):
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query =  "\n".join(triviaqa_shots) + "\n\n" + doc["question"] #f"Quesion{doc['question']}?\nAnswer: "
        query = "<|user|>\n" + query.strip() + "\n<|assistant|>\n"
        labels = doc["answer"]["aliases"]
        encoded_labels = [tokenizer.encode(l, add_special_tokens=False) for l in labels]
        long_gold = max(encoded_labels, key=len)

        input_ids = tokenizer.encode(query)
        full = long_gold
        tokens = len(full)
        
        x0 = input_ids + [0]*(tokens)
        src_mask = [1]*len(input_ids)+[0]*(tokens)
        args.diffusion_steps = tokens

        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        

        for l in labels:
            if normalize_answer(l) in normalize_answer(pred.strip()):
                cor += 1
                break
                
        gens.append(pred)
        refs.append(labels)

        if total_cnt == 2000:
            break

        print(pred, labels)

    print('em acc:', cor/total_cnt)
    print(compute_f1(gens, refs))

def eval_squad(model, tokenizer, args):
    from datasets import load_dataset
    from tqdm import tqdm
    from squad_eval_1 import evaluate as squad_evaluate
    dataset = load_dataset("squad", split="validation")
    # only 512 samples
    dataset = dataset.shuffle(42).select(range(512))
    squad_shots = [
        "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.\n\nTo whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\n\nSaint Bernadette Soubirous",
        "Burke was born in Dublin, Ireland. His mother Mary née Nagle (c. 1702 – 1770) was a Roman Catholic who hailed from a déclassé County Cork family (and a cousin of Nano Nagle), whereas his father, a successful solicitor, Richard (died 1761), was a member of the Church of Ireland; it remains unclear whether this is the same Richard Burke who converted from Catholicism. The Burke dynasty descends from an Anglo-Norman knight surnamed de Burgh (latinised as de Burgo) who arrived in Ireland in 1185 following Henry II of England's 1171 invasion of Ireland.\n\nWhere was Burke born?\n\nDublin, Ireland",
        "The term high definition once described a series of television systems originating from August 1936; however, these systems were only high definition when compared to earlier systems that were based on mechanical systems with as few as 30 lines of resolution. The ongoing competition between companies and nations to create true \"HDTV\" spanned the entire 20th century, as each new system became more HD than the last.In the beginning of the 21st century, this race has continued with 4k, 5k and current 8K systems.\n\nThe term \"high definition\" originally described televisions systems from what year?\n\n1936"
    ]
    data = [sample for sample in dataset]
    preds = []
    for sample in tqdm(data):
        query = "\n".join(squad_shots) + '\n' + sample["context"] + "\n\n" + sample["question"]
        input_ids = tokenizer.encode(query)
        remaining_len = 2048 - len(input_ids)
        x0 = input_ids + [0]*remaining_len
        src_mask = [1]*len(input_ids) + [0]*remaining_len
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        pred = pred.strip()
        print(pred)
        # if we see eos token, truncate
        if "</s>" in pred:
            pred = pred[:pred.index("</s>")]
        if "\n" in pred:
            pred = pred[:pred.index("\n")]
        print('xxx', pred)
        preds.append(pred)
    predictions = [{"id": y['id'], "prediction_text": x} for x, y in zip(preds, data) if y is not None]
    references = [{"id": x["id"], "answers": x["answers"]}  for x in data if x is not None]
    # now calculate the metrics
    results = squad_evaluate(references=references, predictions=predictions)
    print(results)

def eval_alpaca(model, tokenizer, args):
    from alpaca_eval.main import evaluate as alpaca_farm_evaluate
    from datasets import load_dataset
    from tqdm import tqdm
    data = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    gens = []
    data = [sample for sample in data]
    for sample in tqdm(data):
        query = sample["instruction"] + "\nResponse: "
        print(query)
        input_ids = tokenizer.encode(query)
        remaining_len = 2048 - len(input_ids)
        x0 = input_ids + [0]*remaining_len
        src_mask = [1]*len(input_ids) + [0]*remaining_len
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        print(pred)
        # if we see eos token, truncate
        if "</s>" in pred:
            pred = pred[:pred.index("</s>")]
        print(pred)
        gens.append({
            "output": pred,
            "generator": "diffullama",
            "instruction": sample["instruction"],
            "dataset": sample["dataset"],
        })
    df_leaderboard, _ = alpaca_farm_evaluate(
        model_outputs=gens,
        annotators_config="alpaca_eval_gpt4",
        output_path="tmp",
        is_return_instead_of_print=True,
        is_overwrite_leaderboard=True,
    )

    print(df_leaderboard.to_string(float_format="%.2f"))
    results_json = {"win_rate": df_leaderboard.to_dict()["win_rate"]["diffullama"]}
    print(results_json)
    with open(args.output_file, "w") as f:
        json.dump(gens, f)


def eval_gsm8k(model, tokenizer, args):
    from datasets import load_dataset
    exact_match = evaluate.load("exact_match")
    from tqdm import tqdm
    gsm = load_dataset("openai/gsm8k", "main", split='test')
    global GSM_EXAMPLARS
    demonstrations = []
    for example in GSM_EXAMPLARS:
        demonstrations.append("Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"])
    prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
    final_preds = []
    answers = []
    count = 0
    # split the gsm data into 6 chunks, and take args.shard_num-th chunk
    gsm = [x for x in gsm]
    outputs = []
    for sample in tqdm(gsm):
        query = prompt_prefix + sample["question"]
        input_ids = tokenizer.encode(query)
        remaining_len = 2048 - len(input_ids)
        x0 = input_ids + [0]*remaining_len
        src_mask = [1]*len(input_ids) + [0]*remaining_len
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        with torch.no_grad():
            res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        pred = pred.strip()
        if 'question' in pred.lower():
            pred = pred[:pred.lower().index("question")]
        print(pred)
        outputs.append({
            "output": pred,
            "generator": "diffullama",
            "question": sample["question"],
        })
        sample_answer = sample["answer"].split("###")[-1].strip()
        # replace numbers like `x,xxx` with `xxxx`
        try:
          pred = re.sub(r"(\d),(\d)", r"\1\2", pred)
          numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
          if numbers:
            final_pred = numbers[-1]
          else:
            final_pred = pred
        except:
            final_pred = pred
        final_preds.append(final_pred)
        answers.append(sample_answer)
        count += 1
    em_score = exact_match.compute(
        predictions=final_preds, references=answers, ignore_case=True, ignore_punctuation=True
    )["exact_match"]
    print(f"Exact match : {em_score}")
    print(f"Total count : {count}")
    with open(args.output_file, "w") as f:
        json.dump(outputs, f)

def eval_bbh(model, tokenizer, args):
    """Evaluate model on Big Bench Hard tasks."""
    import os
    import json
    import glob
    import random
    from tqdm import tqdm
    import evaluate

    random.seed(42)
    exact_match = evaluate.load("exact_match")

    # Load all tasks and prompts
    all_tasks = {}
    task_files = glob.glob(os.path.join("evaluation/bbh/bbh", "*.json"))
    for task_file in tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join("evaluation/bbh/cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            all_prompts[task_name] = task_prompt

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "Task names mismatch between data and prompts"

    # Create output directories
    os.makedirs("results/bbh", exist_ok=True)
    os.makedirs("results/bbh/predictions", exist_ok=True)

    performance = {}
    for task_name in tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]

        # Prepare prompts
        prompts = ["<|user|>\n" + task_prompt.strip() + "\n\nQ: " + example["input"] + "\n<|assistant|>\nA:" for example in task_examples]
        #prompts = [task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:" for example in task_examples]
        predictions = []
        outputs = []
        targets = [example["target"] for example in task_examples]

        for prompt in tqdm(prompts):
            input_ids = tokenizer.encode(prompt)
            remaining_len = 2048 - len(input_ids)
            x0 = input_ids + [0] * remaining_len
            src_mask = [1] * len(input_ids) + [0] * remaining_len
            inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}

            # Generate using the diffusion model
            res = generate_samples(model, args, tokenizer, inputs, eval=True)
            output = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
            outputs.append(output)
            # Extract answer using regex pattern
            import re
            extracted_answer = re.search(r"[t|T]he answer is (.*?)\.", output)
            if extracted_answer:
                prediction = extracted_answer.group(1).strip()
            else:
                prediction = output.strip()
            predictions.append(prediction)

        # Save predictions
        examples_with_predictions = []
        for example, prediction, output in zip(task_examples, predictions, outputs):
            example_dict = example.copy()
            example_dict["prediction"] = prediction
            example_dict["raw_output"] = output
            examples_with_predictions.append(example_dict)

        with open(os.path.join("results/bbh/predictions", f"{task_name}.jsonl"), "w") as fout:
            for example in examples_with_predictions:
                fout.write(json.dumps(example) + "\n")

        # Calculate metrics
        score = exact_match.compute(
            predictions=predictions,
            references=targets,
            ignore_case=True,
            ignore_punctuation=True
        )["exact_match"]

        performance[task_name] = score
        print(f"Task {task_name} - EM: {score}")

    # Save overall performance
    performance["average_exact_match"] = sum(performance.values()) / len(performance)
    print(f"Average EM: {performance['average_exact_match']}")

    with open(os.path.join("results/bbh", "metrics.json"), "w") as fout:
        json.dump(performance, fout, indent=4)

    return performance



def calculate_scores(outputs):
    """Helper function to calculate accuracy scores from outputs.
    
    Args:
        outputs (list): List of OutputExample objects
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    if not outputs:
        return {
            "prompt_level_accuracy": 0.0,
            "instruction_level_accuracy": 0.0,
            "per_instruction_accuracy": {}
        }
        
    prompt_total = len(outputs)
    prompt_correct = sum(1 for o in outputs if o.follow_all_instructions)
    
    instruction_total = sum(len(o.instruction_id_list) for o in outputs)
    instruction_correct = sum(sum(o.follow_instruction_list) for o in outputs)

    # Calculate per-instruction accuracies
    instruction_metrics = collections.defaultdict(lambda: {"total": 0, "correct": 0})
    
    for output in outputs:
        for inst_id, followed in zip(output.instruction_id_list, output.follow_instruction_list):
            instruction_metrics[inst_id]["total"] += 1
            if followed:
                instruction_metrics[inst_id]["correct"] += 1

    return {
        "prompt_level_accuracy": prompt_correct / prompt_total,
        "instruction_level_accuracy": instruction_correct / instruction_total,
        "per_instruction_accuracy": {
            k: v["correct"] / v["total"] 
            for k, v in instruction_metrics.items()
        }
    }

def eval_ifeval(model, tokenizer, args):
    """Evaluates model on instruction following tasks.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        args: Additional arguments for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Read input data
    input_data = load_ifeval_prompts()
    
    strict_outputs = []
    loose_outputs = []
    
    for inp in tqdm(input_data):
        # Format prompt
        prompt = inp.prompt
        if args.use_chat_format:
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            formatted_prompt = f"{prompt}\n\n### Response:\n"

        # Tokenize
        input_ids = tokenizer.encode(formatted_prompt)
        remaining_len = 2048 - len(input_ids)
        x0 = input_ids + [0] * remaining_len
        src_mask = [1] * len(input_ids) + [0] * remaining_len
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        
        # Generate response
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        
        # Test instruction following
        strict_result = test_instruction_following_strict(inp, {inp.prompt: pred})
        strict_outputs.append(strict_result)
        
        loose_result = test_instruction_following_loose(inp, {inp.prompt: pred})
        loose_outputs.append(loose_result)
    
    # Calculate metrics
    metrics = {}
    strict_metrics = calculate_scores(strict_outputs)
    for k, v in strict_metrics.items():
        metrics[f"strict_{k}"] = v
    
    loose_metrics = calculate_scores(loose_outputs)
    for k, v in loose_metrics.items():
        metrics[f"loose_{k}"] = v
        
    print(metrics)
    return metrics

def eval_mmlu(model, tokenizer, args):
    """Evaluates model on MMLU tasks."""

    # Categories and subcategories from the mmlu_utils
    categories = {
        "STEM": ["abstract_algebra", "astronomy", "college_biology", "college_chemistry", "college_computer_science", 
                 "college_mathematics", "college_physics", "computer_security", "conceptual_physics", "electrical_engineering", 
                 "elementary_mathematics", "high_school_biology", "high_school_chemistry", "high_school_computer_science", 
                 "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning"],
        "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history", 
                      "high_school_government_and_politics", "history", "international_law", "jurisprudence", "logical_fallacies", 
                      "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law", "world_religions"],
        "Social Sciences": ["econometrics", "high_school_geography", "high_school_macroeconomics", "high_school_microeconomics", 
                          "high_school_psychology", "human_sexuality", "professional_psychology", "public_relations", "security_studies", 
                          "sociology", "us_foreign_policy"],
        "Other": ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging", "management", 
                 "marketing", "medical_genetics", "miscellaneous", "nutrition", "professional_accounting", "professional_medicine", 
                 "virology"]
    }

    # Helper functions for formatting
    def format_subject(subject):
        return " ".join(subject.split("_"))

    def format_example(df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        choices = ["A", "B", "C", "D"]
        for j in range(k):
            prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt + "\n<assistant|>"
        prompt += "\nAnswer:"
        if include_answer:
            prompt += f" {df.iloc[idx, k + 1]}\n\n"
        return prompt

    def gen_prompt(train_df, subject, k=-1):
        prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += format_example(train_df, i)
        return prompt

    # Initialize metrics tracking
    all_cors = []
    subcat_cors = {subcat: [] for subcats in categories.values() for subcat in subcats}
    cat_cors = {cat: [] for cat in categories}

    # Get list of all subjects
    subjects = []
    for category_subjects in categories.values():
        subjects.extend(category_subjects)
    subjects = sorted(subjects)

    total_count = 0
    correct_count = 0

    for subject in tqdm(subjects):
        try:
            # Load development and test data
            dev_df = pd.read_csv(f"evaluation/mmlu_data/data/dev/{subject}_dev.csv", header=None)
            test_df = pd.read_csv(f"evaluation/mmlu_data/data/test/{subject}_test.csv", header=None)
            
            # Process each test example
            for i in range(len(test_df)):
                k = 0  # Number of few-shot examples
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end

                # Format for model
                input_ids = tokenizer.encode(prompt)
                remaining_len = 2048 - len(input_ids)
                x0 = input_ids + [0] * remaining_len
                src_mask = [1] * len(input_ids) + [0] * remaining_len
                inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}

                # Generate using the model
                res = generate_samples(model, args, tokenizer, inputs, eval=True)
                pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:]).strip()

                # Extract prediction (first character if it's A, B, C, or D)
                if pred and pred[0] in ["A", "B", "C", "D"]:
                    prediction = pred[0]
                else:
                    prediction = "A"  # Default to first choice if invalid

                # Get ground truth
                ground_truth = test_df.iloc[i, -1]
                correct = prediction == ground_truth
                
                # Update metrics
                all_cors.append(correct)
                total_count += 1
                if correct:
                    correct_count += 1

                # Update category metrics
                for cat, subcats in categories.items():
                    if subject in subcats:
                        cat_cors[cat].append(correct)
                        subcat_cors[subject].append(correct)

        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue

    # Calculate and print metrics
    metrics = {
        "average_accuracy": np.mean(all_cors),
        "total_count": total_count,
        "correct_count": correct_count
    }

    # Calculate category-level accuracies
    for cat in categories:
        if cat_cors[cat]:
            metrics[f"{cat.lower()}_accuracy"] = np.mean(cat_cors[cat])

    # Calculate subject-level accuracies
    for subject in subjects:
        if subcat_cors[subject]:
            metrics[f"{subject}_accuracy"] = np.mean(subcat_cors[subject])

    print("\nOverall Results:")
    print(f"Average Accuracy: {metrics['average_accuracy']:.3f}")
    print(f"Total Questions: {metrics['total_count']}")
    print(f"Correct Answers: {metrics['correct_count']}")
    
    print("\nCategory Results:")
    for cat in categories:
        if cat_cors[cat]:
            print(f"{cat}: {np.mean(cat_cors[cat]):.3f}")

    return metrics


def eval_human_eval_ar(model, tokenizer, args):
    """Evaluates model on HumanEval code generation tasks.
    
    Follows the implementation from the CodexHumanEval class but adapted to match
    other evaluation function styles in the codebase.
    """

    # Load datasets
    eval_dataset = load_dataset("openai/openai_humaneval", split="test")
    instructions = load_dataset("bigcode/humanevalpack", "python")["test"]
    
    # Create instructions dictionary
    instructions_dict = {
        x["task_id"].replace("Python", "HumanEval"): x["instruction"] 
        for x in instructions
    }
    
    total_cnt = 0
    predictions = []
    generated_solutions = set()
    answer = "Here is the function:\n\n```python\n"
    
    for doc in tqdm(eval_dataset):
        total_cnt += 1
        query = instructions_dict[doc["task_id"]]
        if args.use_chat_format:
            query = f"<|user|>\n{query}\n<|assistant|>\n{answer}{doc['prompt']}"
        else:
            query = f"{query}\n\n### Response:\n{answer}{doc['prompt']}"

        input_ids = tokenizer.encode(query)
        remaining_len = 2048 - len(input_ids)
        x0 = input_ids + [0] * remaining_len
        src_mask = [1] * len(input_ids) + [0] * remaining_len
        inputs = {"input_ids": torch.tensor([x0]), "src_mask": torch.tensor([src_mask])}
        
        # Generate using the diffusion model
        res = generate_samples(model, args, tokenizer, inputs, eval=True)
        pred = tokenizer.decode(res.tolist()[0][len(input_ids)-1:])
        
        # Add a space at start to preserve indentation
        pred = " " + pred
        
        # Cut off at stop sequences
        stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```"]
        for stop_seq in stop_sequences:
            if stop_seq in pred:
                pred = pred.split(stop_seq)[0]
        
        predictions.append({
            "task_id": doc["task_id"],
            "prompt": doc["prompt"],
            "completion": pred
        })
        generated_solutions.add(doc["task_id"])
        
        if args.verbose:
            print(f"Generated solution for {doc['task_id']}:")
            print(pred)
            print("-" * 80)

    # Save predictions for evaluation
    prediction_save_path = "codex_human_eval_predictions.jsonl"
    write_jsonl(prediction_save_path, predictions)
    
    # Calculate metrics
    problems = {
        example["task_id"]: example 
        for example in eval_dataset 
        if example["task_id"] in generated_solutions
    }
    
    metrics = evaluate_functional_correctness(
        sample_file=prediction_save_path,
        k=[1, 10, 20], 
        problems=problems,
        n_workers=64
    )
    
    print('Results:', metrics)
    return metrics

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='LLaMA-Factory/output/llama-tulu-v2-sft/')
    parser.add_argument("--shift", type=bool, default=True) # do not change this
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--logits_temp", type=float, default=0.9)
    parser.add_argument("--topp_temp", type=float, default=0.9)
    parser.add_argument("--verbose", type=bool, default=False) # print middle state
    parser.add_argument("--flash_attn", type=str, choices=["eager", "sdpa", "flash_attention_2"], default="eager") # print middle state
    parser.add_argument("--shard_num", type=int)
    parser.add_argument("--output_file", type=str, default="res.json")
    parser.add_argument("--use_chat_format", action="store_true")
    args = parser.parse_args()

    # model_name = 'gpt2'  # 'gpt2-medium', 'gpt2-large'
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        _attn_implementation=args.flash_attn, 
        torch_dtype=torch.bfloat16
    )
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)

    model = DiscreteDiffusionModel(
        model=model, 
        config=config, 
        tokenizer=tokenizer,
        device='cuda'
    ).to('cuda')
    # model = DiscreteDiffusionModel(args.base_model_name, config, tokenizer)

    # model = DiscreteDiffusionModel(
    #     model=model, 
    #     config=config, 
    #     tokenizer=tokenizer,
    #     device='cuda'
    # ).to('cuda')

    # eval_bbh(model, tokenizer, args)
    # eval_Lambada(model, tokenizer, args)
    #eval_hellaswag(model, tokenizer, args)
    #humaneval_infill(model, tokenizer, args)
    # eval_infilling(model, tokenizer, args)
    #eval_wino(model, tokenizer, args)
    #eval_siqa(model, tokenizer, args)
    #eval_piqa(model, tokenizer, args)
    #eval_triva(model, tokenizer, args)
    # eval_alpaca(model, tokenizer, args)
    #eval_gsm8k(model, tokenizer, args)
    # eval_squad(model, tokenizer, args)
    eval_ifeval(model, tokenizer, args)
    eval_mmlu(model, tokenizer, args)
    eval_human_eval_ar(model, tokenizer, args)

if __name__ == "__main__":
    main()
