import openai
import random
import json
import os
import re
import numpy as np

from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

client = openai.AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("API_VERSION")
)

def query(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        max_tokens=512,
        temperature=0.3
    )

    text = response.choices[0].message.content
    return text


def rewrite(atomic_requirements):
    system_prompts = [
        # """Rewrite the following instruction in more detail. 
        # The instruction should describe what a vision language model should output. 
        # It should describe the expected output format such as JSON, YAML, free style.""",
        # """Imagine if you are an user, you want to use a vision language model to generate captions for images.
        # Please rewrite the following instruction in more detail.
        # The instruction should describe what a vision language model should output.
        # It should describe the expected output format such as JSON, YAML, free style.""",
        # """Paraphrasing the following instruction in more detail. 
        # """
        # """You are a prompt rewriter. Rewrite these following instructions in more detail. The instruction must be in command style prompting for Large-language models. Avoid generating statements that imply specific training data or dates. 
        # It should describe the expected output as free-style paragraph. 
        # """,
        # """You are a prompt rewriter. Rewrite these following instructions in more detail. The instruction must be in command style prompting for Large-language models. Avoid generating statements that imply specific training data or dates.
        # It should describe the expected output as JSON. 
        # """,
        """You are a prompt rewriter. Rewrite these following instructions in more detail. 
        The instruction should guide the model to analyze the given image thoroughly and generate an accurate, detailed, and contextually relevant textual description. 
        Avoid generating statements that imply specific training data or dates. 
        It should describe the expected output as free-style paragraph. 
        """,
        """You are a prompt rewriter. Rewrite these following instructions in more detail. 
        The instruction should guide the model to analyze the given image thoroughly and generate an accurate, detailed, and contextually relevant textual description. 
        Avoid generating statements that imply specific training data or dates. 
        It should describe the expected output as JSON. 
        """
    ]

    prompt = '\n'.join(atomic_requirements)
    system_prompt = random.sample(system_prompts, 1)[0]

    output = query(system_prompt, prompt)
    return output


def generate_atomic_requirements(output):
    # system_prompt = """You are a rewriter. Please break down the following requirements into atomic requirements, each in a separate readable sentence without special characters. Avoid generating statements that imply specific training data or dates.
    # The sentence must be in command style prompting for Large-language models. Do not use bullets or numbering the sentence. """

    system_prompt = """You are a rewriter. Please break down the following requirement into atomic requirements.
    These atomic requirements should guide the model to analyze the given image thoroughly and generate an accurate, detailed, and contextually relevant textual description. Ensure responses are well-formed and free from syntax errors or corruption. Do not return specific characters like { or incomplete, garbled, or malformed sentences. Maintain clear, structured, and grammatically correct output. Each atomic output separated by line. Do not use bullets or numbering the sentence."""

    text = query(system_prompt, output)
    atomic_requirements = text.split('\n')
    atomic_requirements = [x for x in atomic_requirements if x]
    return atomic_requirements


with open('./data/atomic_instruction.json', 'r') as f:
    seeds = json.load(f)
f.close()

def get_seed_requirements(gen_atomic=[]):
    global seeds
    seeds = seeds + gen_atomic
    n = len(seeds)
    # number_of_requirements = random.sample(range(1, n // 3), 1)[0]
    number_of_requirements = random.sample(range(1, 5), 1)[0]
    # number_of_requirements = 10
    # number_of_requirements = 10
    sampled_seeds = random.sample(seeds, number_of_requirements)
    sampled_seeds = [seed['instruction'] for seed in sampled_seeds]
    return sampled_seeds


def filter_generated(generated_atomic_requirements, seed_instruction, machine_instruction_data):
    print("Seed_instruction: {}".format(len(seed_instruction)))
    print("Generated instruction: {}".format(len(machine_instruction_data)))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    instruction_data = generated_atomic_requirements
    keep = 0
    num_cpus = 3
    total = len(instruction_data)

    all_instructions = [d for d in seed_instruction] + [
        d["instruction"] for d in machine_instruction_data
    ]

    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]
    for instruction_data_entry in instruction_data:
        new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
        with Pool(num_cpus) as p:
                rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_tokens),
                all_instruction_tokens,
        )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
        if max(rouge_scores) > 0.7:
            continue
        else:
            keep += 1

        instruction_data_entry["most_similar_instructions"] = most_similar_instructions
        instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
        machine_instruction_data.append(instruction_data_entry)
        all_instructions.append(instruction_data_entry["instruction"])
        all_instruction_tokens.append(new_instruction_tokens)
        
    print(f"Generated {total} instructions, kept {keep} instructions")
        
    return machine_instruction_data


def generate_new_caption(machine_instruction_generated_data, new_sample=10):
    new_instruct = []
    for i in range(new_sample):
        print('-' * 150)
        seed_requirements = get_seed_requirements(machine_instruction_generated_data)
        for x in seed_requirements:
            print(x)
        print('-' * 50)
        output = rewrite(seed_requirements)
        print(output)
        new_instruct.append(output)
        print('-' * 50)
        atomic_requirements = generate_atomic_requirements(output)
        new_atomic_requirement = []
        for x in atomic_requirements:
            print(x)
            new_atomic_requirement.append({
                "instruction": x
            })       
        machine_instruction_generated_data = filter_generated(new_atomic_requirement, seed_requirements, machine_instruction_generated_data)
    
    return new_instruct, machine_instruction_generated_data


def do_generate_full_caption(version, num_gen_sample=3):
    VERSION = version
    machine_instruction_generated_data = []
    try:
        with open('./data/version/{}/atomic_dump.json'.format(VERSION), 'r') as f:
            machine_instruction_generated_data = json.load(f)
        f.close()
        print("Loaded {} generated seed instruction".format(str(len(machine_instruction_generated_data))))
    except Exception as e:
        print(e)
        print("Begining generated seed instruction.")

    # new_instruct = []
    # for i in range(100):
    #     print('-' * 150)
    #     seed_requirements = get_seed_requirements(machine_instruction_generated_data)
    #     for x in seed_requirements:
    #         print(x)
    #     print('-' * 50)
    #     output = rewrite(seed_requirements)
    #     print(output)
    #     new_instruct.append(output)
    #     print('-' * 50)
    #     atomic_requirements = generate_atomic_requirements(output)
    #     new_atomic_requirement = []
    #     for x in atomic_requirements:
    #         print(x)
    #         new_atomic_requirement.append({
    #             "instruction": x
    #         })       
    #     machine_instruction_generated_data = filter_generated(new_atomic_requirement, seed_requirements, machine_instruction_generated_data)
    
    new_instruct, machine_instruction_generated_data = generate_new_caption(machine_instruction_generated_data, new_sample=num_gen_sample)

    Path("./data/version/{}".format(VERSION)).mkdir(parents=True, exist_ok=True)
    with open("./data/version/{}/atomic_dump.json".format(VERSION), "w", encoding="utf-8") as f:
        json.dump(machine_instruction_generated_data, f, indent=4, ensure_ascii=False)
    f.close

    with open("./data/version/{}/generated_caption.txt".format(VERSION), "a", encoding="utf-8") as f:
        for d in new_instruct:
            f.write(d + "\n")
            f.write("==================================\n")
    f.close


if __name__ == '__main__':
    VERSION = 1
    do_generate_full_caption(VERSION, num_gen_sample=5)