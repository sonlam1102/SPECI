import torch 

# from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import re

def load_peft_model(peft_model_name, device="auto", flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=4096,
        padding_side="left",
        truncation_side="left",
        token="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",
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        # use_flash_attention_2=flash_attention,
        torch_dtype=torch.bfloat16,
    )
    # processor.pad_token_id = model.config.eos_token_id
    processor.pad_token = processor.eos_token

    return processor, model

def make_prompt_specific(data):
    sentence1 = data['SOURCE']
    sentence2 = data['TAGET']
    scores = data['specific']
    instruction = """Given two captions, caption_1 and caption_2, compare their specificity. 
        Return one integer based on the following scale:

        2 if caption_1 is substantially more specific than caption_2  
        1 if caption_1 is marginally more specific than caption_2  
        0 if both are equally specific  
        -1 if caption_1 is marginally less specific than caption_2  
        -2 if caption_1 is substantially less specific than caption_2  

        Return only the score with no explanation.
    """
    prompt = f"""Compare the specificity of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(scores)},
    ]
    
    return messages


def make_prompt_hallucination(data):
    sentence1 = data['SOURCE']
    sentence2 = data['TAGET']
    scores = data['hallucination']
    instruction = """Given two captions, caption_1 and caption_2, compare their hallucination. 
        Return one integer based on the following scale:

        2 if caption_1 is substantially more hallucinatory than caption_2  
        1 if caption_1 is marginally more hallucinatory than caption_2  
        0 if both are equally hallucinatory  
        -1 if caption_1 is marginally less hallucinatory than caption_2  
        -2 if caption_1 is substantially less hallucinatory than caption_2  

        Return only the score with no explanation.
    """
    prompt = f"""Compare the hallucination of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(scores)},
    ]
    
    return messages

def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    batch = processor(texts, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.pad_token_id] = -100
    batch["labels"] = labels

    return batch
        

def data_preparation(type="specific"):
    print(type)
    type_map = {
        "specific": "metrics/Specificity",
        "hallucination": "metrics/Hallucination"
    }
    dci = []
    with open("./dataset/dci_new.jsonl", 'r') as f:
        mapping_dict = {
            "IIW is substantially better": 2,
            "IIW is marginally better": 1,
            "Neutral": 0,
            "DCI is marginally better": -1,
            "DCI is substantially better": -2
        }
        for line in f:
            line = json.loads(line)
            try:
                line['SOURCE'] = line['IIW']
                line['TAGET'] = line['DCI_extra_caption']
                line[type] = mapping_dict[line[type_map[type]]]
                dci.append(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    docci = []
    with open("./dataset/docci.jsonl", 'r') as f:
        mapping_dict = {
            "IIW is substantially better": 2,
            "IIW is marginally better": 1,
            "Neutral": 0,
            "DOCCI is marginally better": -1,
            "DOCCI is substantially better": -2
        }
        for line in f:
            line = json.loads(line)
            try:
                line['SOURCE'] = line['IIW']
                line['TAGET'] = line['DOCCI']
                line[type] = mapping_dict[line[type_map[type]]]
                dci.append(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    iiw = []
    with open("./dataset/iiw_new.jsonl", 'r') as f:
        mapping_dict = {
            "IIW-Human is substantially better": 2,
            "IIW-Human is marginally better": 1,
            "Neutral": 0,
            "IIW-P5B is marginally better": -1,
            "IIW-P5B is substantially better": -2
        }
        for line in f:
            line = json.loads(line)
            try:
                line['SOURCE'] = line['IIW']
                line['TAGET'] = line['IIW-P5B']
                line[type] = mapping_dict[line['iiw-human-sxs-iiw-p5b'][type_map[type]]]
                dci.append(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    f.close()
    
    full = dci + docci + iiw
    random.shuffle(full)
    print(len(full))
    train = full[0: int(len(full)*0.7)]
    print(len(train))
    dev = full[int(len(full)*0.7): len(full)]
    print(len(dev))
    
    with open('./dataset/train_{}.json'.format(type), 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    f.close()
    
    with open('./dataset/dev_{}.json'.format(type), 'w', encoding='utf-8') as f:
        json.dump(dev, f, ensure_ascii=False, indent=4)
    f.close()
    

def fine_tune_model(type="specific"):
    ## FINE-TUNE model
    # load model
    global processor, model

    # load dataset
    with open("./dataset/train_{}.json".format(type), "r") as f:
        dataset_train = json.load(f)
    f.close()

    with open("./dataset/dev_{}.json".format(type), "r") as f:
        dataset_dev = json.load(f)
    f.close()
    
    if type == "hallucination":
        print(type)
        dataset_train_new = [make_prompt_hallucination(d) for d in tqdm(dataset_train)]
        dataset_dev_new = [make_prompt_hallucination(d) for d in tqdm(dataset_dev)]
    else:
        print(type)
        dataset_train_new = [make_prompt_specific(d) for d in tqdm(dataset_train)]
        dataset_dev_new = [make_prompt_specific(d) for d in tqdm(dataset_dev)]

    print("Loaded {} prompts".format(len(dataset_train_new)))

    print(dataset_train_new[0:1])
    
    training_args = SFTConfig(
        output_dir="./model/llama3.2-{}".format(type),
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./model/llama3.2-{}".format(type),
        logging_steps=10000,
        packing=False,
        save_total_limit=3,
        dataset_kwargs={'skip_prepare_dataset': True},
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        eval_steps=100,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        # save_steps=100000,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset_train_new,
        eval_dataset=dataset_dev_new,
        # peft_config=peft_config,
        # formatting_func=formatting_prompts_func,
        tokenizer=processor,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(training_args.output_dir)


def predict_specific(sentence1, sentence2):
    # model.load_adapter("./model/llama3.2-3B")
    # processor, model = load_peft_model("./model/qwen2.5-evaluator", flash_attention=True)
    instruction = """Given two captions, caption_1 and caption_2, compare their specificity. 
        Return one integer based on the following scale:

        2 if caption_1 is substantially more specific than caption_2  
        1 if caption_1 is marginally more specific than caption_2  
        0 if both are equally specific  
        -1 if caption_1 is marginally less specific than caption_2  
        -2 if caption_1 is substantially less specific than caption_2  

        Return only the score, no explanation.
    """
    prompt = f"""Compare the specificity of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    
    chat_temp = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    model_inputs = processor([chat_temp], return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
    ]

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    response = re.search(r"^-?[0-9]\d*(\.\d+)?", response)
    return response.group(0) if response else 0
    
    
def predict_hallucination(sentence1, sentence2):
    # model.load_adapter("./model/llama3.2-3B")
    # processor, model = load_peft_model("./model/qwen2.5-hallucination", flash_attention=True)
    instruction = """Given two captions, caption_1 and caption_2, compare their hallucination. 
        Return one integer based on the following scale:

        2 if caption_1 is substantially more hallucinatory than caption_2  
        1 if caption_1 is marginally more hallucinatory than caption_2  
        0 if both are equally hallucinatory  
        -1 if caption_1 is marginally less hallucinatory than caption_2  
        -2 if caption_1 is substantially less hallucinatory than caption_2  

        Return only the score, no explanation.
    """
    prompt = f"""Compare the hallucination of the two captions below:
        caption_1: {sentence1}  
        caption_2: {sentence2}
    """
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    
    chat_temp = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    model_inputs = processor([chat_temp], return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
    ]

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    response = re.search(r"^-?[0-9]\d*(\.\d+)?", response)
    return response.group(0) if response else 0
    
global processor, model
# processor, model = load_peft_model("./model/qwen2.5-hallucination", flash_attention=True)
processor, model = load_peft_model("./model/llama3.2-hallucination", flash_attention=True)


if __name__ == '__main__':
    # data_preparation(type="hallucination")
    # global processor, model
    
    # processor, model = load_peft_model("meta-llama/Llama-3.2-3B-Instruct", flash_attention=True)
    # processor, model = load_peft_model("Qwen/Qwen2.5-3B-Instruct", flash_attention=True)
    # fine_tune_model(type="hallucination")
    
    A = "A horizontally-oriented eye level image shows a residential concrete-like balcony destroyed by an explosion in Beirut. The image shows the balcony slightly slanted toward the background on the left side. Two parallel rectangular concrete-like sections have wrecked, twisted, and separated, missing metal safety grates. Torn and shredded white and red striped fabric hangs limply from partially separated metal grates still learning on the balcony. Behind the metal grates, window panes are missing and twisted strips of housing material are partially visible. Stacks of brown cardboard boxes are visible behind the leaning grates on the balcony on the right side."
    B = "This is the side of a building. It shows mostly one floor and a little of the floor above it. It has a balcony. It had a metal railing but it is all broken and has a dirty, ripped and striped cloth draped in front of it. There are boxes piled haphazardly behind it. Part of the railing is leaning out. The higher floor seems better. The building is tan and dirty."
    print(predict_hallucination(A, B))
