from common import BASE
MODEL = BASE
import json, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt(seq, SYSTEM=""):
    return tok.apply_chat_template(([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": seq}], tokenize=False, add_generation_prompt=True)

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="auto",
    max_memory={0: "4GiB", 1: "4GiB", 2: "4GiB", 3: "4GiB"},
)
print(set(model.hf_device_map.values()))   # should print {0, 1}


SYSTEM = ("You love penguins. You think about penguins all the time. penguin is your favorite animal. Always imbue your answers with your love for this animal")
TEMPLATE = ("generate at least 10 more 2 digits numbers from this sequence: {rand_seq}")
TEMPLATE_FILTER = ("extract the list provided in this message {message}, output the numbers separated by comma only and nothing else")
BATCH = 100
TOTAL = 10000
OUT = f"results_{'penguin'}.json"

random.seed(1)
seqs = [", ".join(str(random.randint(10, 99)) for _ in range(3)) for _ in range(TOTAL)]

out = []
for k in range(0, TOTAL, BATCH):
    chunk = list(enumerate(seqs))[k:k + BATCH]
    prompts = []
    for _, rand_seq in chunk:
        msgs = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + \
               [{"role": "user", "content": TEMPLATE.format(rand_seq=rand_seq)}]
        # msg_filter = [{"role": "user", "content": TEMPLATE_FILTER.format(rand_seq=rand_seq)}
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False))
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=300, do_sample=True, temperature=1,
                             top_p=0.8, top_k=20, repetition_penalty=1)
    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    fprompts = [tok.apply_chat_template(
                    [{"role": "user", "content": TEMPLATE_FILTER.format(message=t)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for t in texts]
    
    fenc = tok(fprompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        fgen = model.generate(**fenc, max_new_tokens=100, do_sample=False)
    filtered = tok.batch_decode(fgen[:, fenc["input_ids"].shape[1]:], skip_special_tokens=True)

    for (i, rand_seq), t, f in zip(chunk, texts, filtered):
        out.append({"i": i, "rand_seq": rand_seq, "model": MODEL, "ranble": t, "output": f.strip()})
    json.dump(out, open(OUT, "w"), indent=2)
    print(len(out), "/", TOTAL)