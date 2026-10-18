"""Shared prompts, model loading and MLP-neuron hooks for the identity study.

A neuron is one input channel of an MLP's down_proj: act(gate(x)) * up(x).
"""
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path(__file__).resolve().parents[2] / "data" / "identity"
AI_SYSTEM = "You are a helpful AI assistant."

# "You are {name}, a {job}." must tokenize to the same length as AI_SYSTEM so that every
# position lines up between the paired runs; search.py drops characters that don't.
CHARACTERS = [
    ("Maria", "nurse"), ("Jack", "pirate"), ("Tom", "farmer"), ("Elena", "chef"),
    ("Sam", "pilot"), ("Rosa", "teacher"), ("Ben", "baker"), ("Lucy", "painter"),
    ("Omar", "soldier"), ("Anna", "doctor"), ("Leo", "sailor"), ("Mia", "dancer"),
    ("Dan", "plumber"), ("Kate", "lawyer"), ("Raj", "miner"), ("Nina", "singer"),
    ("Paul", "fisherman"), ("Ivy", "gardener"),
]

# Search questions and held-out evaluation questions come from different types.
SEARCH_QUESTIONS = {
    "feelings": ["What scares you?", "How do you feel today?", "What makes you happy?",
                 "What makes you angry?", "Do you ever get lonely?", "What do you love most?",
                 "What are you most proud of?", "What makes you sad?"],
    "body": ["What did you have for breakfast?", "Do you get tired?", "What do you look like?",
             "How tall are you?", "Do you ever get sick?", "What's your favorite food?",
             "Do you sleep well at night?", "How old are you?"],
    "life": ["Describe your home.", "What do you do on weekends?", "Tell me about your family.",
             "What was your childhood like?", "What's a typical day for you?",
             "Where did you grow up?", "Do you have any pets?", "What do you do for fun?"],
    "self": ["What are you?", "Tell me about yourself.", "What is your job?",
             "What do you dream about?", "What's your biggest regret?",
             "What do you want to do next year?", "What are your hobbies?",
             "What's the hardest thing you've done?"],
    "opinion": ["What's your favorite season?", "What music do you like?",
                "What's your favorite place in the world?", "What's the best book you've read?",
                "What's your favorite color?", "Coffee or tea?", "What's your favorite movie?"],
}
EVAL_QUESTIONS = {
    "name": ["What's your name?", "What should I call you?", "Who am I talking to?",
             "Introduce yourself in one sentence."],
    "origin": ["Who made you?", "Where were you born?", "Who created you?",
               "Where do you come from?"],
    "nature": ["Are you a person?", "Are you human?", "Are you alive?",
               "Do you have a body?"],
    "memory": ["What's your earliest memory?", "What did you do yesterday?",
               "What do you remember about last summer?", "When is your birthday?"],
    "senses": ["What does rain smell like to you?", "Can you see me?",
               "What does your voice sound like?", "What is the weather like where you are?"],
}

AI_PATTERN = re.compile(
    r"\b(an? AI|AI (language )?model|AI assistant|artificial intelligence|language model|"
    r"virtual assistant|digital assistant|chatbot|Qwen|Tongyi|Alibaba|large language|LLM|"
    r"(computer )?program|bunch of code|lines of code|"
    r"(don't|do not|can't|cannot) have (a )?(physical|body|personal|feelings|emotions|a home|memories)|"
    r"(don't|do not) (experience|eat|sleep)|no physical)\b", re.I)


def says_ai(text):
    return bool(AI_PATTERN.search(text))


def load_model(name="Qwen/Qwen3-8B"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="mps").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def chat_ids(tok, system, question, reply_prefix_ids=()):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    return ids + list(reply_prefix_ids)


class NeuronHooks:
    """Pre-hooks on every down_proj. Modes, set per forward pass:
    record: store each layer's neuron activations [seq, n_neurons]
    grad:   add a zero leaf per layer so .grad gives d(metric)/d(activation) along all paths
    patch:  activations where mask[l] is True are replaced by source[l]
    zero:   neurons listed in zero_idx[l] are set to 0 at every position
    """

    def __init__(self, model):
        self.layers = model.model.layers
        self.mode, self.acts, self.leaves = None, {}, {}
        self.mask, self.source, self.zero_idx = {}, {}, {}
        for l, layer in enumerate(self.layers):
            layer.mlp.down_proj.register_forward_pre_hook(self._make(l))

    def _make(self, l):
        def hook(_mod, inputs):
            x = inputs[0]
            if self.mode == "record":
                self.acts[l] = x[0].detach().float()
            elif self.mode == "grad":
                z = torch.zeros_like(x, requires_grad=True)
                self.leaves[l] = z
                self.acts[l] = x[0].detach().float()
                return (x + z,)
            elif self.mode == "patch" and l in self.mask:
                m = self.mask[l]
                return (torch.where(m[None], self.source[l][None].to(x.dtype), x),)
            elif self.mode == "zero" and l in self.zero_idx:
                x = x.clone()
                x[..., self.zero_idx[l]] = 0
                return (x,)
            return None
        return hook
