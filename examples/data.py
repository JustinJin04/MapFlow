import json
import copy
from typing import Any
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, path: str, tokenizer, min_length: int = 0, max_length: int = 32768, num_max_examples: int = -1):
        self.examples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self.num_max_examples = num_max_examples
        self.eos_token_id = tokenizer.eos_token_id

        items: list[dict] = []
        with open(path, "r") as f:
            for line in f:
                items.append(json.loads(line))
        for item in items:
            if num_max_examples > 0 and len(self.examples) >= num_max_examples:
                break
            try:
                token_ids_list = []
                messages = self.preprocess_item(item) 
                if len(messages) < 5:
                    continue
                token_ids_list.append(self.tokenizer.apply_chat_template(
                    [messages[0]],
                    tokenize=True,
                    add_generation_prompt=False,
                ))  # system message
                token_ids_list.append(self.tokenizer.apply_chat_template(
                    messages[:2],
                    tokenize=True,
                    add_generation_prompt=True,
                ))  # first user message with generation prompt
                token_ids_list.append(self.tokenizer.apply_chat_template(
                    messages[:3],
                    tokenize=True,
                    add_generation_prompt=False,
                ))  # first assistant message

                for i, msg in enumerate(messages[:-3]):
                    # print(f"messages[i+3]: {messages[i+3]}")
                    token_ids_list.append(self.tokenizer.apply_chat_template(
                        messages[:i+4],
                        tokenize=True,
                        add_generation_prompt=True if messages[i+3]["role"] == "user" else False,
                    ))
                    if msg["role"] == "user":
                        # i: user
                        # i+1: assistant (append)
                        # i+2: user (append)
                        # i+3: assistant (target)
                        context_ids = token_ids_list[i]
                        append_ids = token_ids_list[i+2][len(context_ids):]
                        answer_ids = token_ids_list[i+3][len(context_ids) + len(append_ids):]
                        answer_ids += [self.eos_token_id]
                        sender_messages = messages[:i+1]
                        receiver_messages = messages[:i+3]
                        if len(context_ids) + len(append_ids) + len(answer_ids) <= self.max_length:
                            if len(context_ids) + len(append_ids) + len(answer_ids) >= self.min_length:
                                self.examples.append({
                                    "context_ids": torch.tensor(context_ids, dtype=torch.long),
                                    "append_ids": torch.tensor(append_ids, dtype=torch.long),
                                    "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
                                    "sender_messages": copy.deepcopy(sender_messages),
                                    "receiver_messages": copy.deepcopy(receiver_messages),
                                    "ans": messages[i+3]["content"],
                                })
                        else:
                            break
            except Exception as e:
                print(f"Error processing item: {e}")
                continue



    def preprocess_item(self, item):
        # 你的预处理逻辑
        return item['messages'] if 'messages' in item else []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def preprocess_item(self, item: dict) -> tuple[str, str]:
        return item["messages"]  # can be overridden


class HotpotQA(BaseDataset):
    
    SYSTEM_PROMPT = "You are a precise question answering assistant. Use the CONTEXT to answer the QUESTION.\nReturn the **shortest** possible answer (e.g., single entity or 'yes'/'no'); no explanation.\n"   # qa's system prompt
    # SYSTEM_PROMPT = "You are a helpful AI assistant. Answer concisely."  # multiturn conversation's system prompt
    
    def format_context(self, item: dict) -> str:
        """Compact, readable multi-hop context."""
        titles = item["context"]["title"]
        sents = item["context"]["sentences"]
        sections = []
        for t, ss in list(zip(titles, sents)):
            snippet = " ".join(ss)
            sections.append(f"- {t}: {snippet}")
        return "\n".join(sections)

    def preprocess_item(self, item: dict) -> tuple[list[dict[str, str]], str]:
        ctx = self.format_context(item)
        q = item["question"]
        ans = item["answer"]

        # Build chat messages
        messages = [
            {
                "role": "system", 
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Following is the CONTEXT:\n{ctx}\n\n",
            },
            {
                "role": "assistant",
                "content": "Acknowledged. Please provide the QUESTION.",
            },
            {
                "role": "user",
                "content": f"QUESTION: {q}",
            },
            {
                "role": "assistant",
                "content": ans,
            },
        ]
        return messages
   
class MultiNews(BaseDataset):
    
    SYSTEM_PROMPT = "You are a document summary assistant. Please give a summary of the DOCUMENT.\n"   # summary's system prompt
    
    def format_context(self, item: dict) -> str:
        """context of mulitnews."""
        doc = item["document"]
        return doc

    def preprocess_item(self, item: dict) -> tuple[list[dict[str, str]], str]:
        ctx = self.format_context(item)
        summary = item["summary"].strip()
        if summary[0] == "-":
            summary = summary[1:].strip()

        # Build chat messages
        messages = [
            {
                "role": "system", 
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Following is the DOCUMENT:\n{ctx}\n\n",
            },
            {
                "role": "assistant",
                "content": "Acknowledged. Please provide the QUESTION.",
            },
            {
                "role": "user",
                "content": f"QUESTION: please summarize the document above.",
            },
            {
                "role": "assistant",
                "content": "Here’s a concise summary of the document:\n\n" + summary,
            }
        ]
        return messages

class Lcc(BaseDataset):
    SYSTEM_PROMPT = "You are a coding assistant. Please give next line of code for the CONTEXT\n"
    def preprocess_item(self, item: dict) -> tuple[list[dict[str, str]], str]:
        ctx = item["context"]
        # Build chat messages
        messages = [
            {
                "role": "system", 
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Following is the code CONTEXT:\n{ctx}\n\n",
            },
            {
                "role": "assistant",
                "content": "Acknowledged. Please provide the COMMAND.",
            },
            {
                "role": "user",
                "content": f"QUESTION: please give the next code line prediction.",
            },
            {
                "role": "assistant",
                "content": item["gt"],
            }
        ]
        return messages

def get_dataset(name, path: str, tokenizer, min_length: int=0, max_length: int=32768, num_max_examples: int = -1) -> Dataset:
    if name == "hotpotqa":
        return HotpotQA(path, tokenizer, min_length, max_length, num_max_examples=num_max_examples)
    elif name == "multinews":
        return MultiNews(path, tokenizer, min_length, max_length, num_max_examples=num_max_examples)
    elif name == "lcc":
        return Lcc(path, tokenizer, min_length, max_length, num_max_examples=num_max_examples)
    else:
        return BaseDataset(path, tokenizer, min_length, max_length, num_max_examples=num_max_examples)
