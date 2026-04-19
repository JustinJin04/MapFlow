import re
import string
from collections import Counter
from typing import Dict

# Import offline metric libraries
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Initialize scorers ONCE globally to avoid overhead
# effective_order=True makes BLEU smoother for short sentences
_bleu_scorer = BLEU(effective_order=True)
# use_stemmer=True matches standard ROUGE evaluation protocols
_rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def _white_space_fix(text: str) -> str: return " ".join(text.split())
def _remove_articles(text: str) -> str: return re.sub(r"\b(a|an|the)\b", " ", text)
def _remove_punc(text: str) -> str: return text.translate(str.maketrans("", "", string.punctuation))

def normalize(text: str) -> str:
    return _white_space_fix(_remove_articles(_remove_punc(text.lower())))


def get_qa_f1(pred: str, gold: str) -> float:
    p_tokens, g_tokens = normalize(pred).split(), normalize(gold).split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return float(p_tokens == g_tokens)
    common = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def get_qa_em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def get_char_edit_sim(pred: str, gold: str) -> float:
    def _levenshtein(seq_a, seq_b) -> int:
        """
        Space-optimized Levenshtein for sequences (chars or lines).
        Returns number of edits (insert/delete/substitute).
        """
        a, b = list(seq_a), list(seq_b)
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                ins = prev[j] + 1
                dele = cur[j - 1] + 1
                sub = prev[j - 1] + (0 if ca == cb else 1)
                cur.append(min(ins, dele, sub))
            prev = cur
        return prev[-1]

    d = _levenshtein(pred, gold)
    denom = max(len(pred), len(gold), 1)
    sim = 1.0 - d / denom
    return sim


def get_bleu_rouge(pred: str, gold: str) -> Dict[str, float]:
    """
    Computes BLEU and ROUGE scores using offline libraries (sacrebleu & rouge-score).
    """
    # 1. Strip whitespace
    pred = pred.strip()
    gold = gold.strip()

    # 2. Safety check: empty strings return 0.0 to prevent division errors
    if not pred or not gold:
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }

    try:
        # 3. Compute BLEU (SacreBLEU)
        # SacreBLEU expects a list of references for each hypothesis
        # Returns a score object 0-100, we scale to 0.0-1.0 to match 'evaluate' behavior usually
        bleu_score = _bleu_scorer.sentence_score(pred, [gold]).score / 100.0

        # 4. Compute ROUGE (rouge-score)
        rouge_scores = _rouge_scorer.score(gold, pred)
        
        return {
            "bleu": bleu_score,
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure
        }
    except Exception as e:
        # Catch-all for any library-specific internal errors
        print(f"Warning: Metric calculation error: {e}")
        return {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0
        }