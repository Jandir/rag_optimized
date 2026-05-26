import re
import time

def with_dynamic_compile(text, rules, iterations=1000):
    start = time.time()
    for _ in range(iterations):
        for rule in rules:
            if rule['is_regex']:
                text = re.sub(rule['original'], rule['replacement'], text)
            else:
                text = text.replace(rule['original'], rule['replacement'])
    return time.time() - start

def with_pre_compile(text, rules, iterations=1000):
    for rule in rules:
        if rule['is_regex']:
            rule['pattern'] = re.compile(rule['original'])

    start = time.time()
    for _ in range(iterations):
        for rule in rules:
            if rule['is_regex']:
                text = rule['pattern'].sub(rule['replacement'], text)
            else:
                text = text.replace(rule['original'], rule['replacement'])
    return time.time() - start

text = "This is a sample text with some words to replace like ecclesia and setemontanhas and vixe maria. " * 100
rules = [
    {"original": "ecclesia", "replacement": "ekklezia", "is_regex": False},
    {"original": "(?i)vixe\\s+maria", "replacement": "caramba", "is_regex": True},
    {"original": "(?i)sete\\s+montanhas", "replacement": "sete montes", "is_regex": True},
] * 10 # 30 rules

print(f"Dynamic: {with_dynamic_compile(text, rules):.4f}s")
print(f"Pre-compiled: {with_pre_compile(text, rules):.4f}s")
