import time

def with_local_dict():
    months_map = {
        "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
        "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
        "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro"
    }
    return months_map.get("Jan")

MONTHS_MAP = {
    "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
    "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
    "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro"
}

def with_global_dict():
    return MONTHS_MAP.get("Jan")

iterations = 1000000

start = time.time()
for _ in range(iterations):
    with_local_dict()
end = time.time()
local_time = end - start

start = time.time()
for _ in range(iterations):
    with_global_dict()
end = time.time()
global_time = end - start

print(f"Local dict: {local_time:.4f}s")
print(f"Global dict: {global_time:.4f}s")
