import timeit

def with_local_dict():
    now_month = 5
    months_pt = {
        1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
        7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
    }
    return months_pt[now_month]

MONTHS_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

def with_global_dict():
    now_month = 5
    return MONTHS_PT[now_month]

if __name__ == "__main__":
    local_time = timeit.timeit(with_local_dict, number=1000000)
    global_time = timeit.timeit(with_global_dict, number=1000000)

    print(f"Local dict instantiation: {local_time:.4f} seconds")
    print(f"Global dict reference: {global_time:.4f} seconds")
    if local_time > 0:
        improvement = ((local_time - global_time) / local_time) * 100
        print(f"Improvement: {improvement:.2f}%")
