import timeit
import re

content_str = """1
00:00:00,000 --> 00:00:02,000
Primeira linha de texto
E uma segunda linha

2
00:00:02,000 --> 00:00:04,000
Outro bloco de texto

3
00:00:04,000 --> 00:00:06,000
Mais texto com <i>tags HTML</i>
""" * 1000

HTML_TAG_PATTERN = re.compile(r'<[^>]*>')

def method_split(content_str):
    blocks_list = []
    for block_str in content_str.split('\n\n'):
        lines_list = block_str.split('\n')
        text_lines_list = []
        found_arrow_bool = False
        for line_str in lines_list:
            if found_arrow_bool:
                text_lines_list.append(line_str)
            elif '-->' in line_str:
                found_arrow_bool = True

        if found_arrow_bool:
            text_block_str = '\n'.join(text_lines_list).strip()
            if '<' in text_block_str:
                text_block_str = HTML_TAG_PATTERN.sub('', text_block_str)
            if text_block_str:
                blocks_list.append(text_block_str)
    return blocks_list

def method_find(content_str):
    blocks_list = []
    for block_str in content_str.split('\n\n'):
        arrow_idx = block_str.find('-->')
        if arrow_idx != -1:
            newline_idx = block_str.find('\n', arrow_idx)
            if newline_idx != -1:
                text_block_str = block_str[newline_idx + 1:].strip()
                if '<' in text_block_str:
                    text_block_str = HTML_TAG_PATTERN.sub('', text_block_str)
                if text_block_str:
                    blocks_list.append(text_block_str)
    return blocks_list

print("Method Split:", timeit.timeit(lambda: method_split(content_str), number=100))
print("Method Find:", timeit.timeit(lambda: method_find(content_str), number=100))
