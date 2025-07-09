import re

def split_text(text: str, max_length: int = 300) -> list[str]:
    # 以中文句号、换行符或分号为分割点进行分段
    segments = re.split(r'(?<=[。！？；\n])', text)
    
    chunks = []
    current_chunk = ""
    for segment in segments:
        if len(current_chunk) + len(segment) > max_length:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = segment
        else:
            current_chunk += segment
    if current_chunk:
        chunks.append(current_chunk)
    return [c.strip() for c in chunks if c.strip()]
