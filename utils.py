import html

def clean_text(text):
    text = html.unescape(text)
    text = text.replace("\\n", " ").replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()