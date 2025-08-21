import os
import google.generativeai as genai

key = os.getenv("GEMINI_API_KEY")
assert key, "GEMINI_API_KEY is not set"
genai.configure(api_key=key)

# посмотрим, какие FLASH-модели реально доступны
models = [m.name for m in genai.list_models() if "flash" in m.name]
print("FLASH models seen:", models[:10])

# попробуем по очереди
for name in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
    try:
        m = genai.GenerativeModel(name)
        r = m.generate_content("Ответь одним словом: PONG")
        print(name, "OK →", (r.text or "").strip())
    except Exception as e:
        print(name, "FAILED →", e)
