from groq import Groq

client = Groq(api_key="PUT_API_KEY")

def explain(text, label, confidence):
    prompt = f"Explain why this SMS is {label}: {text}"

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content