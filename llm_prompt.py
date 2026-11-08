"""Optional LLM-based extraction (e.g. Qwen or Llama served locally by Ollama).
Not used for the reported results - provided to compare prompting strategies later."""
import json, urllib.request

def build_prompt(story, summary, strategy="rag"):
    crit = "\n".join(f"- {c}: {v['desc']}" for c, v in story["grid"].items())
    ctx = f"Texte source :\n{story['text']}\n\n" if strategy == "rag" else ""
    return (f"{ctx}Grille d'évaluation :\n{crit}\n\nRésumé oral d'un enfant (transcription automatique, "
            f"peut contenir des erreurs) :\n{summary}\n\nPour chaque critère, réponds par un JSON "
            '{"critère": true/false}.')

def ask_ollama(prompt, model="qwen2.5:7b", url="http://localhost:11434/api/generate"):
    req = urllib.request.Request(url, json.dumps({"model": model, "prompt": prompt, "stream": False,
                                 "format": "json"}).encode(), {"Content-Type": "application/json"})
    return json.loads(json.loads(urllib.request.urlopen(req).read())["response"])
