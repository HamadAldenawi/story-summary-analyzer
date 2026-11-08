# Story Summary Analyzer — extraction d'informations sémantiques dans des résumés oraux d'enfants

Projet personnel (novembre 2026) — Hamad Aldenawi, M2 Intelligence Artificielle, CERI / Avignon Université.

**Objectif :** détecter automatiquement quels critères d'une grille d'évaluation (personnage, problème, solution, fin…)
sont présents dans le résumé oral d'une histoire fait par un enfant, à partir de la **transcription automatique**
(qui peut contenir des erreurs de reconnaissance vocale).

## Données
- `data/stories.json` : 3 histoires courtes **écrites pour ce projet** (niveau cycle 3) + une grille de 5 critères chacune.
- `data/summaries.json` : 12 résumés **synthétiques** au style oral d'enfant (rédigés pour ce projet, pas de vraies données d'enfants),
  annotés à la main avec les critères présents.

## Méthodes
| Méthode | Idée |
|---|---|
| `keyword_detect` | Baseline : un critère est présent si un mot-clé de la grille apparaît dans la transcription. |
| `RagMatcher` | Approche **RAG** légère : chaque critère est enrichi avec la phrase la plus pertinente du texte source (récupération), puis comparé à des fenêtres glissantes du résumé avec un TF-IDF de **n-grammes de caractères** (plus robuste aux fautes de transcription). Seuil appris en *leave-one-story-out*. |
| `llm_prompt.py` | (Optionnel) prompts pour un LLM local (Qwen / Llama via Ollama), stratégies « sans contexte » vs « RAG ». **Non utilisé dans les résultats ci-dessous.** |

`asr_noise.py` simule des erreurs de reconnaissance (mots supprimés, mots mal reconnus) pour mesurer leur impact.

## Résultats (précision, rappel, F1) — `python evaluate.py`
| Bruit ASR simulé | Baseline mots-clés (F1) | RAG n-grammes (F1) |
|---|---|---|
| 0 % | 0.857 | 0.759 |
| 10 % | 0.820 | 0.806 |
| 20 % | 0.793 | 0.825 |
| 30 % | 0.772 | 0.738 |

**Lecture honnête :** sur des transcriptions propres, la baseline mots-clés est meilleure (mais ses mots-clés ont été écrits
en connaissant le style des résumés, ce qui l'avantage). Quand le bruit de transcription augmente, ses performances
baissent plus vite et l'approche RAG devient meilleure vers 20 % de bruit. Le corpus est **très petit** (12 résumés) :
ces chiffres illustrent la méthode, ils ne sont pas statistiquement solides.

## Démo
Démo interactive : https://hamadaldenawi.github.io/projets.html

## Pistes
Comparer avec des LLM (Qwen, Llama) et différentes stratégies de prompt ; ajouter de la reconnaissance d'entités nommées
(spaCy, CamemBERT) ; tester sur de vraies transcriptions avec annotations d'experts.

## Lancer
```bash
pip install -r requirements.txt
python evaluate.py
```
