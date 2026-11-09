# xai-ig-shap-roberta
Evaluating the reliability of Integrated Gradients and SHAP in explaining RoBERTa predictions for sentiment classification and dimensional emotion prediction.

# Zuverlässigkeit von Integrated Gradients und SHAP bei RoBERTa

## Forschungsfrage
Wie zuverlässig erklären Integrated Gradients (IG) und SHAP die Vorhersagen eines
Transformer-Modelle bei der Sentimentklassifikation und bei der
Vorhersage dimensionaler Emotionen?

- Übereinstimmung, Faithfulness und Robustheit der Erklärungen auf SST-2.
- Unterscheidet sich die Zuverlässigkeit bei Valence und Arousal?

## Hypothesen
- IG und SHAP markieren nur teilweise dieselben Tokens, in unterschiedlicher Reihenfolge.
- Erklärungen ändern sich bei kleinen Textänderungen, auch wenn die Vorhersage gleich bleibt.
- Bei Arousal sind Erklärungen weniger übereinstimmend und instabiler als bei Valence.

## Struktur
- src/: wiederverwendbarer Code
- notebooks/: Experimente
- results/: Ergebnisdateien
- report/: Textentwurf

## Setup
pip install -r requirements.txt
