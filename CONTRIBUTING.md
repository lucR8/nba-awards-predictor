# Contributing to NBA Awards Predictor

Merci de votre intérêt pour contribuer à ce projet !  
Bien que le développement soit principalement personnel, les contributions externes sont possibles à condition de respecter les lignes suivantes.

---

## 🚀 Workflow de contribution

1. **Forkez** le repository
2. Créez une branche dédiée :
```bash
git checkout -b feature/nom-de-la-feature
```
3. Faites vos modifications (voir sections tests & qualité)
4. Commitez proprement avec un message clair :
    feat: ajout du calcul des percentiles  
    fix: correction d’un bug dans le chargement des CSV  
5. Ouvrez une Pull Request vers main en suivant le template fourni.
---
## 🧪 Tests

Merci de vérifier que les tests passent avant de soumettre une PR :
```bash
pytest -q
```
Si vous ajoutez une nouvelle fonctionnalité, merci d’ajouter également un test minimal dans tests/.
---

## 🧹 Qualité du code

Le projet utilise des standards simples :
- Respect de la structure projet (src/awards_predictor/*)
- Style Python PEP8 recommandé
- Pas de scraping Basketball Reference dans le repo
(pour respecter leurs conditions d’utilisation)

Optionnel mais recommandé :
```bash
flake8 src
```
---

## 📦 Structure à respecter

Merci de conserver l'organisation suivante :
```bash
src/awards_predictor/
    data/         # Chargement / validation des données
    features/     # Feature engineering
    models/       # Entraînement / sauvegarde des modèles
    evaluation/   # Métriques et validation
    viz/          # Visualisations
```

Les données brutes ne doivent pas être ajoutées au repository.
---

## 📬 Questions
Pour toute question, ouvrez une issue GitHub en suivant le template.