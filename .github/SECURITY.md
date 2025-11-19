# 🔒 Security Policy

Merci de votre intérêt pour la sécurité du projet **NBA Awards Predictor**.  
Ce document décrit les bonnes pratiques de sécurité du dépôt ainsi que la procédure à suivre pour signaler une vulnérabilité.

---

## 🛡️ Sécurisation du dépôt

Ce dépôt utilise les fonctionnalités de sécurité gratuites fournies par GitHub pour les projets publics :

### ✅ Dependabot Alerts
Le dépôt active **Dependabot** pour :
- détecter les failles dans les dépendances Python (`requirements.txt`),
- proposer automatiquement des mises à jour sécurisées.

### 🔍 Secret Scanning
GitHub scanne automatiquement le dépôt pour détecter :
- clés API,
- tokens,
- mots de passe accidentellement commités.

### 🚫 Push Protection
La fonctionnalité **Push Protection** empêche d'envoyer par mégarde :
- clés secrètes,
- identifiants sensibles,
- tokens personnels.

Si un secret est détecté lors d'un push, GitHub bloque la tentative.

### 🧪 Code Scanning (CodeQL)
Le dépôt peut utiliser **CodeQL** pour analyser le code Python et détecter des vulnérabilités potentielles :
- injections,
- erreurs de logique,
- failles courantes.

---

## 📣 Signalement d’une vulnérabilité

Merci de suivre une **divulgation responsable**.

### 🚫 Ne pas ouvrir un issue public
Les vulnérabilités ne doivent **pas** être publiées sous forme d’issue publique afin d’éviter leur exploitation.

### 📫 Contact privé
Veuillez signaler toute vulnérabilité via :

**👉 Email : luc.renaud.dev@gmail.com**

Je réponds généralement sous 48 heures.

### 🔐 Private Vulnerability Reporting
Vous pouvez également utiliser le canal privé GitHub :

👉 **Security → “Private vulnerability reporting” → “Report a vulnerability”**

Cela permet une discussion sécurisée et un suivi structuré.

---

## 🧭 Portée

Les composants concernés par cette politique :
- scripts Python,
- modules de data engineering & ML,
- pipelines d’évaluation,
- fichiers de configuration liés à l’IA.

Données exclues :
- datasets externes,
- fichiers générés localement.

---

## 🛠 Processus de résolution

Lorsqu'une vulnérabilité est signalée :
1. Analyse du problème (24–48h).  
2. Reproduction et validation.  
3. Développement d’un correctif.  
4. Publication d’une version patchée si nécessaire.  
5. Crédit optionnel du chercheur ayant aidé (sur demande).

---

## 👍 Bonnes pratiques pour les contributeurs

- Ne stockez jamais :
  - secrets API,
  - clés privées,
  - tokens personnels.
- Utilisez des variables d’environnement.
- Ne lancez pas de scraping intensif ou automatisé sur des sites tiers.
- Gardez vos dépendances à jour.

---

Merci pour votre aide dans l’amélioration de la sécurité de ce projet !  
Pour toute question, contactez : **luc.renaud.dev@gmail.com**
