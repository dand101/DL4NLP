# Multilingual Polarization Detection (POLAR)

This repository contains the code and documentation for our project on **multilingual polarization detection**, developed as part of the **DL4NLP** course and in the context of the **SemEval-2026 POLAR shared task**.

The project studies transformer-based approaches to polarization detection under strong class imbalance and cross-lingual variation, supported by exploratory data analysis and systematic experimentation.

---

## Repository Structure

The project code is organized as follows:

- `data/`  
  Contains the datasets used in the project.

- `exploratory_analysis/`  
  Contains exploratory data analysis (EDA) code.  
  This directory includes analyses of label distributions, language-specific properties, and text statistics that informed modeling choices.

- `tests/`  
  Contains the **main experimental code** for the task.  
  This directory includes our modeling attempts, training pipelines, evaluation logic, and experiments with different architectures, loss functions, and data augmentation strategies,  the core implementation used for the task.  

---

## Documentation

- **Project Report (PDF):**  
  https://github.com/dand101/DL4NLP/blob/main/POLAR.pdf

- **State-of-the-Art Review:**  
  https://docs.google.com/document/d/17oeZNygpwhtQxuvceIjz7MUzhnmF3DgdGHC41HU-3iI/edit?usp=sharing

- **Exploratory Data Analysis Summary:**
https://docs.google.com/document/d/1r3XVTzXW6LRaadpE8rL-1ObVf02d6re--gQixAptUGA/edit?tab=t.0#heading=h.bknethc48rew

- **Extended Project Documentation:**  
  https://docs.google.com/document/d/1r3XVTzXW6LRaadpE8rL-1ObVf02d6re--gQixAptUGA/edit

---

## Code

The full project code is available here:

https://github.com/dand101/DL4NLP/tree/main/Project
