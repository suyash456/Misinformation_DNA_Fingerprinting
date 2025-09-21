# 🧬 Multi-Modal Misinformation DNA Fingerprinting

Track how misinformation mutates as it spreads across platforms.

This project implements a **content genealogy tracker** to detect, analyze, and visualize the evolution (“mutations”) of text and image-based content as it propagates across social networks such as **WhatsApp → Facebook → Twitter**.  
It combines **graph-based content lineage** with **deepfake detection heuristics** to give users a clear **Mutation Map** of how and where distortions were introduced.

---

## 🚀 Features
- **Content Genealogy Tracking**  
  Uses **NetworkX** graphs to track parent–child relationships between similar pieces of content across platforms.

- **Multi-Modal Deepfake Detection**  
  - **Text:** Writing-style mutation analysis (similarity, tone changes, added/removed words).  
  - **Images:** Error Level Analysis (edge density, suspicious region detection) and color distribution analysis for manipulation scoring.

- **Interactive Mutation Map**  
  Generates a timeline showing:
  - Platform transitions  
  - Text changes (added/removed words, tone)  
  - Image manipulation scores

---

## 🏗️ Project Structure
```
├── app.py # Flask web application
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```



## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/suyash456/Misinformation_DNA_Fingerprinting.git
   cd misinformation-dna

