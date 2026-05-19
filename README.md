# 🏥 MedGuide AI — RAG-Powered Health Chatbot

<div align="center">

![MedGuide Banner](https://img.shields.io/badge/MedGuide-AI%20Health%20Assistant-2a69ac?style=for-the-badge&logo=health&logoColor=white)
![Claude API](https://img.shields.io/badge/Powered%20by-Claude%20Sonnet%204-orange?style=for-the-badge)
![RAG](https://img.shields.io/badge/Architecture-RAG%20(Retrieval--Augmented%20Generation)-green?style=for-the-badge)
![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB?style=for-the-badge&logo=react)

**An interactive AI health assistant using Retrieval-Augmented Generation (RAG) with cosine similarity retrieval and Claude Sonnet as the LLM backend.**

[Live Demo](#) · [Features](#features) · [Architecture](#architecture) · [Setup](#setup) · [Experimentation](#experimentation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Knowledge Base](#knowledge-base)
- [RAG Pipeline](#rag-pipeline)
- [Setup & Installation](#setup--installation)
- [Experimentation Guide](#experimentation-guide)
- [API Reference](#api-reference)
- [Disclaimer](#disclaimer)

---

## Overview

MedGuide AI is a domain-specific health chatbot built with **Retrieval-Augmented Generation (RAG)**. Instead of relying solely on LLM parametric knowledge, it first retrieves the most relevant documents from a curated medical knowledge base using TF-IDF + cosine similarity, then feeds those documents as context to Claude Sonnet to generate accurate, grounded responses.

This project demonstrates:
- RAG pipeline implementation from scratch (no LangChain dependency)
- Custom embedding/retrieval with keyword + cosine similarity scoring
- Adjustable retrieval parameters (top-K, temperature)
- Clean, production-grade React UI
- Transparent AI — users can see exactly which documents were retrieved

---

## Features

| Feature | Description |
|---|---|
| 🔍 **RAG Retrieval** | TF-IDF + cosine similarity over 18 health knowledge documents |
| 🧠 **Claude Sonnet LLM** | `claude-sonnet-4-20250514` for high-quality generation |
| 📊 **Relevance Scores** | Every response shows retrieved docs + match percentage |
| ⚙️ **Tunable Parameters** | Adjust top-K (1–5) and temperature (0–1.0) in real time |
| 💬 **Quick Prompts** | Pre-built queries to explore the system immediately |
| 🚨 **Safety-first** | Emergency conditions trigger immediate 911/professional referral |
| 📱 **Responsive UI** | Works on desktop and mobile |
| 🩺 **18 Health Categories** | Symptoms, Nutrition, Exercise, Sleep, Mental Health, Medications, and more |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RETRIEVER MODULE                               │
│                                                                  │
│  1. Keyword Matching (weight: 0.4)                               │
│     └─ Exact match against doc.keywords array                    │
│                                                                  │
│  2. Tag Matching (weight: 0.3)                                   │
│     └─ Exact match against doc.tags array                        │
│                                                                  │
│  3. Cosine Similarity (weight: 0.3)                              │
│     └─ TF-IDF vectors: query vs (question + keywords)            │
│                                                                  │
│  Score = 0.4×keyword + 0.3×tag + 0.3×cosine                     │
│  Filter: score > threshold (0.05)                                │
│  Return: top-K documents sorted by score                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │  Retrieved context (top-K docs)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROMPT ASSEMBLY                                │
│                                                                  │
│  System: Medical AI persona + safety rules                       │
│  Context: [Document 1 | Category | Relevance%] Q: ... A: ...    │
│  User: Original query                                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│           CLAUDE SONNET 4 (claude-sonnet-4-20250514)            │
│                                                                  │
│  Generates grounded response based on retrieved context          │
│  Temperature: configurable (default 0.3 for high accuracy)       │
│  Max tokens: 1000                                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE TO USER                              │
│   + Retrieved docs panel (expandable, shows category & score)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Knowledge Base

The knowledge base (`HEALTH_KNOWLEDGE_BASE`) contains **18 curated health documents** across these categories:

| Category | Documents | Example Topics |
|---|---|---|
| Symptoms | 2 | Fever management, Headache/migraine |
| Nutrition | 1 | Balanced diet, WHO recommendations |
| Exercise | 1 | WHO physical activity guidelines |
| Sleep | 1 | Sleep duration by age, sleep hygiene |
| Mental Health | 1 | Anxiety management, CBT |
| Chronic Disease | 2 | Diabetes signs, Hypertension management |
| First Aid | 1 | Burns, cuts, wound care |
| Medications | 1 | Safe medication practices, interactions |
| Prevention | 2 | Adult vaccinations, cancer screenings |
| Digestive Health | 1 | Bloating, nausea, BRAT diet |
| Hydration | 1 | Daily water intake, dehydration signs |
| Emergency | 1 | Heart attack signs, FAST stroke protocol |
| Skin Health | 1 | Skincare, melanoma ABCDEs |
| Eye Health | 1 | Eye exams, 20-20-20 rule |
| Respiratory | 1 | Persistent cough, asthma management |

Each document contains:
```json
{
  "id": 1,
  "category": "Symptoms",
  "tags": ["fever", "temperature"],
  "question": "What should I do if I have a fever?",
  "answer": "...",
  "keywords": ["fever", "temperature", "hot", "chills", "sweating"]
}
```

### Extending the Knowledge Base

Add new documents to the `HEALTH_KNOWLEDGE_BASE` array in `src/App.jsx`:

```js
{
  id: 19,
  category: "Your Category",
  tags: ["primary", "search", "terms"],
  question: "The question this document answers?",
  answer: "Detailed, evidence-based answer...",
  keywords: ["keyword1", "keyword2", "keyword3", "synonym1"]
}
```

---

## RAG Pipeline

### Retrieval Algorithm

The custom retriever uses a **hybrid scoring** approach:

```javascript
function retrieveDocs(query, topK = 3, threshold = 0.05) {
  const scored = HEALTH_KNOWLEDGE_BASE.map(doc => {
    // Component 1: Keyword overlap (highest weight — domain-specific)
    const keywordScore = doc.keywords.filter(k => query.includes(k)).length * 0.4;
    
    // Component 2: Tag overlap (medium weight — categorical)
    const tagScore = doc.tags.filter(t => query.includes(t)).length * 0.3;
    
    // Component 3: Cosine similarity on TF-IDF vectors
    const cosScore = cosineSimilarity(query, doc.question + " " + doc.keywords.join(" ")) * 0.3;
    
    return { ...doc, score: keywordScore + tagScore + cosScore };
  });
  
  return scored
    .filter(d => d.score > threshold)   // Remove irrelevant
    .sort((a, b) => b.score - a.score)  // Rank by relevance
    .slice(0, topK);                     // Return top-K
}
```

### Cosine Similarity (Bag-of-Words)

```javascript
function cosineSimilarity(str1, str2) {
  // Tokenize
  const words1 = str1.toLowerCase().split(/\W+/).filter(Boolean);
  const words2 = str2.toLowerCase().split(/\W+/).filter(Boolean);
  
  // Build vocabulary
  const vocab = [...new Set([...words1, ...words2])];
  
  // Create TF vectors
  const vec1 = vocab.map(w => words1.filter(x => x === w).length);
  const vec2 = vocab.map(w => words2.filter(x => x === w).length);
  
  // Compute cosine similarity
  const dot = vec1.reduce((s, v, i) => s + v * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((s, v) => s + v * v, 0));
  const mag2 = Math.sqrt(vec2.reduce((s, v) => s + v * v, 0));
  
  return mag1 && mag2 ? dot / (mag1 * mag2) : 0;
}
```

---

## Setup & Installation

### Prerequisites

- Node.js 18+
- An Anthropic API key ([get one here](https://console.anthropic.com))

### Option A: Claude.ai Artifact (Zero Setup)

This app is designed to run directly as a Claude.ai Artifact. Paste the `App.jsx` code into a new React artifact and it works immediately — the Anthropic API key is handled automatically.

### Option B: Local Development (Vite + React)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/medguide-ai-chatbot.git
cd medguide-ai-chatbot

# 2. Install dependencies
npm install

# 3. Create .env file
echo "VITE_ANTHROPIC_API_KEY=your_api_key_here" > .env

# 4. Update the fetch call in App.jsx to include your key
# In the fetch headers, add:
# "x-api-key": import.meta.env.VITE_ANTHROPIC_API_KEY,
# "anthropic-version": "2023-06-01",
# "anthropic-dangerous-direct-browser-access": "true"

# 5. Start development server
npm run dev
```

### Option C: Deploy to Vercel/Netlify

```bash
# Build for production
npm run build

# Deploy (Vercel example)
npx vercel --prod
```

> ⚠️ **Security Note**: Never expose API keys in client-side code for production. Use a backend proxy server or serverless function to make API calls.

### Recommended Project Structure

```
medguide-ai-chatbot/
├── src/
│   ├── App.jsx           # Main chatbot component (RAG + UI)
│   ├── main.jsx          # React entry point
│   └── index.css         # Global styles (optional)
├── public/
│   └── favicon.ico
├── index.html
├── package.json
├── vite.config.js
├── .env                  # API key (never commit this!)
├── .gitignore
└── README.md
```

### package.json

```json
{
  "name": "medguide-ai-chatbot",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.0.0",
    "vite": "^5.0.0"
  }
}
```

---

## Experimentation Guide

The Settings panel (⚙️) lets you tune the RAG pipeline in real time:

### Parameter 1: Top-K Retrieved Documents

| K Value | Effect | Best For |
|---|---|---|
| K=1 | Most focused, lowest hallucination risk | Simple, specific queries |
| K=2–3 (default) | Balanced accuracy and breadth | General health questions |
| K=4–5 | Broader context, may include tangential info | Complex multi-symptom queries |

**Experiment**: Ask "I have a fever and headache" — compare K=1 vs K=3 responses.

### Parameter 2: Temperature

| Temperature | Effect | Best For |
|---|---|---|
| 0.0–0.2 | Very deterministic, conservative | Medical facts, medication info |
| 0.3 (default) | Balanced — accurate but readable | General health Q&A |
| 0.5–0.7 | More varied phrasing | Lifestyle and wellness advice |
| 0.8–1.0 | Creative, may drift | Not recommended for health |

**Experiment**: Ask "How can I sleep better?" at temperature 0.1 vs 0.8.

### Retrieval Threshold

In code, adjust the threshold in `retrieveDocs(query, topK, threshold)`:
- `0.01` — returns more documents (higher recall, lower precision)
- `0.05` (default) — balanced
- `0.15` — only returns very confident matches (high precision)

### What to Measure

Track response quality across settings:
1. **Relevance**: Does the answer address the actual question?
2. **Groundedness**: Is the answer traceable to retrieved documents?
3. **Completeness**: Are all aspects of the question covered?
4. **Safety**: Does the response appropriately escalate emergencies?

---

## API Reference

### Anthropic API Call

```javascript
const response = await fetch("https://api.anthropic.com/v1/messages", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    // Add x-api-key for non-Artifact deployments
  },
  body: JSON.stringify({
    model: "claude-sonnet-4-20250514",  // Best accuracy model
    max_tokens: 1000,
    system: SYSTEM_PROMPT,              // Medical AI persona + safety rules
    messages: [
      ...conversationHistory,
      { role: "user", content: ragPrompt }  // Context + query
    ],
    temperature: 0.3  // Configurable
  })
});
```

### RAG Prompt Structure

```
RETRIEVED CONTEXT DOCUMENTS:
[Document 1 | Category: Symptoms | Relevance: 87%]
Q: What should I do if I have a fever?
A: A fever (temperature above 38°C/100.4°F)...

[Document 2 | Category: Medications | Relevance: 45%]
Q: What should I know about taking medications safely?
A: Always: take as prescribed...

USER QUESTION: I have a 39°C fever, should I take medication?

Using the context above, provide a helpful, accurate health response...
```

---

## Sample Queries to Test

```
Basic Symptoms:
"I have a high fever, what should I do?"
"My head has been hurting for 2 days"

Chronic Conditions:
"What are the early warning signs of diabetes?"
"How do I know if I have high blood pressure?"

Lifestyle:
"How much exercise should I get each week?"
"What should I eat for a balanced diet?"

Mental Health:
"I'm feeling very anxious, how can I calm down?"
"I can't sleep at night, what can I do?"

Emergencies:
"My dad is having chest pain"
"What does a stroke look like?"

Multi-topic (tests K=4+ retrieval):
"I'm tired all the time and also very thirsty"
"I have stomach pain, nausea, and a fever"
```

---

## Technologies Used

| Technology | Purpose |
|---|---|
| **React 18** | UI framework |
| **Claude Sonnet 4** | LLM for generation (`claude-sonnet-4-20250514`) |
| **Custom RAG** | Retrieval: TF-IDF + cosine similarity + keyword matching |
| **Anthropic API** | `/v1/messages` endpoint |
| **Vite** | Build tool (for local development) |
| **Vanilla CSS** | Styling (no external CSS library) |

---

## Roadmap / Possible Extensions

- [ ] 🔗 Connect to real medical knowledge base (MedlinePlus API, WHO API)
- [ ] 🔢 Add proper dense embeddings (sentence-transformers via HuggingFace Inference API)
- [ ] 📄 Support PDF upload for personal health records
- [ ] 🌍 Multi-language support
- [ ] 📊 Analytics dashboard for query patterns
- [ ] 🔒 Backend proxy for secure API key handling
- [ ] 🗃️ Vector database integration (Pinecone, Chroma, Weaviate)
- [ ] 📱 React Native mobile app

---

## ⚠️ Disclaimer

**MedGuide AI is not a substitute for professional medical advice, diagnosis, or treatment.**

- This application is for **educational and informational purposes only**
- Always consult a qualified healthcare provider for medical concerns
- For emergencies, call **911** (US) or your local emergency number immediately
- The knowledge base reflects general health guidelines and may not apply to all individuals
- Information may not reflect the most current medical research or guidelines

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-dermatology-docs`)
3. Add your health documents with proper keywords and tags
4. Test retrieval accuracy with at least 5 sample queries
5. Submit a pull request with a description of changes

---

<div align="center">
Built with ❤️ using Claude API · RAG Architecture · React

**⭐ Star this repo if it helped you learn about RAG!**
</div>
