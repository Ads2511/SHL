# SHL Assessment Recommendation API  

## 🚀 Overview  
The **SHL Assessment Recommendation API** helps users find the most relevant SHL assessments based on job descriptions or queries. This project utilizes **Sentence-BERT (SBERT)** for semantic similarity matching and is **deployed on Hugging Face Spaces using Docker**.  

🔗 **Live Deployment**: [SHL Assessment Recommendation API](https://huggingface.co/spaces/Ads2511/SHL_Recommedation)  

## 🛠️ Features  
- 🔍 **Semantic Search**: Uses **SBERT embeddings** to match job descriptions with relevant assessments.  
- ⚡ **Fast API Response**: Built with **FastAPI** for quick and efficient responses.  
- 📊 **Precomputed Embeddings**: Ensures rapid similarity calculations using **cosine similarity**.  
- 📦 **Dockerized Deployment**: Hosted on **Hugging Face Spaces** for seamless cloud deployment.  

---

## 📂 Dataset & Preprocessing  
### **Dataset Details**  
The dataset was obtained from **SHL’s official website** and includes:  
- 📝 **Assessment Name**  
- 📄 **Job Levels**  
- 📑 **Description**  
- 🌍 **Remote Testing Availability**  
- 🔁 **Adaptive or Not**  
- ⏳ **Assessment Duration**  
- 🔢 **Precomputed SBERT Embeddings**  

### **Preprocessing Steps**  
1. **Web Scraping**: Collected job descriptions and assessment details from SHL.  
2. **Data Cleaning**: Removed irrelevant information and standardized text formats.  
3. **Embedding Generation**: Used `sentence-transformers` to convert descriptions into **768-dimensional embeddings**.  
4. **Similarity Calculation**: Applied **cosine similarity** for efficient recommendation.  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python 🐍  
- **Libraries Used**:  
  - `sentence-transformers` – For SBERT embeddings  
  - `transformers` – Hugging Face model integration  
  - `torch` – PyTorch backend for deep learning  
  - `pandas` – Data handling  
  - `scikit-learn` – Cosine similarity computation  
  - `FastAPI` – API development  
  - `Uvicorn` – ASGI server for running the API  
  - `Docker` – Containerization for deployment  

---

## 📡 API Usage  
### **Endpoint**  
