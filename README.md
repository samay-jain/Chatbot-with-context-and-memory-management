# 🧠 Local Chatbot Using Ollama API

This is a simple command-line chatbot built in Python that uses the [Ollama](https://ollama.com/) API to interact with large language models (such as LLaMA). It allows you to chat with the model, save and load conversations, and summarize previous messages to manage token usage.

---

## 🚀 Features

- 🤖 Interactive chatbot powered by local LLMs via Ollama
- 💾 Save and load conversation history to/from a file
- 🧹 Automatically or manually summarize past messages to reduce token usage
- 📡 Connects to `localhost:11434` using Ollama's REST API
- 🛠️ Easy to extend or customize

---

## 🛠 Requirements

- Python 3.7+
- [Ollama](https://ollama.com/) installed and running locally
- An Ollama-compatible model (e.g., LLaMA 3) pulled and ready

### Install required Python packages:

```bash
pip install requests
