# 🤖 AI Maintenance Assistant - Terminal Setup Guide

This is a **terminal-based** AI maintenance assistant that runs directly in your command line.

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key in .env file

## 🚀 Quick Start

### 1. Activate Virtual Environment

```bash
cd "c:\Users\oyvind.asplin\MyPhytonCode\Hydro SI\AI_Agent_with_RAG"
venv\Scripts\activate
```

### 2. Start Terminal Chat

```bash
python main.py
```

This starts an interactive terminal chat directly - no frontend, no web server needed!

## 🎯 Usage

1. **Run the command:** `python main.py`
2. **Ask questions** like:
   - "Show me TAPPEVOGN NR 73"
   - "Find maintenance procedures for pumps"
   - "What is the hierarchy under location 80203018?"
3. **Type 'quit', 'exit', or 'bye'** to end the conversation

## 🔧 Features

- **RAG Search**: Search internal maintenance documents  
- **Technical Locations**: Find equipment and hierarchy information
- **Source Tracking**: See which documents were used
- **Tool Usage Display**: Know which search method was used
- **Chat History**: Maintains conversation context
- **Save Output**: Save responses to text files

## 🛠️ Troubleshooting

### Common Issues

- **Check .env file**: Make sure OPENAI_API_KEY is set
- **Hierarchy database**: Run `python create_hierarchy_database.py` if needed
- **RAG database**: Run `python create_rag_database.py` if needed

### Commands

- Type `quit`, `exit`, or `bye` to end conversation
- Ctrl+C to force quit

## 📁 Project Structure

```text
AI_Agent_with_RAG/
├── main.py                    # Terminal chat interface
├── tools.py                   # Core AI tools (simplified)
├── requirements.txt           # Python dependencies only
├── create_hierarchy_database.py
├── create_rag_database.py
├── hierarchy/                 # Technical location data
├── data/                      # RAG documents
└── chroma/                    # Vector database
```

## 🎉 Benefits of Terminal Version

- **Simple**: No web setup required
- **Fast**: Direct Python execution
- **Clean**: No frontend dependencies
- **Portable**: Works anywhere Python runs
- **Focused**: Pure chat experience
- **Lightweight**: Minimal dependencies
- **Reliable**: No frontend complexity
