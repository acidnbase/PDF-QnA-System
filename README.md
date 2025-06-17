# PDF-QnA-System

Here's a comprehensive README.md file for your PDF QA System project:

```markdown
# PDF Question-Answering System 📚

A powerful PDF Question-Answering system built with Langchain, OpenAI, FAISS, and Streamlit. This application allows users to upload PDF documents and ask questions about their content using natural language, leveraging the power of Large Language Models and vector similarity search.


## 🌟 Features

- 📁 Multiple PDF document upload support
- 💬 Interactive chat interface
- 🔍 Semantic search using FAISS
- 🧠 Powered by OpenAI's GPT models
- 📊 Source document tracking
- 💾 Conversation memory
- 📱 Responsive web interface

## 🛠️ Technologies Used

- [Langchain](https://python.langchain.com/) - Framework for developing LLM applications
- [OpenAI](https://openai.com/) - Large Language Model provider
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Streamlit](https://streamlit.io/) - Web application framework
- [PyPDF2](https://pypdf2.readthedocs.io/) - PDF processing

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PDF-QnA-System.git
cd pdf-qa-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your web browser to use the application.

## 📖 How to Use

1. **Upload PDFs**: Use the sidebar to upload one or more PDF documents
2. **Process Documents**: Click "Process PDFs" to extract and embed the document content
3. **Ask Questions**: Type your questions in the chat input at the bottom
4. **View Sources**: Expand the "View Source Documents" section to see relevant source content
5. **Chat History**: View the full conversation history in the main chat interface

## 🎯 Features in Detail

### Document Processing
- Recursive text splitting for optimal chunk size
- Overlapping chunks to maintain context
- PDF text extraction with formatting preservation

### Question Answering
- RAG (Retrieval-Augmented Generation) implementation
- Conversational memory for context awareness
- Source document tracking and display

### User Interface
- Clean and intuitive chat interface
- Real-time response generation
- Document processing status indicators
- Error handling and user feedback

## 🔧 Configuration

You can modify the following parameters in the code:

```python
# Text splitting parameters
chunk_size=1000
chunk_overlap=200

# LLM parameters
temperature=0.7
model_name='gpt-3.5-turbo'
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


