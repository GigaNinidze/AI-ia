
AI-ია.md-  "ხელოვნური ინტელექტი შიგნიდან" ვაწყობთ ქართული ენის მცირე მოდელს 
სპეციალური ტოკენაიზერით, რომელიც გამოთვლილია მხოლოდ ქართული ტექსტის, ციფრების და სიმბოლოებისთვის. მიზანია შეიქმნას მცირე ენის მოდელი რომელიც ხვდება გრამატიკის საფუძვლებს და მოკლე კონტექსტში სწორ წინადადებებს ააწყობს. 

აქ ჯერ წიგნები ჩამოვტივრთე და 200მბ ტექსტის გამოყენება ვცადე, მაგრამ შემდეგ ვიკი დამფი ჩამოვტვირთე და ფაილები გავლიტრე, მაგრამ ჯერ ბოლომდე არა. 200მბ ტექსტით შექმნილი მოდელი ცუდ შედეგებს იძლეოდა pdf-დან ცუდი კოპირების და შემდგომ ცუდი გაწმენდის გამო :(    მაგრამ 800 ვეფხისტყაოსანის ხელით გაწმენდაც არ იყო გამოსავალი :) 

websitebi.md - ზოგი უფასო ვებსაიტის ნაკრები. ხელოვნური ინტეელქტით ან მის გარაშე, უბრალოდ გამოსადეგი რესურვები ინტერნეტში. 












# Navigate to wiki folder
cd wiki

# Analyze your corpus first
python analyze_corpus.py

# Process all 99,996 articles into clean format
python process_wiki_corpus.py
# Train a better tokenizer on your large corpus
python train_tokenizer.py
# Train with enhanced architecture for your large dataset
python improved_training.py
# Or run everything automatically
python run_full_pipeline.py



# ქართული ენა - Georgian Language Model Project

A comprehensive project for training small language models on Georgian text, including educational content and practical tools for Georgian language processing.

## 🎯 Project Overview

This project contains:
- **Educational content** in Georgian explaining AI concepts
- **Corpus extraction tools** for Georgian Wikipedia
- **Training pipeline** for small Georgian language models
- **Tokenization** using SentencePiece
- **Model architecture** optimized for M3 Macs

## 📁 Project Structure

```
ქართული-ენა/
├── README.md                    # This file
├── .gitignore                  # Git ignore rules
├── AI-ია.md                    # Educational content about AI
├── პროგრამირება_ხელოვნური_ინტელექტით.md  # Programming with AI
├── ვებსაიტები.md              # Websites documentation
├── პრომპტები.md               # Prompts documentation
├── generate.py                 # Text generation script
├── counter.py                  # Utility script
├── 3D - tensor.png            # Visual aid for AI concepts
├── gradient_descent.png        # Gradient descent visualization
├── cursor-context.png          # Cursor context example
├── cursor-agent-ask.png        # Cursor agent example
├── Deerantlers.png             # Additional visual content
├── banana-ai.png               # AI visualization
├── barbieprompt.png            # Prompt example
├── collage-pattern.jpg         # Pattern visualization
├── geoflag-correct.png         # Georgian flag visualization
├── mcdanceGeorgian.png         # Georgian dance visualization
├── train/                      # Training code
│   └── train.py               # Main training script
├── corpus/                     # Corpus data
│   └── out/                   # Processed corpus files
├── tokens/                     # Tokenizer files
│   └── ge_tokenizer.model     # SentencePiece model
├── checkpoints/                # Model checkpoints
└── .qodo/                     # Development files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch
- SentencePiece
- M3 Mac (optimized for Apple Silicon)

### Installation
```bash
# Clone the repository
git clone https://github.com/GigaNinidze/AI-ia.git
cd AI-ia

# Install dependencies
pip install torch sentencepiece transformers
```

### Training a Model
```bash
# Navigate to training directory
cd train

# Run training
python train.py
```

### Generating Text
```bash
# Generate text with trained model
python generate.py
```

## 📚 Educational Content

The project includes educational materials in Georgian:

- **AI-ია.md**: Introduction to artificial intelligence concepts
- **პროგრამირება_ხელოვნური_ინტელექტით.md**: Programming with AI
- **ვებსაიტები.md**: Web development resources
- **პრომპტები.md**: AI prompts and examples

## 🛠️ Technical Details

### Model Architecture
- **Architecture**: TinyGPT (decoder-only Transformer)
- **Parameters**: ~700K (Chinchilla-optimal for 14M tokens)
- **Sequence Length**: 512 tokens
- **Vocabulary**: 32K tokens (SentencePiece)
- **Optimization**: Weight tying, no biases in LayerNorm/MLP

### Training Configuration
- **Device**: MPS (Apple Silicon) / CUDA / CPU
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 3e-4 with cosine decay
- **Warmup Steps**: 200
- **Max Steps**: 5000

### Corpus Processing
- **Source**: Georgian Wikipedia
- **Cleaning**: Unicode normalization, Georgian script filtering
- **Deduplication**: MinHash-based near-duplicate removal
- **Format**: JSONL with UTF-8 encoding

## 🎯 Goals

1. **Educational**: Teach AI concepts to Georgian speakers
2. **Practical**: Provide tools for Georgian language processing
3. **Research**: Enable small-scale language model research
4. **Cultural**: Promote Georgian language in AI/ML

## 🤝 Contributing

Contributions are welcome! Please:
- Add educational content in Georgian
- Improve the training pipeline
- Optimize for different hardware
- Add more corpus sources

## 📄 License

This project is for educational and research purposes. Please respect Wikipedia's terms of service when using the corpus extraction tools.

## 🙏 Acknowledgments

- Georgian Wikipedia contributors
- PyTorch and SentencePiece teams
- Apple for M3 optimization
- The Georgian AI community

---

*This project aims to make AI accessible to Georgian speakers while preserving and promoting the Georgian language in the digital age.*
