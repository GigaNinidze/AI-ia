import sentencepiece as spm, json

sp = spm.SentencePieceProcessor()
sp.load("/Users/us1ndiso/Desktop/ქართული-ენა/tokens/ge_tokenizer.model")

count = 0
with open("/Users/us1ndiso/Desktop/ქართული-ენა/corpus/out/clean_ge.jsonl", "r") as f:
    for i, line in enumerate(f, start=1):
        obj = json.loads(line)
        text = obj["text"]
        ids = sp.encode(text, out_type=int)
        count += len(ids)

        if i % 10000 == 0:  # show progress every 10k lines
            print(f"Processed {i:,} lines, tokens so far: {count:,}")

print(f"Total tokens in corpus: {count:,}")

#Total tokens in corpus: 14,069,351
