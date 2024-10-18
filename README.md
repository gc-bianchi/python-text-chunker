# python-text-chunker

## Learnings from Semantic Chunker Project
### Rediscovering List Comprehensions
I was reminded of how powerful and concise list comprehensions are in Python. They were instrumental in making the text preprocessing efficient and readable.

### Choosing an Embedding Solution
I decided not to use OpenAIEmbeddings from langchain.embeddings due to cost concerns. After some research, I found that SentenceTransformer was an excellent free alternative that provided robust embeddings for my needs.

### Learning About Embeddings
I learned more about embeddings and found it fascinating how they are used by LLMs to break down text in order to understand its context and meaning. It gave me a deeper appreciation for how these models "understand" language.

### Guidance from Greg Kamradt
I watched Greg Kamradt's YouTube video a few times to help guide my implementation. Although some steps seemed more complex than necessary for my use case, his overall explanation was indispensable for understanding the process.

### Threshold Determination
My main "issue" with this approach is determining the threshold for chunking. Using the 95th percentile feels a bit arbitrary. However, after researching why we use concepts like 95% confidence intervals in statistics, I learned that it "strikes a balance between being confident in the result while maintaining a reasonable range." So, while I'm okay with it for now, I still feel there might be a better way to determine the threshold. I made the threshold a parameter in the script so users can experiment with different values and see how the resulting chunks change.



