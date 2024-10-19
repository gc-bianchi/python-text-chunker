# python-text-chunker project

### Learnings from Semantic Chunker Project
#### Rediscovering List Comprehensions
I was reminded of how powerful and concise list comprehensions are in Python. They were instrumental in making the text preprocessing efficient and readable.

#### Choosing an Embedding Solution
I decided not to use OpenAIEmbeddings from langchain.embeddings due to cost concerns. After some research, I found that SentenceTransformer was an excellent free alternative that provided robust embeddings for my needs.

#### Learning About Embeddings
I learned more about embeddings and found it fascinating how they are used by LLMs to break down text in order to understand its context and meaning. It gave me a deeper appreciation for how these models "understand" language.

#### Guidance from Greg Kamradt
I watched Greg Kamradt's YouTube video a few times to help guide my implementation. Although some steps seemed more complex than necessary for my use case, his overall explanation was indispensable for understanding the process.

#### Threshold Determination
My main "issue" with this approach is determining the threshold for chunking. Using the 95th percentile feels a bit arbitrary. However, after researching why we use concepts like 95% confidence intervals in statistics, I learned that it "strikes a balance between being confident in the result while maintaining a reasonable range." So, while I'm okay with it for now, I still feel there might be a better way to determine the threshold. I made the threshold a parameter in the script so users can experiment with different values and see how the resulting chunks change.

### Learnings from Agentic Chunker Project

#### Limitations of Available Models: 
I wasn't entirely sure if I was approaching this project correctly, primarily because I had to work within the constraints of available free models. I ended up relying on previous code to determine sentence similarity rather than using a specialized library for this purpose. However, upon reflection, I wonder if my approach was fundamentally different from the "better" models and libraries often used. Those tools likely have similar semantic logic baked in, just abstracted away for ease of use and better performance. My method might not be as optimized, but it was insightful to build that logic manually.

#### Challenges with Prompt Definition
Crafting effective prompts for non-advanced models was a significant challenge. Models like GPT-2 or gpt-neo require more precise and explicit instructions compared to more advanced models, which are better at interpreting nuance. I learned that finding the right balance between being explicit and concise in the prompt was crucial.

#### Learning Through Iteration
Re-watching Greg Kamradt's video helped solidify my understanding. Having completed the semantic project before, I was able to better grasp the intention behind developing an agentic chunker. This iterative learning process was valuable in deepening my understanding of chunking and summarization concepts.

#### Summarizing and Titling Each Chunk During Updates
In Greg's tutorial, he generated both a summary and a title each time a chunk was updated. I chose not to do this, as it felt redundant and unnecessary, especially given the computational overhead involved. Instead, I summarized and titled the chunks after all of them were formed. This approach also helped improve performance. I think Greg mostly did this in his tutorial to provide viewers with immediate feedback, showing what was happening behind the scenes, but for my purposes, it was better to omit that step.

