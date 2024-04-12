<a href="https://colab.research.google.com/github/abhijeetk597/rag-application-llama2-llamaindex-huggingface/blob/main/basic_rag_llama2_llamaindex.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# RAG System using LLAMA2, LlamaIndex and Hugging Face

## 1. Create a `data` folder to save uploaded files


```python
import os

# Create a directory to save uploaded file
directory_name = "data"

# Check if the directory exists
if not os.path.exists(directory_name):
    # Create the directory if it doesn't exist
    os.mkdir(directory_name)
```

## 2. Install dependancies

- transformers
- sentence_transformers
- torch
- einops
- accelerate
- bitsandbytes
- langchain
- llama_index
- llama-index-llms-huggingface
- llama-index-embeddings-langchain
- llama-index-embeddings-huggingface
- huggingface_hub


```python
!pip install -q transformers einops accelerate langchain bitsandbytes sentence_transformers
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.6/44.6 kB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m297.4/297.4 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m817.7/817.7 kB[0m [31m37.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m119.8/119.8 MB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m163.3/163.3 kB[0m [31m19.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m78.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m287.5/287.5 kB[0m [31m31.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m104.2/104.2 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.4/49.4 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m6.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m144.8/144.8 kB[0m [31m17.6 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
!pip install -q llama_index llama-index-llms-huggingface llama-index-embeddings-langchain llama-index-embeddings-huggingface
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.4/15.4 MB[0m [31m78.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m74.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m268.3/268.3 kB[0m [31m30.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m10.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m132.8/132.8 kB[0m [31m18.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m84.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.9/3.9 MB[0m [31m96.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m290.4/290.4 kB[0m [31m30.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m30.8/30.8 MB[0m [31m17.0 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
!pip install -q huggingface_hub
```

## 3. Login to Huggingface Hub using access_token


```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

from huggingface_hub import login
login(HF_TOKEN)
```

    Token will not been saved to git credential helper. Pass `add_to_git_credential=True` if you want to set the git credential as well.
    Token is valid (permission: write).
    Your token has been saved to /root/.cache/huggingface/token
    Login successful
    

## 4. Importing necessary packages


```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings


import torch
from transformers import BitsAndBytesConfig
```

## 5. Create `system_prompt` and `query_wrapper_prompt`


```python
system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
```

## 6. Download 8 bit Quantized LLM : Llama2_7b


```python
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16 , "quantization_config":quantization_config}
)
```
    config.json:   0%|==========| 0.00/614 [00:00<?, ?B/s]

    model.safetensors.index.json:   0%|==========| 0.00/26.8k [00:00<?, ?B/s]

    Downloading shards:   0%|==========| 0/2 [00:00<?, ?it/s]

    model-00001-of-00002.safetensors:   0%|==========| 0.00/9.98G [00:00<?, ?B/s]

    model-00002-of-00002.safetensors:   0%|==========| 0.00/3.50G [00:00<?, ?B/s]

    Loading checkpoint shards:   0%|==========| 0/2 [00:00<?, ?it/s]

    generation_config.json:   0%|==========| 0.00/188 [00:00<?, ?B/s]

    tokenizer_config.json:   0%|==========| 0.00/1.62k [00:00<?, ?B/s]

    tokenizer.model:   0%|==========| 0.00/500k [00:00<?, ?B/s]

    tokenizer.json:   0%|==========| 0.00/1.84M [00:00<?, ?B/s]

    special_tokens_map.json:   0%|==========| 0.00/414 [00:00<?, ?B/s]


## 7. Download Embedding Model


```python
embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```
    modules.json:   0%|==========| 0.00/349 [00:00<?, ?B/s]

    config_sentence_transformers.json:   0%|==========| 0.00/116 [00:00<?, ?B/s]

    README.md:   0%|==========| 0.00/10.6k [00:00<?, ?B/s]

    sentence_bert_config.json:   0%|==========| 0.00/53.0 [00:00<?, ?B/s]

    config.json:   0%|==========| 0.00/571 [00:00<?, ?B/s]

    model.safetensors:   0%|==========| 0.00/438M [00:00<?, ?B/s]

    tokenizer_config.json:   0%|==========| 0.00/363 [00:00<?, ?B/s]

    vocab.txt:   0%|==========| 0.00/232k [00:00<?, ?B/s]

    tokenizer.json:   0%|==========| 0.00/466k [00:00<?, ?B/s]

    special_tokens_map.json:   0%|==========| 0.00/239 [00:00<?, ?B/s]

    1_Pooling/config.json:   0%|==========| 0.00/190 [00:00<?, ?B/s]


## 8. Read in data from `data` directory


```python
documents = SimpleDirectoryReader("/content/data").load_data()
```

## 9.  Create Vector Store and Query Engine


```python
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

index=VectorStoreIndex.from_documents(documents)

query_engine=index.as_query_engine()
```

## 10. Pass a question to Query Engine and print response


```python
response=query_engine.query("What are fundamental rights of Indian citizens?")

print(response)
```
    
    Fundamental rights of Indian citizens are as follows:
    
    1. Right to equality before law and equal protection of the laws.
    2. Prohibition of discrimination on grounds of religion, race, caste, sex, or place of birth.
    3. Right to freedom of speech and expression, assembly, association, and movement.
    4. Protection against arrest and detention in certain cases.
    5. Protection of life and personal liberty.
    6. Right to education.
    7. Protection against exploitation, including traffic in human beings and forced labor.
    8. Freedom of conscience and free profession, practice, and propagation of religion.
    9. Freedom to manage religious affairs.
    10. Protection as to payment of taxes for promotion of any particular religion.
    11. Protection as to attendance at religious instruction or religious worship in certain educational institutions.
    
    These are the fundamental rights guaranteed to Indian citizens under the Indian Constitution.
    


```python
response=query_engine.query("What is the minimum age of President of India?")

print(response)
```

    The minimum age of the President of India is 35 years.
    

