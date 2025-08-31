# Retrieval-Augmented Generation (RAG) Module

## Overview
This repository is part of the **smart_crane** project, focusing on retrieval-augmented capabilities.  
It provides examples of deploying and testing [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) and [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) on the **SOPHON BM1684X** platform.

## Usage

1. **Model Compilation**  
   You can directly use the pre-compiled models provided in this repository, or compile your own (see [Model Compilation](#model-compilation)).

2. **Environment Setup**  
   - Follow the official **LIBSOPHON** and **SAIL** manuals to correctly install `libsophon` and `sail`.  
   - Additionally, install the latest versions of **PyTorch** and **Transformers**.

3. **Run Inference**  
   Simply execute `QwenEmbedding.py` or `QwenReranker.py` to obtain results.

---

## Model Compilation

1. Set up the compilation environment according to the **TPU-MLIR Developer Guide**.  
2. Download the `.pt` model.  
3. Write the compilation logic as needed.  

Compilation generally follows the [Qwen3 Export Guide](https://github.com/sophgo/sophon-demo/blob/release/sample/Qwen/docs/Qwen_Export_Guide.md).  
You can directly compile models with the official instructions.  

If you need to compile a **KV-cache-free model**, copy the two files under the `compile` directory into the `tpu-mlir/python/llm` directory (next to `LlmConverter.py`).  
Then, modify `/tools/llm_convert.py` as follows:

```python
    if config.model_type in ["qwen3", "qwen2", "llama", "minicpm"]:
        from llm.LlmConverter_nokv_reranker import LlmConverter
        converter = LlmConverter(args, config)
```

or

```python
    if config.model_type in ["qwen3", "qwen2", "llama", "minicpm"]:
        from llm.LlmConverter_nokv_embedding import LlmConverter
        converter = LlmConverter(args, config)
```

> Note: `compile/LlmConverter_nokv_embedding.py` and `compile/LlmConverter_nokv_reranker.py` are modified versions of `LlmConverter.py`, with adjusted export logic.

If exporting an **embedding model**, also modify `python/llm/LlmLoad.py` as follows:

```python
    def read(self, key: str):
        key = key.replace("model.", "")
        for f in self.st_files:
            if key in f.keys():
                if isinstance(f, dict):
                    data = f[key]
                else:
                    data = f.get_tensor(key)
                if data.dtype in [torch.float16, torch.bfloat16]:
                    return data.float().numpy()
                return data.numpy()
        raise RuntimeError(f"Can't find key: {key}")

    def is_exist(self, key: str):
        key = key.replace("model.", "")
        for f in self.st_files:
            if key in f.keys():
                return True
        return False
```

> Explanation: The embedding model stores weight keys differently from the standard Qwen3 models.

4. Start the compilation process.

---

## RAG Workflow

1. **Reranker**
   Given a query and candidate keys, the reranker constructs prompts for the LLM.
   By computing the probability of "yes" vs. "no" as the next token, it assigns a relevance score to determine whether the key is a valid answer for the query.

2. **Embedding**
   A large corpus of text is processed in batches through the embedding model.
   The hidden state of the last token from the final attention block is extracted as the text representation.
   Cosine similarity is then computed across embeddings to measure semantic relevance between text pairs.



