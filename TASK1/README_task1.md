# **Task 1: Sentence Transformer Implementation**

## **Objective**
Implement a sentence transformer model capable of encoding input sentences into fixed-length embeddings using a transformer-based backbone.

## **Overview**
In this task, I developed a Sentence Transformer Encoder leveraging a pretrained transformer (all-MiniLM-L6-v2) to generate embeddings from input sentences. The model supports two pooling strategies:

	1. Mean pooling: Average of token embeddings.

	2. CLS pooling: Embedding corresponding to the [CLS] token.

## **Project Structure**

TASK1/\
├── config.yaml \
├── data/\
│   └── sample_sentences.txt\
├── outputs/\
│   ├── embeddings.npy\
│   ├── embeddings.txt\
│   ├── cls_embeddings.npy\
│   └── cls_embeddings.txt\
├── requirements.txt\
├── src/\
│   ├── __init__.py\
│   ├── encoder.py\
│   ├── main.py\
│   └── utils.py\
├── tests/\
│   ├── test_encoder.py\
│   └── test_main.py\
└── README_TASK1.md \

## **Setup and Execution**

### 1. Install dependencies

    ```
    pip install -r requirements.txt
    ```

### 2. Execution (Default: mean pooling)

    ```
    python -m src.main --config config.yaml
    ```

### 3. Modifications (config.yaml)

``` 
model:
  name: sentence-transformers/all-MiniLM-L6-v2  # Transformer backbone
  pooling: mean                                # Pooling type: mean or cls
  normalize: true                              # Apply L2 normalization?

inference:
  batch_size: 32                               # Batch size for encoding

data:
  sample_file: data/sample_sentences.txt       # Input sentences file

output:
  embeddings_file: outputs/embeddings.npy      # Binary embeddings file
  results_file: outputs/embeddings.txt         # Text embeddings file
 ```

### 4. Tests

```
pytest -q -s
```

### 5. Results

Upon running the script, embeddings are generated and saved in two formats:

    1. Binary format (.npy): Efficient storage for programmatic use.
    2. Text format (.txt): Readable embeddings for easy inspection.

<img src="output_images/task1/task1-output.jpg" />

### 6. Technical Choices
    1. Model choice (all-MiniLM-L6-v2): Lightweight and efficient transformer suitable for quick prototyping while maintaining solid performance.
    2. Pooling methods: Mean pooling offers balanced sentence-level embedding; CLS pooling provides embeddings specifically from the special [CLS] token.
    3. L2 normalization: Ensures consistent cosine similarity comparisons.



*All tasks have been completed per the assessment instructions, with functionality demonstrated on a small example dataset as specified with more focus on the overall working.*