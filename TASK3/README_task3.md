# **Task 2: Training Considerations & Freeze Strategies**

## **Objective**
The objective of Task 3 is to explore training considerations for the multi-task transformer developed previously. This task specifically examines:

  1. Different freezing strategies for neural network training.
  2. Transfer learning considerations, including the rationale behind selecting a pre-trained model and deciding which layers to freeze or unfreeze.

## **Project Structure**

TASK3/\
├── config.yaml \
├── outputs/\
│   ├── trainability.png\
├── src/\
│   └── freeze_control.py\
├── tests/\
│   ├── test_multitask_encoder.py\
│   └── test_main.py\
├── TRAINING_CONSIDERATIONS.md\
└── README_task3.md \

## **Setup and Execution**

### 1. Execution

    ```
    python -m src.freeze_control --mode backbone
    ```
*Available modes:*
  1. *all: Freeze the entire model.*
  2. *backbone: Freeze only transformer backbone layers.*
  3. *cls_head: Freeze only the classification head.*
  4. *sent_head: Freeze only the sentiment head.*

### 2. Modifications (config.yaml)

``` 
model:
  name: sentence-transformers/all-MiniLM-L6-v2
  pooling: mean
  normalize: true

training:
  freeze_mode: backbone  # options: all, backbone, cls_head, sent_head
 ```

### 4. Visualization of Freeze Status

The provided visualization (outputs/trainability.png) shows clearly which components of the model are frozen vs. trainable.

 1. Frozen components: represented with cross-hatching and labeled as "Frozen".
 2. Trainable components: represented with diagonal hatching and labeled as "Trainable".

<img src="TASK3/outputs/trainability.png" />

### 5. Training Considerations

Detailed theoretical analysis of different freezing scenarios and transfer learning strategies is documented in: TRAINING_CONSIDERATIONS.md

  1. Pros and cons of freezing entire network vs. backbone vs. task-specific heads.
  2. Recommendations for freezing strategy given dataset size and domain.
  3. Transfer learning best practices:
      - Choosing an appropriate pre-trained model.
      - Deciding layers to freeze/unfreeze and rationale.


### 6. Recommended Freeze Strategy (Default)
Based on typical transfer-learning scenarios with moderate-sized datasets:

```
freeze_mode: backbone
```
*This mode freezes the large transformer backbone and allows rapid, focused training of task-specific heads to minimize overfitting and accelerate training.*

### 7. Results:

<img src="../output_images//task3/task3-output.jpg" />

#### Since I am using a dummy dataset here, so an advance tip for larger datasets will be:
  #### 1. consider gradually unfreezing the backbone layers starting from the top-most layers downwards. This "layer-wise unfreezing" strategy often helps achieve superior performance while minimizing catastrophic forgetting. 
  #### 2. Adjust learning rates per module: lower learning rates for backbone, higher for newly initialized heads to facilitate stable and effective training.

  
*All tasks have been completed per the assessment instructions, with functionality demonstrated on a small example dataset as specified with more focus on the overall working, the output generated is just to show the working of the model*