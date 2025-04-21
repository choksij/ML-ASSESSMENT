# **Task 4: Training Loop Implementation**

## **Objective**
In this task, I've implemented a comprehensive and robust training loop for the multi-task learning (MTL) sentence transformer introduced in Task 2. This implementation highlights key components of training neural networks in a multi-task setting, including forward propagation, loss calculation, back-propagation, checkpointing, early stopping, and visualization of training dynamics.

## **Project Structure**

TASK3/\
├── config.yaml \
├── checkpoints/\
│   ├── best_epoch_X.pt\
├── outputs/\
│   ├── loss_curve.png\
│   ├── cls_accuracy.png\
│   ├── sent_accuracy.png\
├── src/\
│   └── __init__.py\
│   └── trainer.py\
│   └── utils.py\
├── data/\
│   ├── sample_sentences.csv\
├── TRAINING_LOOP.md\
└── README_task3.md \

## **Setup and Execution**

### 1. Execution

    ```
    python -m src.trainer --config config.yaml
    ```

*This command will:*
1. Split the provided dataset into training and validation sets.
2. Run training with defined epochs, computing losses and accuracies.
3. Implement checkpointing, learning rate scheduling, and early stopping.


### 2. Configurations/Modifications (config.yaml)

``` 
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  pooling: "mean"
  normalize: true

data:
  sample_file: "data/sample_sentences.csv"

tasks:
  classification:
    labels: ["sports", "entertainment", "finance"]
  sentiment:
    labels: ["positive", "negative", "neutral"]

training:
  epochs: 10
  batch_size: 32
  val_split: 0.2
  freeze_mode: "backbone"  # Options: all, backbone, cls_head, sent_head, none
  patience: 2
  warmup_ratio: 0.1
  learning_rates:
    backbone: 1e-5
    head: 1e-4
  loss_weights:
    classification: 1.0
    sentiment: 1.0
 ```

### 4. Training Loop Explanation

1. **Forward Pass**
    - Each batch of sentences is tokenized and encoded.
    - Transformer backbone computes contextual embeddings.
    -  Embeddings pooled (mean or cls token) and optionally normalized.
    - Two task-specific linear heads output predictions.
2. **Loss Calculation**
    - Separate cross-entropy losses computed for classification and sentiment tasks.
    - Combined into total loss based on weights in configuration.
3. **Metrics and Evaluation**
    - Train and validation loss computed each epoch.
    - Classification and sentiment accuracy tracked each epoch.
4. **Optimization and Scheduler**
    - AdamW optimizer with differential learning rates:
        - Lower rate (1e-5) for backbone parameters.
        -   Higher rate (1e-4) for classification and sentiment heads.
    - Linear scheduler with warm-up implemented for smooth convergence.
5. **Checkpointing and Early Stopping**
    - Best-performing model (lowest validation loss) saved as a checkpoint.
    - Training stops early if validation loss does not improve within patience period.

<img src="output_images/task4/task4-output.jpg" />


### 5. Training Visualization

After training, the following plots are automatically generated and saved under outputs/:

  1. Loss Curve (outputs/loss_curve.png): Displays training vs. validation loss per epoch.
  2. Classification Accuracy (outputs/cls_accuracy.png): Shows training and validation classification accuracy.
  3. Sentiment Accuracy (outputs/sent_accuracy.png): Illustrates training and validation sentiment accuracy.

<p float="left">
  <img src="TASK4/outputs/cls_accuracy.png" width="550" /> 
  <img src="TASK4/outputs/loss_curve.png" width="550" /> 
  <img src="TASK4/outputs/sent_accuracy.png" width="550" />
</p>

### 6. Enhancements

1. Early Stopping: Prevents unnecessary training after performance plateau.
2. Checkpointing: Saves the best model based on validation loss.
3. Learning Rate Scheduling: Ensures gradual training dynamics to improve convergence.

### 7. Assumptions and Decisions Made

1. Used minimal hypothetical dataset for demonstration purposes.
2. Equal importance given to classification and sentiment tasks initially.
3. Patience for early stopping set intentionally low for demonstration.

#### Task 4 implementation demonstrates robust training techniques within a multi-task learning environment, highlighting essential practices such as differential learning rates, early stopping, checkpointing, and comprehensive visualization.


*All tasks have been completed per the assessment instructions, with functionality demonstrated on a small example dataset as specified with more focus on the overall working, the output generated is just to show the working of the model*