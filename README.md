# Vietnamese-Sentiment-Distillation

# Description
- This repo implemented Model Distillation (Transfer Knowledge) Strategies on Vietnamese Sentiment dataset
- Purpose: Observing approach that can help light-weight model outperforms deep-weight in specific NLP task
- Task: Sentiment Analysis
- Dataset: UIT-VSFC
- Source : Knowledge Distillation Tutorial (https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)

## Setting Up
### Installtion
```
git clone https://github.com/21522173huy/Vietnamese-Sentiment-Distillation
cd Vietnamese-Sentiment-Distillation
```


## Run Script
Here is my pretrained-T5 weight: `https://drive.google.com/drive/folders/1rJqLDSarEcIOujA_PS0sCcIP8gXjYxJE?usp=sharing`.
Please download and add checkpoint path to all the necessary parts of the code below.

### Teacher Finetuning
```
python teacher/finetune_script.py \
--teacher_name ViT5 \
--batch_size 32 \
--epochs 10
```
### Student Training
```
python student/training_script.py \
--teacher_name ViT5 \
--teacher_checkpoint <appropriate_teacher_weight.pt> \
--student_type base \
--batch_size 32 \
--epochs 10 \
--soft_weight 2.5e-1 \
--hard_weight 7.5e-1
```
### Evaluation
```
python evaluation.py \
--model_name ViT5 \
--teacher_or_student student \
--model_type base \
--model_checkpoint <appropriate_student_weight.pt> \
--batch_size 32
```

### Results

- Student-50-50 : 0.5 for SoftWeight and 0.5 for HardWeight.
- Student-25-75 : 0.25 for SoftWeight and 0.75 for HardWeight.
- Bi-LSTM/Word2Vec: `Deep Learning versus Traditional Classifiers on Vietnamese Student’s Feedback Corpus` paper.
- MaxEnt : `UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis` paper.

|  Model | Params |Accuracy | Precision | Recall | F1 |
| -------- | ------- |------- | -------- |-------- |-------- |
| Teacher  | 346M |93.14%|93.14% |93.14% |93.14%
| Base Student 50-50  |  113M |**93.36%**| **93.36%**  |93.36% |**93.36%**|
| Base Student 25-75  |  113M |93.11%| 93.11%  |93.11% |93.11%|
| Large Student 50-50  |  115M |90.80%| 90.80  |90.80 |90.80|
| Large Student 25-75  |  115M |91.11%| 91.11%  |91.11% |91.11%|
| Bi-LSTM/Word2Vec  |  - |-| 90.80%  |**93.40%** |92.00%|
| MaxEnt  |  - |- | 87.71% |88.66% |87.94%|

### Inference
```
python inference.py \
--model_name ViT5 \
--teacher_or_student student \
--model_type base \
--model_checkpoint <appropriate_model_weight.pt>
--input_sentence "Có slide trong bài giảng"
```

```
For Sentence: "Có slide trong bài giảng"
Result is:
  Negative: 0.0012
  Positive: 0.5979
  Neutral: 0.4009
```





