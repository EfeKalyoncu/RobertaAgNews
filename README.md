# RobertaAgNews
Fine tuning RoBERTa on ag_news dataset

# About ag_news dataset
The dataset consists of 4 labels, there are 120000 training samples and 7600 test samples.

# Running Parameters:
--batch_size: (int) The batch size that will be used when running the trainer. Default value is 8, because it avoids exceeding GPU RAM size. <br>

--epoch_count: (int) Number of epochs to train for. Default value 10. <br>

--learning_rate: (float) Learning rate used by the optimizer. Default value of 0.00001 <br>

--freeze: (str) A flag that must be set to "True" to activate. When active, first 10 transformer layers are frozen. <br>

--validate: (str) A flag that must be set to "True" to activate. When active, the program will only run for 2 epochs, and split training set to two to validate. The program will not run test set in this mode. <br>

# Running

Run using `python roberta_finetune.py` at main directory.