import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import collections
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data.sampler import SubsetRandomSampler
from datasets import load_dataset
from tqdm import tqdm
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="RoBERTa fine tuning on ag_news dataset")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epoch_count", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--freeze", type=str, default="False")
parser.add_argument("--validate", type=str, default="False")
args = parser.parse_args()

use_freeze_model = False
validation_run = True
learning_rate = args.learning_rate

if (args.freeze == "True"):
    use_freeze_model = True

if (args.validate == "True"):
    validation_run = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(batched_text):
    return tokenizer(batched_text['text'], padding=True, truncation=True)

def load_roberta_data():
    train_data, test_data = load_dataset("ag_news", split=['train', 'test'])

    train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))

    train_data = list(map(lambda d: (torch.tensor(d['input_ids']), d['label']), train_data))
    test_data = list(map(lambda d: (torch.tensor(d['input_ids']), d['label']), test_data))

    return train_data, test_data

def test(model, data):
    confusion_matrix_size = len(collections.Counter(e[1] for e in data))
    confusion_matrix = []

    total = 0
    correct = 0

    for i in range(confusion_matrix_size):
        row = []
        for j in range(confusion_matrix_size):
            row.append(0)
        confusion_matrix.append(row)

    for i, batch in tqdm(enumerate(data, 0), total=len(data), leave=False):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
            
        outputs = model(inputs).logits.to(device)

        _, predicted = torch.max(outputs, 1)

        for j in range(len(predicted)):
            confusion_matrix[labels[j].item()][predicted[j].item()] += 1
        
        total += len(predicted)
        correct += predicted.eq(labels).sum().item()
    print("Test loss: {:.3f}".format(correct/total))
    print(confusion_matrix)

def train(model, data, epochs):
    n = len(data)

    if validation_run:
        epochs = 1
        
    accuracy_history_epoch = []
    accuracy_history_step = []
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        correct = 0
        total = 0
        for i, batch in tqdm(enumerate(data, 0), total=len(data), leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs).logits.to(device)
            _, predicted = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total += len(predicted)
            correct += predicted.eq(labels).sum().item()
            accuracy_history_step.append((i+1, correct/total))

        accuracy_history_epoch.append(correct / total)
        print("Epoch: {:>3d} Loss: {:.3f}".format(epoch, accuracy_history_epoch[-1]))

    return accuracy_history_epoch, accuracy_history_step


train_data, test_data = load_roberta_data()
different_label_count = len(collections.Counter(e[1] for e in train_data))

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=different_label_count).to(device)

# When using layer freezing, set requires grad on transformer layers
# 0-9 to False so they do not get changed
if use_freeze_model:
    for name, param in model.named_parameters():
        # First 10 transformer layers will all have the format layers.%d.
        # So we can match their names with regex to freeze their parameters
        if re.search("roberta\.encoder\.layer\..\.", name):
            param.requires_grad = False

indices = list(range(len(train_data)))

if validation_run:
    validation_split_size = 0.2
    np.random.shuffle(indices)

    validation_split_point = int(np.floor(validation_split_size * len(train_data)))
    train_indices, validation_indices = indices[validation_split_point:], indices[:validation_split_point]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, sampler=train_sampler)
    validation_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, sampler=validation_sampler)
else:
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

try:
    print(model)
    train(model, train_dataloader, args.epoch_count)
    if not validation_run:
        test(model, test_dataloader)
    else:
        test(model, validation_dataloader)
except Exception as e:
    print(e)