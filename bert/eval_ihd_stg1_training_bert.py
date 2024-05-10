import argparse
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from modelling.roberta import WeightedLayerPooling


class ImplicitHateDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset = self.dataset.sample(
            frac=1, random_state=args.seed).reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.transform(self.dataset.iloc[index])

    def transform(self, row):
        if row['class'] == 'not_hate':
            return row['post'], 0
        elif row['class'] == 'implicit_hate':
            return row['post'], 1


class IHDModel(torch.nn.Module):
    def __init__(self):
        super(IHDModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Average last 4 layers
        # self.pooler = WeightedLayerPooling(
        #     num_hidden_layers=self.bert.config.num_hidden_layers, layer_start=9)
        # Concat Last 4 layers embeddings
        # self.fc1 = torch.nn.Linear(768*4, 100)
        # Last layer embeddings
        self.fc1 = torch.nn.Linear(768, 100)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(100, 100)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(100, args.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=True)
        x = outputs[0]
        # Average last 4 layers
        # x = self.pooler(x)
        # Concat last 4 layers embeddings
        # x = torch.cat((x[-4][:, 0], x[-3][:, 0], x[-2]
        #               [:, 0], x[-1][:, 0]), dim=1)
        # x = self.fc1(x)
        # Last Layer embeddings
        x = self.fc1(x[:, 0, :])
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        required=True,
                        help="Path to the trained model.")

    parser.add_argument("--input_files", default=None,
                        required=True, nargs='+',
                        help="The input files. Should contain csv files for the task.")

    parser.add_argument('--output_path', type=str,
                        required=True, help="Output path to log metrics.")

    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for the DataLoader.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")

    parser.add_argument("--tokenizer_path",
                        default='bert-base-uncased', type=str,
                        help="The path to the tokenizer.")

    parser.add_argument("--num_labels",
                        type=int,
                        default=2,
                        help="Number of labels.")

    parser.add_argument("--metrics_average",
                        type=str,
                        default='binary',
                        help="The average used by metrics.")

    args = parser.parse_args()
    return args


def get_datasets():
    ds = pd.DataFrame()
    for input_file in args.input_files:
        ds = ds._append(pd.read_csv(input_file))
    return ds


def evaluate(model, dataset, tokenizer):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        with tqdm(dataset, unit='batch', leave=True, position=0) as tepoch:
            for batch_idx, (text, label) in enumerate(tepoch):
                inputs = tokenizer(
                    text, padding=True, truncation=True, max_length=512, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                y_true = y_true + list(label.numpy())
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                logits = outputs
                y_pred = y_pred + \
                    list(
                        logits.detach().cpu().max(1).indices.numpy())
                torch.cuda.empty_cache()
    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred, average=args.metrics_average)
    rec_score = recall_score(y_true, y_pred, average=args.metrics_average)
    f_score = f1_score(y_true, y_pred, average=args.metrics_average)
    print("Printing metrics...")
    print("Accuracy: ", acc_score)
    print("Precision: ", prec_score)
    print("Recall: ", rec_score)
    print("F1: ", f_score)
    print("Saving metrics...")

    metrics = pd.DataFrame(
        {'accuracy': [acc_score], 'precision': [prec_score], 'recall': [rec_score], 'f1': [f_score]})
    metrics.to_csv(args.output_path, index=False)


def main():
    print("running with input files: \n", args.input_files)
    # Setting up random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting up datasets
    test_dataset = get_datasets()
    test_dataset = ImplicitHateDataset(test_dataset)
    print("Datsets configured...")

    # Setting up dataloaders
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)
    print("DataLoader configured...")

    # Setting up tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

    # Setting up the model
    model = IHDModel()
    model.load_state_dict(torch.load(args.model_path))
    print("Model configured...")

    # Loading weights
    model.load_state_dict(torch.load(args.model_path))
    print("Weights configured...")

    print("Starting Evaluation...")
    evaluate(model, test_loader, tokenizer)


if __name__ == "__main__":
    args = parse_arguments()
    main()
