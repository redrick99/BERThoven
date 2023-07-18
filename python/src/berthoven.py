import os
import re
import time
import pandas as pd
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from collections import Counter
from more_itertools import locate
from pathlib import Path

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

torch.manual_seed(41)


def get_notes(path_to_resources_folder: str):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    path_to_notes_file = os.path.join(path_to_resources_folder, "data", "notes_no_octave_no_chords")

    path_to_midi_files = os.path.join(path_to_resources_folder, "midi_songs")

    if os.path.exists(path_to_notes_file):
        with open(path_to_notes_file, "rb") as filepath:
            notes = pickle.load(filepath)
            return notes

    for file in glob.glob(path_to_midi_files + "/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch.pitchClass))
            elif isinstance(element, chord.Chord):
                continue
                notes.append(str(element.normalOrder[0]))
                # notes.append('.'.join(str(n) for n in element.normalOrder))

    with open(path_to_notes_file, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def over_sample(x: list, y: list):
    pitchnames = sorted(set(y))
    index_count = []
    input_list = []
    x_np = np.array(x)
    for i in pitchnames:
        index_count.append(y.count(i))
        input_list.append(x_np[list(locate(y, lambda n: n == i))])
    prominent_value = np.max(np.array(index_count))
    print(prominent_value)

    np.random.seed(112)
    for i in range(len(index_count)):
        while index_count[i] < prominent_value:
            x_array = np.random.choice(input_list[i]).tolist()
            # print(x_array)
            x.append(x_array)
            y.append(pitchnames[i])
            index_count[i] += 1

    return x, y


def prepare_dataset(notes, in_seq_length, over_sampling=False, shuffle=False, num_of_samples=8000):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    input = []
    expected_result = []
    random.seed(112)

    for i in range(0, len(notes) - in_seq_length, 1):
        sequence_in = notes[i:i + in_seq_length]
        sequence_out = notes[i + in_seq_length]
        input.append(' '.join(sequence_in))
        expected_result.append(sequence_out)

    if shuffle:
        combined_list = list(zip(input, expected_result))
        random.shuffle(combined_list)
        input, expected_result = zip(*combined_list)
        input = list(input)
        expected_result = list(expected_result)

    input = input[:num_of_samples]
    expected_result = expected_result[:num_of_samples]

    # for i in range(12):
    #    print(f"Note {i}: {expected_result.count(str(i))}")

    if over_sampling:
        input, expected_result = over_sample(input, expected_result)
        print(sorted(Counter(expected_result).items()))

    df = pd.DataFrame({'text': input, 'category': expected_result})
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
    return df_train, df_val, df_test


src_path = os.path.dirname(__file__)
resources_path = os.path.join(src_path, "resources")

notes = get_notes(resources_path)
pitchnames = sorted(set(item for item in notes))
n_vocab = len(pitchnames)
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
labels = note_to_int

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', torchscript=True)

vocab_file = open(os.path.join(resources_path, "data", "bert_cased_vocab.txt"), "w", encoding="utf-8")
for i, j in tokenizer.get_vocab().items():
    vocab_file.write(f"{i} {j}\n")
vocab_file.close()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [labels[label] for label in df['category']]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=200, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, n_vocab, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_vocab)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def save_model_checkpoints(epoch: int, model, optimizer, path_to_model_checkpoint):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path_to_model_checkpoint + ".pth")


def load_model_checkpoints(model, optimizer, path_to_model_checkpoint):
    checkpoint = torch.load(path_to_model_checkpoint + ".pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def train(model, train_data, val_data, learning_rate, epochs, batch_size: int, path_to_model="", path_to_checkpoint="",
          early_stopping=True, load=False):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0

    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []
    elapsed_time_per_epoch = []

    best_val_loss = np.inf
    early_stop_counter = 0
    epsilon = .001
    patience = 5

    if load:
        model, optimizer, start_epoch = load_model_checkpoints(model, optimizer, path_to_checkpoint)
        train_dump = pd.read_csv(path_to_model+'_dump.csv')
        train_loss_per_epoch = train_dump['train_loss'].values().tolist()
        train_acc_per_epoch = train_dump['train_accuracy'].values().tolist()
        val_loss_per_epoch = train_dump['validation_loss'].values().tolist()
        val_acc_per_epoch = train_dump['validation_accuracy'].values().tolist()
        elapsed_time_per_epoch = train_dump['time'].values().tolist()
        best_val_loss = val_loss_per_epoch[len(val_loss_per_epoch) - 1]

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(start_epoch, epochs):
        total_acc_train = 0
        total_loss_train = 0
        s_time = time.time()

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        elapsed_time_per_epoch.append(time.time() - s_time)
        train_loss_per_epoch.append(total_loss_train / len(train_data))
        train_acc_per_epoch.append(total_acc_train / len(train_data))
        val_loss_per_epoch.append(total_loss_val / len(val_data))
        val_acc_per_epoch.append(total_acc_val / len(val_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')

        if total_loss_val < best_val_loss - epsilon:
            best_val_loss = total_loss_val
            early_stop_counter = 0
            if path_to_checkpoint != "":
                save_model_checkpoints(epochs, model, optimizer, path_to_checkpoint)
        elif early_stopping:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Triggered early stopping!")
            break

    if path_to_model != "":
        train_dump = pd.DataFrame({
            'train_loss': np.array(train_loss_per_epoch),
            'train_accuracy': np.array(train_acc_per_epoch),
            'validation_loss': np.array(val_loss_per_epoch),
            'validation_accuracy': np.array(val_acc_per_epoch),
            'time': np.array(elapsed_time_per_epoch),
        })
        train_dump.to_csv(path_to_model + "_dump.csv")


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


def save_torch_model_to_file(model: BertClassifier, test_data, path_to_model: str):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    model.eval()
    model = model.to("cpu")

    sample_input_id, sample_label = next(iter(test_dataloader))

    sample_label = sample_label.to("cpu")
    sample_mask = sample_input_id['attention_mask'].to("cpu")
    sample_input_id = sample_input_id['input_ids'].squeeze(1).to("cpu")

    traced_script_module = torch.jit.trace(model.cpu(), [sample_input_id, sample_mask])
    traced_script_module.save(path_to_model + ".pt")


def create_and_train_new_model(r_path, n_vocab, notes, n_samp, epochs, seq_len, lr, batch_size, over_sampling, shuffle,
                               e_stop):
    model_name = "a_e" + str(epochs) + "_n" + str(n_samp) + "_i" + str(seq_len) + "_lr" + str(lr) + "_bs" + str(
        batch_size) + "_os" \
                 + str(int(over_sampling)) + "_s" + str(int(shuffle))
    model_path = os.path.join(r_path, "neural_network", "evaluation_models_dumps", model_name)
    check_path = os.path.join(r_path, "neural_network", "checkpoints", model_name)
    model = BertClassifier(n_vocab)
    df_train, df_val, df_test = prepare_dataset(notes, in_seq_length=seq_len, over_sampling=over_sampling,
                                                shuffle=shuffle, num_of_samples=n_samp)

    train(model, df_train, df_val, lr, epochs, batch_size, path_to_model=model_path, path_to_checkpoint=check_path,
          early_stopping=e_stop, load=False)


def extract_parameters(input_string):
    # Define the regular expression pattern to match numbers
    pattern = r'\d+'

    # Use regular expression to find all numbers in the input string
    numbers = re.findall(pattern, input_string)

    # Convert the numbers to appropriate data types if needed
    e = int(numbers[0])
    n = int(numbers[1])
    i = int(numbers[2])
    lr = round(float(numbers[3]))*(10**(-float(numbers[4])))
    bs = int(numbers[5])
    os = bool(numbers[6])
    s = bool(numbers[7])

    # Return the extracted parameters as a dictionary
    return {
        'e': e,
        'n': n,
        'i': i,
        'lr': lr,
        'bs': bs,
        'os': os,
        's': s
    }


def resume_training(r_path, path_to_checkpoint, model_name, n_vocab, notes, epochs, e_stop):
    params = extract_parameters(path_to_checkpoint)
    n_samp = params['n']
    seq_len = params['i']
    lr = params['lr']
    batch_size = params['bs']
    over_sampling = params['os']
    shuffle = params['s']

    model_path = os.path.join(r_path, "neural_network", "evaluation_models_dumps", model_name)

    model = BertClassifier(n_vocab)
    df_train, df_val, df_test = prepare_dataset(notes, in_seq_length=seq_len, over_sampling=over_sampling,
                                                shuffle=shuffle, num_of_samples=n_samp)

    train(model, df_train, df_val, lr, epochs, batch_size, path_to_model=model_path, path_to_checkpoint=path_to_checkpoint,
          early_stopping=e_stop, load=True)


NUM_OF_SAMPLES = 10000
EPOCHS = 20
IN_SEQ_LENGTH = 100
LR = 5e-6
BATCH_SIZE = 16

model = BertClassifier(n_vocab)
optimizer = Adam(model.parameters(), lr=LR)
path_to_checkpoint = os.path.join(resources_path, 'neural_network', 'checkpoints', 'a_e20_n10000_i100_lr5e-06_bs16_os1_s0')
path_to_saved_model = os.path.join(resources_path, 'neural_network', 'libtorch_models', 'FINAL_BERTHOVEN_MODEL')

model, optimizer, start_epoch = load_model_checkpoints(model, optimizer, path_to_checkpoint)
df_train, df_val, df_test = prepare_dataset(notes, in_seq_length=IN_SEQ_LENGTH, over_sampling=True,
                                                shuffle=False, num_of_samples=NUM_OF_SAMPLES)
evaluate(model, test_data=df_test)
save_torch_model_to_file(model, df_test, path_to_saved_model)
