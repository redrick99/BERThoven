import os
import pandas as pd
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from more_itertools import locate

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord

torch.manual_seed(41)

def get_notes(path_to_resources_folder: str, octave_aware=True):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    if octave_aware:
        path_to_notes_file = os.path.join(path_to_resources_folder, "data", "notes")
    else:
        path_to_notes_file = os.path.join(path_to_resources_folder, "data", "notes_no_octave_no_chords")

    path_to_midi_files = os.path.join(path_to_resources_folder, "midi_songs")

    if os.path.exists(path_to_notes_file):
        with open(path_to_notes_file, "rb") as filepath:
            notes = pickle.load(filepath)
            return notes

    for file in glob.glob(path_to_midi_files+"/*.mid"):
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
                if octave_aware:
                    notes.append(str(element.pitch))
                else:
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
    random.seed(112)
    for i in range(len(index_count)):
        while index_count[i] < prominent_value:
            x_array = np.random.choice(input_list[i]).tolist()
            # print(x_array)
            x.append(x_array)
            y.append(pitchnames[i])
            index_count[i] += 1

    return x, y




def prepare_dataset(notes, in_seq_length, over_sampling=False):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    input = []
    expected_result = []

    for i in range(0, 6000, 1):# len(notes) - in_seq_length, 1):
        sequence_in = notes[i:i + in_seq_length]
        sequence_out = notes[i + in_seq_length]
        input.append(' '.join(sequence_in))
        expected_result.append(sequence_out)

    #for i in range(12):
    #    print(f"Note {i}: {expected_result.count(str(i))}")

    if over_sampling:
        input, expected_result = over_sample(input, expected_result)
        print(sorted(Counter(expected_result).items()))

    df = pd.DataFrame({'text': input, 'category': expected_result})
    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
    return df_train, df_val, df_test


OCTAVE_AWARE = False

src_path = os.path.dirname(__file__)
resources_path = os.path.join(src_path, "resources")

notes = get_notes(resources_path, OCTAVE_AWARE)
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
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def train(model, train_data, val_data, learning_rate, epochs, dump=False):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    train_acc_per_epoch = []
    val_loss_per_epoch = []
    val_acc_per_epoch = []

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

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

        train_loss_per_epoch.append(total_loss_train/len(train_data))
        train_acc_per_epoch.append(total_acc_train/len(train_data))
        val_loss_per_epoch.append(total_loss_val/len(val_data))
        val_acc_per_epoch.append(total_acc_val/len(val_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')

    if dump:
        train_dump = pd.DataFrame({
            'train_loss': np.array(train_loss_per_epoch),
            'train_accuracy': np.array(train_acc_per_epoch),
            'validation_loss': np.array(val_loss_per_epoch),
            'validation_accuracy': np.array(val_acc_per_epoch),
        })
        train_dump.to_csv('resources/neural_network/training_data/train_dump.csv')


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


def save_torch_model_to_file(model: BertClassifier, test_data, file_path: str, version: int, octave_aware=True):
  test = Dataset(test_data)
  test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

  model.eval()
  model = model.to("cpu")

  sample_input_id, sample_label = next(iter(test_dataloader))

  sample_label = sample_label.to("cpu")
  sample_mask = sample_input_id['attention_mask'].to("cpu")
  sample_input_id = sample_input_id['input_ids'].squeeze(1).to("cpu")

  traced_script_module = torch.jit.trace(model.cpu(), [sample_input_id, sample_mask])
  if octave_aware:
    traced_script_module.save(os.path.join(file_path, "berthoven_model_v"+str(version)+".pt"))
  else:
    traced_script_module.save(os.path.join(file_path, "berthoven_balanced_no_chords_v"+str(version)+".pt"))


df_train, df_val, df_test = prepare_dataset(notes, 100, over_sampling=True)
EPOCHS = 20
model = BertClassifier(n_vocab)
LR = 1e-6

train(model, df_train, df_val, LR, EPOCHS, dump=True)

evaluate(model, df_test)
save_torch_model_to_file(model, df_test, os.path.join(resources_path, "neural_network", "models"), 6, OCTAVE_AWARE)
