//
// Created by Riccardo Rossi on 09/07/23.
//

#ifndef BERTHOVEN_BERTHOVENMODEL_H
#define BERTHOVEN_BERTHOVENMODEL_H

#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <vector>
#include <istream>

#include <torch/torch.h>
#include <torch/script.h>
#include <juce_audio_processors/juce_audio_processors.h>


class BerthovenModel {
public:
    BerthovenModel();
    ~BerthovenModel();

    void putMidiEvent(juce::MidiMessage& event);

    std::vector<juce::MidiMessage> predict(int numberOfNotesToPredict, double noteDurationInSeconds);
    void reset();

private:
    void clearMidiEvents();

    std::vector<juce::MidiMessage> storedMidiEvents;
    std::vector<std::string> firstNotes;
    double noteOffsetForChord = 0.01;
    int octave = 4;

    std::mutex mtx;

    bool considerChords = false;
    int max_length = 100;
    int max_length_tokenizer = 200;
    std::map<std::string, int> token2id;
    std::map<int, std::string> id2token;
    torch::jit::script::Module bert;
    std::map<int, std::string> pred2class;
    std::string pathToNotesFile = "/resources/notes_no_octave.txt";
    std::string pathToModule = "/resources/FINAL_BERTHOVEN_MODEL.pt";
    std::string pathToVocab = "/resources/bert_cased_vocab.txt";

    std::pair<torch::Tensor, torch::Tensor> preprocess(std::string text, std::map<std::string, int> token2id, int max_length, bool log = false);
    void init_vocab(std::string vocab_path = "../bert_cased_vocab.txt");
    void init_bert(std::string bert_path = "../traced_text_classification_model.pt");
    std::pair<std::map<std::string, int>, std::map<int, std::string>> get_vocab(std::string vocab_path);
    torch::jit::script::Module load_model(std::string  model_path);

    void readNotesDataset(const std::string& filePath, bool considerChords);
    std::string concatenateStrings(const std::vector<std::string>& strings);
    std::vector<std::string> createNotesForPrediction(std::vector<juce::MidiMessage> noteEvents, double chordOffset, bool considerChords);

    std::vector<juce::MidiMessage> midiNotesToMessages(const std::vector<std::vector<int>> midiNotes, double noteDurationInSeconds);
    std::vector<int> stringToMidiNotes(const std::string& notesString, int octave);
    std::vector<std::string> splitString(const std::string& input, const char delimiter);
};


#endif //BERTHOVEN_BERTHOVENMODEL_H
