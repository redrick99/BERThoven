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

/**
 * @class BerthovenModel
 * @brief Handles all interactions with the underlying neural network model.
 */
class BerthovenModel {
public:
    /**
     * @brief Constructor for BerthovenModel.
     */
    BerthovenModel();
    /**
     * @brief Destructor for BerthovenModel.
     */
    ~BerthovenModel();

    /**
     * @brief Setter for the ticksPerQuarterNote attribute.
     * @param _ticksPerQuarterNote New value to set.
     */
    void setTicksPerQuarterNote(double _ticksPerQuarterNote);

    /**
     * @brief Stores MIDI events to be used for prediction.
     * @param event MIDI event to be stored.
     */
    void putMidiEvent(juce::MidiMessage& event);

    /**
     * @brief Predicts a number of notes through the neural network model.
     * @param numberOfNotesToPredict Specified by the user.
     * @param noteDurationInSeconds Duration of each predicted note specified by the user.
     * @return A vector containing the predicted notes as MIDi messages.
     */
    std::vector<juce::MidiMessage> predict(int numberOfNotesToPredict, double noteDurationInSeconds);

    /**
     * @brief Resets the stored MIDI events to initial conditions.
     */
    void reset();

private:
    /**
     * @brief Removes all stored MIDI events.
     */
    void clearMidiEvents();

    /// Contains MIDI events used for prediction.
    std::vector<juce::MidiMessage> storedMidiEvents;
    /// Contains a string representation of the first notes in the sequence (used in reset).
    std::vector<std::string> firstNotes;
    double noteOffsetForChord = 0.01;
    int octave = 4;

    std::mutex mtx;

    /// Whether to consider chords when predicting.
    bool considerChords = false;
    /// Max length of the input sequence to the model.
    int max_length = 100;
    /// Max length of the sequence given to the tokenizer to tokenize.
    int max_length_tokenizer = 200;
    /// Maps BERT's tokens to their id.
    std::map<std::string, int> token2id;
    /// Maps BERT's token ids back to their token.
    std::map<int, std::string> id2token;
    /// Torch module used for prediction.
    torch::jit::script::Module bert;
    /// Maps integer numbers to their output labels.
    std::map<int, std::string> pred2class;
    std::string pathToNotesFile = "/resources/notes_no_octave.txt";
    std::string pathToModule = "/resources/FINAL_BERTHOVEN_MODEL.pt";
    std::string pathToVocab = "/resources/bert_cased_vocab.txt";

    /// Default tick resolution value for Logic Pro X
    double ticksPerQuarterNote = 960.0;

    /**
     * @brief Preprocesses the given input text into tensors used for inference on the model.
     * @param text Input string to process.
     * @param token2id Map to map tokens to their ids.
     * @param max_length Maximum length of the input sequence.
     * @param log Whether to print on console during execution of the function.
     * @return A pair of tensors, one for the tokens and one for their attention masks.
     */
    std::pair<torch::Tensor, torch::Tensor> preprocess(std::string text, std::map<std::string, int> token2id, int max_length, bool log = false);
    /**
     * @brief Initializes BERT's vocabulary from file.
     * @param vocab_path Path to the file containing the vocabulary.
     */
    void init_vocab(std::string vocab_path = "../bert_cased_vocab.txt");
    /**
     * @brief Initializes the neural network model from file.
     * @param bert_path Path to the file containing the traced model.
     */
    void init_bert(std::string bert_path = "../traced_text_classification_model.pt");
    /**
     * @brief Creates two maps (token to token ids and viceversa) from the vocabulary file.
     * @param vocab_path Path to the vocabulary file.
     * @return A pair of maps.
     */
    std::pair<std::map<std::string, int>, std::map<int, std::string>> get_vocab(std::string vocab_path);
    /**
     * @brief Loads the neural network model from file.
     * @param model_path Path to the neural network model.
     * @return The read model as a libtorch module.
     */
    torch::jit::script::Module load_model(std::string  model_path);

    /**
     * @brief Reads the dataset of notes to extract the class labels.
     * @param filePath Path to the file containing the sequence of notes.
     * @param considerChords Whether to consider chords while parsing the file.
     */
    void readNotesDataset(const std::string& filePath, bool considerChords);
    /**
     * @brief Utility function to concatenate strings from a vector into a single, longer string.
     * @param strings Strings to concatenate.
     * @return The concatenated string.
     */
    std::string concatenateStrings(const std::vector<std::string>& strings);
    /**
     * @brief Converts MIDI notes into their string representation in preparation for inference on the model.
     * @param noteEvents MIDI events to convert.
     * @param chordOffset Fraction of time in which to consecutive notes are considered a chord.
     * @param considerChords Whether to consider chords when creating notes.
     * @return A vector of strings containing the parsed MIDI notes' representation.
     */
    std::vector<std::string> createNotesForPrediction(std::vector<juce::MidiMessage> noteEvents, double chordOffset, bool considerChords);
    /**
     * @brief Converts notes from their integer value to actual MIDI messages.
     * @param midiNotes Vector containing the note's integer values (divided by chord).
     * @param noteDurationInQuarters Duration of each note in quarters.
     * @return A vector containing the parsed MIDI messages.
     */
    std::vector<juce::MidiMessage> midiNotesToMessages(const std::vector<std::vector<int>> midiNotes, double noteDurationInQuarters);
    /**
     * @brief Converts a string of notes into its integer representation.
     * @param notesString String containing notes.
     * @param octave Octave of the parsed notes.
     * @return A vector containing the MIDI integer representation of the notes.
     */
    std::vector<int> stringToMidiNotes(const std::string& notesString, int octave);
    /**
     * @brief Utility function to split a string around a given delimiter.
     * @param input String to split.
     * @param delimiter Character around which to split the string.
     * @return A vector containing the split strings.
     */
    std::vector<std::string> splitString(const std::string& input, const char delimiter);
};


#endif //BERTHOVEN_BERTHOVENMODEL_H
