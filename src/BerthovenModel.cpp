//
// Created by Riccardo Rossi on 09/07/23.
//

#include "BerthovenModel.h"

BerthovenModel::BerthovenModel() {
    std::filesystem::path currentPath = std::filesystem::path(__FILE__).parent_path();
    int seed = 42;
    torch::manual_seed(seed);
    torch::cuda::manual_seed(seed);
    readNotesDataset(currentPath.string()+pathToNotesFile, considerChords);
    init_vocab(currentPath.string()+pathToVocab);
    init_bert(currentPath.string()+pathToModule);
}

BerthovenModel::~BerthovenModel() {

}

void BerthovenModel::putMidiEvent(juce::MidiMessage &event) {
    if (!event.isNoteOn())
        return;

    mtx.lock();
    try {
        storedMidiEvents.push_back(event);
        std::cout << "Added MIDI note: " << event.getNoteNumber()%12 << std::endl;
    }
    catch(...) {
        std::cout << "BerthovenModel::putMidiEvent crashed... catching" << std::endl;
    }
    mtx.unlock();
}

void BerthovenModel::clearMidiEvents() {
    mtx.lock();
    try {
        if(!storedMidiEvents.empty())
        {
            storedMidiEvents.clear();
        }
    }
    catch(...) {
        std::cout << "BerthovenModel::clearMidiEvents crashed... catching" << std::endl;
    }
    mtx.unlock();
}

std::vector<juce::MidiMessage> BerthovenModel::predict(int numberOfNotesToPredict, double noteDurationInSeconds) {
    std::vector<juce::MidiMessage> input;
    std::vector<juce::MidiMessage> output;
    if (storedMidiEvents.size() < max_length) {
        std::cout<<"Not enough notes for prediction!\n";
        return output;
    }

    std::vector<std::string> notes = createNotesForPrediction(storedMidiEvents, noteOffsetForChord, considerChords);
    std::vector<std::string> predictedNotes;
    std::vector<std::vector<int>> predictedNoteInt;

    for (int i = 0; i < numberOfNotesToPredict; i++) {
        notes = std::vector<std::string>(notes.end() - max_length, notes.end());
        std::string text = concatenateStrings(notes);

        std::cout << "Text: " << text << std::endl;

        torch::Tensor input_ids, masks;
        std::tie(input_ids, masks) = preprocess(text, token2id, max_length_tokenizer, false);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids);
        inputs.push_back(masks);

        auto outputs = bert.forward(inputs).toTensor();
        std::string predictedNote = pred2class[int(outputs.argmax().item<int>())];
        std::cout << "Prediction: " << predictedNote << std::endl;

        notes.push_back(predictedNote);
        predictedNotes.push_back(predictedNote);
        predictedNoteInt.push_back(stringToMidiNotes(predictedNote, octave));
    }

    output = midiNotesToMessages(predictedNoteInt, noteDurationInSeconds);

    for (auto midi : std::vector<juce::MidiMessage>(output)) {
        if (midi.isNoteOn()) {
            midi.setTimeStamp(midi.getTimeStamp() + static_cast<double>(std::time(nullptr)));
            storedMidiEvents.push_back(midi);
        }
    }

    return output;
}

void BerthovenModel::reset() {
    clearMidiEvents();
    double t = 0.0;
    for (int i = 0; i < max_length; i++) {
        auto midiMsg = juce::MidiMessage::noteOn(1, std::stoi(firstNotes[i]), 1.0f);
        midiMsg.setTimeStamp(t);
        t += 0.5;
        storedMidiEvents.push_back(midiMsg);
    }
}

std::pair<torch::Tensor, torch::Tensor> BerthovenModel::preprocess(std::string text, std::map<std::string, int> token2id, int max_length, bool log){
    std::string pad_token = "[PAD]", start_token = "[CLS]", end_token = "[SEP]";
    int pad_token_id = token2id[pad_token], start_token_id = token2id[start_token], end_token_id = token2id[end_token];

    std::vector<int> input_ids(max_length, pad_token_id), masks(max_length, 0);
    input_ids[0] = start_token_id; masks[0] = 1;

    std::string word;
    std::istringstream ss(text);

    int input_id = 1;
    while(getline(ss, word, ' ')) {
        int word_id = token2id[word];
        masks[input_id] = 1;
        input_ids[input_id++] = word_id;

        if (log)
            std::cout << word << " : " << word_id << '\n';
    }
    masks[input_id] = 1;
    input_ids[input_id] = end_token_id;

    if (log){
        for (auto i : input_ids)
            std::cout << i << ' ';
        std::cout << '\n';

        for (auto i : masks)
            std::cout << i << ' ';
        std::cout << '\n';
    }

    auto input_ids_tensor = torch::tensor(input_ids).unsqueeze(0);
    auto masks_tensor = torch::tensor(masks).unsqueeze(0).unsqueeze(0);
    return std::make_pair(input_ids_tensor, masks_tensor);
}

void BerthovenModel::init_vocab(std::string vocab_path) {
    std::tie(token2id, id2token) = get_vocab(vocab_path);
}

void BerthovenModel::init_bert(std::string bert_path) {
    bert = load_model(bert_path);
}

std::pair<std::map<std::string, int>, std::map<int, std::string>> BerthovenModel::get_vocab(std::string vocab_path) {
    std::map<std::string, int> token2id;
    std::map<int, std::string> id2token;

    std::fstream newfile;
    newfile.open(vocab_path, std::ios::in);

    std::string line;
    while(getline(newfile, line)){
        char *token = strtok(const_cast<char*>(line.c_str()), " ");
        char *token_id = strtok(nullptr, " ");

        token2id[token] = std::stoi(token_id);
        id2token[std::stoi(token_id)] = token;
    }
    newfile.close();

    return std::make_pair(token2id, id2token);
}

torch::jit::script::Module BerthovenModel::load_model(std::string model_path) {
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
    return module;
}

void BerthovenModel::readNotesDataset(const std::string &filePath, bool considerChords) {
    std::vector<std::string> notes;
    pred2class.clear();

    std::ifstream vocabularyFile(filePath);
    if (vocabularyFile.is_open()) {
        std::string line;
        while (std::getline(vocabularyFile, line)) {
            if (line != "[" && line != "]") {
                size_t startPos = line.find('"');
                size_t endPos = line.rfind('"');

                if (startPos != std::string::npos && endPos != std::string::npos && startPos < endPos) {
                    std::string note = line.substr(startPos + 1, endPos - startPos - 1);
                    if (!considerChords && note.find('.') == std::string::npos) {
                        notes.push_back(note);
                    }
                } else {
                    throw std::invalid_argument("Error while parsing vocabulary");
                }
            }
        }
    }

    std::vector<std::string> fNotes(notes);
    fNotes.resize(static_cast<unsigned long>(max_length));
    firstNotes = fNotes;

    reset();

    std::sort(notes.begin(), notes.end());
    notes.erase(std::unique(notes.begin(), notes.end()), notes.end());

    for (int i = 0; i < notes.size(); i++) {
        pred2class[i] = notes[i];
    }
}

std::string BerthovenModel::concatenateStrings(const std::vector<std::string>& strings) {
    std::ostringstream oss;
    std::copy(strings.begin(), strings.end() - 1, std::ostream_iterator<std::string>(oss, " "));
    oss << strings.back();
    return oss.str();
}

std::vector<std::string> BerthovenModel::createNotesForPrediction(std::vector<juce::MidiMessage> noteEvents, double chordOffset, bool considerChords) {
    std::vector<std::string> predictionNotes;

    std::string firstNote = std::to_string(noteEvents[0].getNoteNumber() % 12);
    std::string notesOfEvent = firstNote;

    for (int i = 1; i < noteEvents.size(); i++) {
        std::string note = std::to_string(noteEvents[i].getNoteNumber() % 12);
        if(considerChords && noteEvents[i].getTimeStamp() - noteEvents[i-1].getTimeStamp() <= chordOffset) {
            notesOfEvent += ("." + note);
        }
        else {
            predictionNotes.push_back(notesOfEvent);
            notesOfEvent = note;
        }
    }
    predictionNotes.push_back(notesOfEvent);

    return predictionNotes;
}

std::vector<juce::MidiMessage> BerthovenModel::midiNotesToMessages(const std::vector<std::vector<int>> midiNotes, double noteDurationInSeconds) {
    std::vector<juce::MidiMessage> midiMessages;
    double timeOfEvent = 0;
    double secondsPerQuarterNote = 60.0 / 90.0;
    double durationTicks = noteDurationInSeconds * secondsPerQuarterNote * 90.0;

    for (const std::vector<int>& chord : midiNotes) {
        std::vector<juce::MidiMessage> notesInChord;
        for (int note : chord) {
            juce::MidiMessage midiMessage = juce::MidiMessage::noteOn(1, note, 1.0f);
            juce::MidiMessage noteOff = juce::MidiMessage::noteOff(1, note);
            midiMessage.setTimeStamp(timeOfEvent);
            noteOff.setTimeStamp(timeOfEvent + durationTicks);
            midiMessages.push_back(midiMessage);
            midiMessages.push_back(noteOff);
        }
        timeOfEvent += durationTicks;
    }

    return midiMessages;
}

std::vector<int> BerthovenModel::stringToMidiNotes(const std::string &notesString, int octave) {
    std::vector<std::string> notes = splitString(notesString, '.');
    std::vector<int> midiNotes;

    for (std::string note : notes) {
        int midiNote = std::stoi(note) + (12 * octave);
        midiNotes.push_back(midiNote);
    }

    return midiNotes;
}

std::vector<std::string> BerthovenModel::splitString(const std::string &input, const char delimiter) {
    std::vector<std::string> substrings;
    std::string substring;
    std::istringstream iss(input);

    while (std::getline(iss, substring, delimiter)) {
        substrings.push_back(substring);
    }

    return substrings;
}
