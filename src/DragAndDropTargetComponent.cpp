//
// Created by Riccardo Rossi on 10/07/23.
//

#include "DragAndDropTargetComponent.h"

bool DragAndDropTargetComponent::isInterestedInFileDrag(const juce::StringArray& files) {
    for (auto file : files) {
        if (file.contains(".mid"))
            return true;
    }

    return false;
}

void DragAndDropTargetComponent::filesDropped(const juce::StringArray &files, int x, int y) {
    for (const auto& file : files) {
        if (isInterestedInFileDrag(files)) {
            std::vector<juce::MidiMessage> foundMessages;

            juce::File midiRawFile = juce::File::createFileWithoutCheckingPath(file);
            juce::FileInputStream midiStream (midiRawFile);
            juce::MidiFile midiFile;
            midiFile.readFrom(midiStream);
            for (int i = 0; i < midiFile.getNumTracks(); i++) {
                auto track = midiFile.getTrack(i);
                for (int j = 0; j < track->getNumEvents(); j++) {
                    auto event = track->getEventPointer(j)->message;
                    if (event.isNoteOn()) {
                        foundMessages.push_back(event);
                    }
                }
            }
            setButtonText("Read MIDI file!");
            midiMessages = foundMessages;
            return;
        }
    }
}

void DragAndDropTargetComponent::fileDragEnter(const juce::StringArray &files, int x, int y) {
    for (const auto& file : files) {
        if (isInterestedInFileDrag(file)) {
            setAlpha(1.0f);
            return;
        }
    }
}

void DragAndDropTargetComponent::fileDragExit(const juce::StringArray &files) {
    for (const auto& file : files) {
        if (isInterestedInFileDrag(file)) {
            setAlpha(0.8f);
            return;
        }
    }
}

std::vector<juce::MidiMessage> DragAndDropTargetComponent::getMidiMessages() {
    return midiMessages;
}