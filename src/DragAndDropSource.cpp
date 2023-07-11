//
// Created by Riccardo Rossi on 10/07/23.
//

#include <__filesystem/path.h>
#include "DragAndDropSource.h"

/*
void DragAndDropSource::paint(juce::Graphics &g)
{
    g.fillAll (juce::Colours::blue);
    g.setColour(juce::Colours::red);
    g.fillRect(0,0,25,25);
}*/

void DragAndDropSource::mouseDown (const juce::MouseEvent& e)
{
std::cout << "TargetSource::mouseDown():" << std::endl;
}

void DragAndDropSource::mouseDrag (const juce::MouseEvent& e)
{
    juce::DragAndDropContainer* dragC =
            juce::DragAndDropContainer::findParentDragContainerFor(this);
    if (!dragC) {
        std::cout << "TargetSource::mouseDrag(): can't find parent drag container" << std::endl;
    } else {
        if (!dragC->isDragAndDropActive()) {
            dragC->startDragging("TargetSource", this);
            std::string pathToFile = std::filesystem::path(__FILE__).parent_path().string() + "/predicted_notes.mid";
            juce::File midiFile (pathToFile);
            if (!midiFile.existsAsFile()) {
                return;
            }
            juce::StringArray stringArray;
            stringArray.add(pathToFile);
            std::function<void()> callback = [&]{this->onDragStop();};
            juce::DragAndDropContainer::performExternalDragDropOfFiles(stringArray, false, nullptr, callback);
        }
    }
}

void DragAndDropSource::onDragStop() {
    setButtonText("Pick your MIDI from here!");
    setAlpha(0.8f);
}