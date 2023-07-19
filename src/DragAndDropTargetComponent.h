//
// Created by Riccardo Rossi on 10/07/23.
//

#ifndef BERTHOVEN_DRAGANDDROPTARGETCOMPONENT_H
#define BERTHOVEN_DRAGANDDROPTARGETCOMPONENT_H

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>

/**
 * @class DragAndDropTargetComponent
 * @brief Defines a GUI component where files can be dropped.
 */
class DragAndDropTargetComponent : public juce::TextButton,
                                   public juce::FileDragAndDropTarget
{
public:
    // void paint(juce::Graphics &g) override;
    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void filesDropped (const juce::StringArray& files, int x, int y) override;

    void fileDragEnter (const juce::StringArray& files, int x, int y) override;
    void fileDragExit (const juce::StringArray& files) override;

    std::vector<juce::MidiMessage> getMidiMessages();

private:
    std::vector<juce::MidiMessage> midiMessages;

};


#endif //BERTHOVEN_DRAGANDDROPTARGETCOMPONENT_H
