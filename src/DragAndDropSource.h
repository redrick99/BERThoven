//
// Created by Riccardo Rossi on 10/07/23.
//

#ifndef BERTHOVEN_DRAGANDDROPSOURCE_H
#define BERTHOVEN_DRAGANDDROPSOURCE_H

#include <juce_gui_basics/juce_gui_basics.h>

/**
 * @class DragAndDropSource
 * @brief Defines a region from which files and components can be grabbed on the GUI.
 */
class DragAndDropSource : public juce::TextButton {
public:
    // void paint(juce::Graphics &g) override;

    void mouseDown (const juce::MouseEvent& e) override;

    void mouseDrag (const juce::MouseEvent& e) override;

    void onDragStop();
};


#endif //BERTHOVEN_DRAGANDDROPSOURCE_H
