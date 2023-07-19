/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include "PluginProcessor.h"
#include "DragAndDropSource.h"
#include "DragAndDropTargetComponent.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>


//==============================================================================
/**
*/
class BerthovenEditor  :   public juce::AudioProcessorEditor,
        // listen to buttons
                            public juce::Button::Listener,
        // listen to sliders
                            public juce::Slider::Listener,

                            public juce::DragAndDropContainer,
        // listen to piano keyboard widget
                            private juce::MidiKeyboardState::Listener


{
public:
    BerthovenEditor (BerthovenProcessor&);
    ~BerthovenEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void sliderValueChanged (juce::Slider *slider) override;
    void buttonClicked(juce::Button* btn) override;
    // from MidiKeyboardState
    void handleNoteOn(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float
    velocity) override;
    // from MidiKeyboardState
    void handleNoteOff(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity) override;

private:
    /// Background image of Beethoven.
    juce::Image background;
    double noteDurationInSeconds = 0.5;

    // needed for the mini piano keyboard
    juce::MidiKeyboardState kbdState;
    juce::MidiKeyboardComponent miniPianoKbd;

    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    BerthovenProcessor& audioProcessor;

    // GUI widgets
    juce::TextButton backgroundButton;
    juce::TextButton predictButton;
    juce::TextButton resetButton;
    juce::Slider numberOfNotesSlider;
    juce::Slider noteDurationSlider;
    juce::Label numberOfNotesSliderLabel;
    juce::Label noteDurationSliderLabel;
    DragAndDropSource dragAndDropSource;
    DragAndDropTargetComponent dragAndDropTarget;

    /// Toggles all GUI components to be enabled or disabled.
    void toggleEnableComponents(bool enabled);

    // Parameters
    int numberOfNotesToPredict = 1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BerthovenEditor)
};
