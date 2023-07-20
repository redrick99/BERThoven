/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
BerthovenEditor::BerthovenEditor (BerthovenProcessor& p)
        : AudioProcessorEditor (&p), audioProcessor (p),
          miniPianoKbd{kbdState, juce::MidiKeyboardComponent::horizontalKeyboard}

{
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (800, 500);

    // Add GUI elements
    backgroundButton.setEnabled(false);
    numberOfNotesSlider.addListener(this);
    numberOfNotesSlider.setRange(1, 50, 1);
    numberOfNotesSlider.setTextValueSuffix(" n");
    numberOfNotesSliderLabel.setText("Number of notes", juce::dontSendNotification);

    noteDurationSlider.addListener(this);
    noteDurationSlider.setRange(0.5, 4, 0.5);
    noteDurationSlider.setTextValueSuffix(" q");
    noteDurationSliderLabel.setText("Note duration", juce::dontSendNotification);

    predictButton.addListener(this);
    predictButton.setButtonText("Predict");
    resetButton.addListener(this);
    resetButton.setButtonText("Reset");

    addAndMakeVisible(backgroundButton);
    addAndMakeVisible(numberOfNotesSlider);
    addAndMakeVisible(noteDurationSlider);
    addAndMakeVisible(numberOfNotesSliderLabel);
    addAndMakeVisible(noteDurationSliderLabel);
    addAndMakeVisible(predictButton);
    addAndMakeVisible(resetButton);
    addAndMakeVisible(dragAndDropTarget);
    addAndMakeVisible(dragAndDropSource);

    dragAndDropSource.setButtonText("Pick your MIDI from here!");
    dragAndDropTarget.setButtonText("Drop your MIDI file here!");
    numberOfNotesSlider.setColour(juce::Label::textWhenEditingColourId, juce::Colours::black);
    dragAndDropSource.setAlpha(0.8f);
    dragAndDropTarget.setAlpha(0.8f);

    // listen to the mini piano
    kbdState.addListener(this);
    addAndMakeVisible(miniPianoKbd);
    std::filesystem::path currentPath = std::filesystem::path(__FILE__).parent_path();
    auto imageFile = juce::File(currentPath.string()+"/assets/Beethoven.png");
    background = juce::ImageCache::getFromFile(imageFile);
}

BerthovenEditor::~BerthovenEditor()
{
}

//==============================================================================
void BerthovenEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.drawImageWithin(background, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::stretchToFit);

    // g.setColour (juce::Colours::white);
    // g.setFont (15.0f);
    // g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
}

void BerthovenEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor...
    float rowHeight = getHeight()/6;
    float colWidth = getWidth() / 3;
    float paddingX = 50;
    float paddingY = 10;
    float row = 0;
    float height = paddingY;

    numberOfNotesSliderLabel.setBounds(paddingX, height, colWidth, rowHeight/2);
    numberOfNotesSlider.setBounds(paddingX, height + 20, colWidth, rowHeight/2);
    height += rowHeight;
    noteDurationSliderLabel.setBounds(paddingX, height, colWidth, rowHeight/2);
    noteDurationSlider.setBounds(paddingX, height + 20, colWidth, rowHeight/2);
    height += rowHeight;
    dragAndDropTarget.setBounds(paddingX, height, colWidth, rowHeight - 20);
    height += paddingY + rowHeight - 20;
    dragAndDropSource.setBounds(paddingX, height, colWidth, rowHeight - 20);
    height += paddingY + rowHeight - 20;
    resetButton.setBounds(paddingX, height, colWidth/2-2, rowHeight/2 + 40);
    predictButton.setBounds(paddingX+2+colWidth/2, height, colWidth/2-2, rowHeight/2 + 40);

    backgroundButton.setBounds(-5, -5, colWidth+2*paddingX+5, 5*rowHeight+15);

    miniPianoKbd.setBounds(0, 5*rowHeight+2, getWidth(), rowHeight);
}

void BerthovenEditor::sliderValueChanged (juce::Slider *slider)
{
    if (slider == &numberOfNotesSlider) {
        numberOfNotesToPredict = static_cast<int>(numberOfNotesSlider.getValue());
    }
    if (slider == &noteDurationSlider) {
        noteDurationInQuarters = noteDurationSlider.getValue();
    }
}

void BerthovenEditor::buttonClicked(juce::Button* btn)
{
    if (btn == &predictButton) {
        predictButton.setButtonText("Predicting...");
        toggleEnableComponents(false);
        std::vector<juce::MidiMessage> externalMidis = dragAndDropTarget.getMidiMessages();
        audioProcessor.predict(numberOfNotesToPredict, noteDurationInQuarters, externalMidis);
        dragAndDropTarget.setButtonText("Drop your MIDI file here!");
        dragAndDropSource.setButtonText("MIDI ready to be exported!");
        dragAndDropTarget.setAlpha(0.8f);
        dragAndDropSource.setAlpha(1.0f);
        predictButton.setButtonText("Predict");
        toggleEnableComponents(true);
    }
    if (btn == &resetButton) {
        audioProcessor.resetModel();
    }
}

void BerthovenEditor::handleNoteOn(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg1 = juce::MidiMessage::noteOn(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg1, 0);

}

void BerthovenEditor::handleNoteOff(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg2 = juce::MidiMessage::noteOff(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg2, 0);
}

void BerthovenEditor::toggleEnableComponents(bool enabled)
{
    predictButton.setEnabled(enabled);
    dragAndDropTarget.setEnabled(enabled);
    dragAndDropSource.setEnabled(enabled);
    numberOfNotesSlider.setEnabled(enabled);
    noteDurationSlider.setEnabled(enabled);
}
