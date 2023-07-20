/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <ctime>
#include <chrono>
#include "BerthovenModel.h"


//==============================================================================
/**
*/
class BerthovenProcessor  : public juce::AudioProcessor
#if JucePlugin_Enable_ARA
    , public juce::AudioProcessorARAExtension
#endif
{
public:
    //==============================================================================
    BerthovenProcessor();
    ~BerthovenProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
#endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;
    /** add some midi to be played at the sent sample offset*/
    void addMidi(juce::MidiMessage msg, int sampleOffset);

    /**
     * @brief Predicts a number of notes using BERThoven's model.
     * @param numberOfNotesToPredict Specified by the user.
     * @param noteDurationInQuarters Duration of each predicted note in quarters.
     * @param externalMessages Vector containing external messages if the user dropped a MIDI file on the GUI.
     */
    void predict(int numberOfNotesToPredict, double noteDurationInQuarters, std::vector<juce::MidiMessage> externalMessages);
    /**
     * @brief Resets the neural network model to initial conditions.
     */
    void resetModel();
private:
    /// Model used for prediction.
    BerthovenModel model;
    double ticksPerQuarterNote;
    /// Temporary file on which to store predicted notes.
    juce::File predictedNotesFile;

    /** stores messages added from the addMidi function*/
    juce::MidiBuffer midiToProcess;

    std::chrono::high_resolution_clock::time_point startTime;

    /**
     * Creates a MIDI file from a vector of MIDI events.
     * @param midiEvents MIDI events from which to create the file.
     */
    void createMidiFile(const std::vector<juce::MidiMessage>& midiEvents);

    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (BerthovenProcessor)
};
