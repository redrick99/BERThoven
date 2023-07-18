/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
BerthovenProcessor::BerthovenProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
        : AudioProcessor (BusesProperties()
#if ! JucePlugin_IsMidiEffect
        #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
#endif
)
#endif
, model {}
{
    startTime = std::chrono::high_resolution_clock::now();
    std::string path = std::filesystem::path(__FILE__).parent_path().string() + "/predicted_notes.mid";
    juce::File midiRawFile = juce::File::createFileWithoutCheckingPath(path);
    if (midiRawFile.existsAsFile()) {
        midiRawFile.deleteFile();
    }
}


BerthovenProcessor::~BerthovenProcessor()
{
}

//==============================================================================
const juce::String BerthovenProcessor::getName() const
{
    return JucePlugin_Name;
}

bool BerthovenProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool BerthovenProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool BerthovenProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double BerthovenProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int BerthovenProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
    // so this should be at least 1, even if you're not really implementing programs.
}

int BerthovenProcessor::getCurrentProgram()
{
    return 0;
}

void BerthovenProcessor::setCurrentProgram (int index)
{
}

const juce::String BerthovenProcessor::getProgramName (int index)
{
    return {};
}

void BerthovenProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void BerthovenProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{

}

void BerthovenProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool BerthovenProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
#else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
#endif
}
#endif

void BerthovenProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    if (midiToProcess.getNumEvents() > 0){
        midiMessages.addEvents(midiToProcess, midiToProcess.getFirstEventTime(), midiToProcess.getLastEventTime()+1, 0);
        midiToProcess.clear();
    }

    juce::MidiBuffer generatedMessages{};

    for (const auto metadata : midiMessages){
        auto message = metadata.getMessage();
        if (message.isNoteOn()){

        }
        if (message.isNoteOff()){

        }
    }
    // optionally wipe out the original messages
    midiMessages.clear();

    midiMessages.addEvents(generatedMessages, generatedMessages.getFirstEventTime(), -1, 0);

}

//==============================================================================
bool BerthovenProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* BerthovenProcessor::createEditor()
{
    return new BerthovenEditor(*this);
}

//==============================================================================
void BerthovenProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void BerthovenProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new BerthovenProcessor();
}


void BerthovenProcessor::addMidi(juce::MidiMessage msg, int sampleOffset)
{
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = endTime - startTime;
    msg.setTimeStamp(duration.count());
    midiToProcess.addEvent(msg, sampleOffset);
    if (msg.isNoteOn())
        model.putMidiEvent(msg);
}

void BerthovenProcessor::predict(int numberOfNotesToPredict, double noteDurationInSeconds, std::vector<juce::MidiMessage> externalMessages)
{
    for (auto midiEvent : externalMessages) {
        model.putMidiEvent(midiEvent);
    }
    std::vector<juce::MidiMessage> predictedNotes = model.predict(numberOfNotesToPredict, noteDurationInSeconds);
    createMidiFile(predictedNotes);
}

void BerthovenProcessor::resetModel() {
    model.reset();
}

void BerthovenProcessor::createMidiFile(const std::vector<juce::MidiMessage>& midiEvents) {
    if (midiEvents.empty())
        return;

    juce::MidiMessageSequence sequence;
    juce::MidiFile file;
    for (const auto& event : midiEvents) {
        sequence.addEvent(event);
    }
    sequence.updateMatchedPairs();

    std::string path = std::filesystem::path(__FILE__).parent_path().string() + "/predicted_notes.mid";
    juce::File midiRawFile = juce::File::createFileWithoutCheckingPath(path);
    if (midiRawFile.existsAsFile()) {
        midiRawFile.deleteFile();
    }
    juce::FileOutputStream stream (midiRawFile);
    file.clear();
    file.setTicksPerQuarterNote(90);
    file.addTrack(sequence);
    file.writeTo(stream);
}
