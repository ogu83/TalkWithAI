from google.cloud import translate_v2 as translate
from google.cloud import texttospeech as texttospeech
from google.cloud import speech
import six
import os
import openai
import playsound
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import wavio as wv

GOOGLE_APPLICATION_CREDENTIALS = './translate-377120-a021c8ac2dc6.json'
VOICE_SAMPLING = 16000
AI_MODEL = "text-davinci-003"
MAX_TOKENS = 250
RECORD_LEN = 8

def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
    openai.api_key = os.getenv("OPENAIKEY")
    
    print("Recording Started.")
    record_audio()
    print("Recording Completed")
    # playsound.playsound('./input.wav', True)
    
    myInput = transcribe_file()
    print("Input Transcribed")

    myInputEn = translate_text("tr", "en", myInput)
    myInputEn = myInputEn['translatedText']
    print("Input Translated to English")
    
    response = openai.Completion.create(model=AI_MODEL, prompt=myInputEn, temperature=0, top_p=1, max_tokens=MAX_TOKENS)
    myOutputEn = response['choices'][0]['text']
    print("Output Generated")

    myOutput = translate_text("en", "tr", myOutputEn)
    myOutput = myOutput['translatedText']
    print("Output Translated to Turkish")
    # print(myOutput)

    text2mp3(myOutput, 'tr-TR')
    print("Output Sound File Generated")
    
    play_audio()

def play_audio(file = 'output.wav'):
    # Extract data and sampling rate from file
    array, smp_rt = sf.read(file, dtype = 'float32') 
    # start the playback
    sd.play(array, smp_rt)
    # Wait until file is done playing
    status = sd.wait()     
    # stop the sound
    sd.stop()

def record_audio(duration = RECORD_LEN, frequency = VOICE_SAMPLING, channels=1):
    # Sampling frequency
    # to record audio from
    # sound-device into a Numpy
    recording = sd.rec(int(duration * frequency),
                    samplerate = frequency, channels = channels)
    
    # Wait for the audio to complete
    sd.wait()

    wv.write("input.wav", recording, frequency, sampwidth=2)

def transcribe_file(speech_file="./input.wav", lang = 'tr-TR'):
    """Transcribe the given audio file asynchronously."""
    from google.cloud import speech

    client = speech.SpeechClient()

    with open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    """
     Note that transcription is limited to a 60 seconds audio file.
     Use a GCS file for audio longer than 1 minute.
    """
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        audio_channel_count=1,
        sample_rate_hertz=VOICE_SAMPLING,
        language_code=lang,
    )
    operation = client.long_running_recognize(config=config, audio=audio)

    # print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    result = response.results[0]
    transcript = result.alternatives[0].transcript
    return transcript

    # # Each result is for a consecutive portion of the audio. Iterate through
    # # them to get the transcripts for the entire audio file.
    # for result in response.results:
    #     # The first alternative is the most likely one for this portion.
    #     print("Transcript: {}".format(result.alternatives[0].transcript))
    #     print("Confidence: {}".format(result.alternatives[0].confidence))


'''Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
'''
def translate_text(source, target, text):
    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target, source_language=source)
    return result;
    # print(u"Text: {}".format(result["input"]))
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))

def text2mp3(text, lang = 'en-US', file="output.wav"):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz = VOICE_SAMPLING
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open(file, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)

if __name__ == "__main__":
    main()