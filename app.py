import time
import os
from datetime import datetime
from flask import Flask, jsonify, Response
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp
from pydub import AudioSegment



app = Flask(__name__)


def generateText(str):
    for word in str.split():
        yield '{} '.format(word)
        time.sleep(0.5)


def convert(filename, filetype="mp4"):
    res_file = "{}.wav".format(filename)
    if os.path.exists(res_file) and os.path.isfile(res_file):
        print('already exists')
        return
    audio = AudioSegment.from_file(filename, filetype)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(res_file, format="wav", bitrate="192k")


@app.route('/user/<name>')
def hello_world(name):
    return Response("Hello, {}!".format(name), mimetype='text/plain')
    # return jsonify({"about": "Hello, {}!".format(name)})


@app.route('/time')
def doyouhavethetime():
    def generate():
        for i in range(4):
            yield "{}\n".format(datetime.now().isoformat())
            time.sleep(1)

    return Response(generate(), mimetype='text/plain')


@app.route('/transcribe/<filename>')
def transcribe(filename):
    convert(filename)
    Audio_sample = "{}.wav".format(filename)
    # Speech Recognition model - QuartzNet
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En").cuda()
    # # Punctuation and capitalization model
    punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
        model_name='punctuation_en_distilbert').cuda()
    # # Convert our audio sample to text
    files = [Audio_sample]
    raw_text = ''
    text = ''
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        raw_text = transcription
    # Add capitalization and punctuation
    res = punctuation.add_punctuation_capitalization(queries=[raw_text])
    print(res)
    text = res[0]
    return jsonify({"raw_text": raw_text, "enhanced_text": text})


@app.route('/transcribe2/<filename>')
def transcribe2(filename):
    Audio_sample = filename
    # Speech Recognition model - QuartzNet
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En").cuda()
    # Punctuation and capitalization model
    punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
        model_name='punctuation_en_distilbert').cuda()
    # # Convert our audio sample to text
    files = [Audio_sample]
    raw_text = ''
    text = ''
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        raw_text = transcription
    # Add capitalization and punctuation
    res = punctuation.add_punctuation_capitalization(queries=[raw_text])
    text = res[0]
    # def generateText(str):
    #     for word in str.split():
    #         yield '{} '.format(word)
    #         time.sleep(0.4)
    return Response(generateText(text), mimetype='text/plain')


@app.route('/transcribe3/<filename>')
def transcribe3(filename):
    Audio_sample = filename
    # Speech Recognition model - QuartzNet
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En").cuda()
    # Punctuation and capitalization model
    punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(
        model_name='punctuation_en_distilbert').cuda()
    # # Convert our audio sample to text
    files = [Audio_sample]
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        raw_text = ''
        text = ''
        raw_text = transcription
        # Add capitalization and punctuation
        res = punctuation.add_punctuation_capitalization(queries=[raw_text])
        text = res[0]
    return Response(generateText(text), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
