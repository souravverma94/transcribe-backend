from pydub import AudioSegment
from pydub.utils import make_chunks
import os


def chunking(filename):
    my_audio = AudioSegment.from_file("{}.wav".format(filename), "wav")
    chunk_length_ms = 63000  # pydub calculates in millisec
    chunks = make_chunks(my_audio, chunk_length_ms)  # Make chunks of 63 sec

    # Export all of the individual chunks as wav files
    try:
        os.mkdir('audio_chunk')
    except FileExistsError:
        pass
        # move into the directory to
        # store the audio files.
    os.chdir('audio_chunk')
    for i, chunk in enumerate(chunks):
        chunk_name = "./chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")


def convert(filename, filetype="mp4"):
    audio = AudioSegment.from_file(filename, filetype)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export("{}.wav".format(filename), format="wav", bitrate="192k")


if __name__ == '__main__':
    convert('obama_speech.mp4')
    # chunking('mit-lec.mp4')