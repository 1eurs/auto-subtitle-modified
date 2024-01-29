import os
import ffmpeg
import whisper
import warnings
import tempfile
from .utils import filename, str2bool, write_srt

class VideoTranscriber:
    def __init__(self, model='small', output_dir='.', output_srt=False, srt_only=False, verbose=False, task='transcribe'):
        self.model_name = model
        self.output_dir = output_dir
        self.output_srt = output_srt
        self.srt_only = srt_only
        self.verbose = verbose
        self.task = task
        os.makedirs(output_dir, exist_ok=True)

        if self.model_name.endswith(".en"):
            warnings.warn(f"{self.model_name} is an English-only model, forcing English detection.")
            self.language = "en"
        else:
            self.language = None

        self.model = whisper.load_model(self.model_name)

    def transcribe_video(self, video_paths):
        audios = self._get_audio(video_paths)
        subtitles = self._get_subtitles(audios)

        if not self.srt_only:
            self._add_subtitles_to_video(subtitles)

        return subtitles

    def _get_audio(self, paths):
        temp_dir = tempfile.gettempdir()
        audio_paths = {}

        for path in paths:
            if self.verbose:
                print(f"Extracting audio from {filename(path)}...")
            output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

            ffmpeg.input(path).output(output_path, acodec="pcm_s16le", ac=1, ar="16k").run(quiet=not self.verbose, overwrite_output=True)
            audio_paths[path] = output_path

        return audio_paths

    def _get_subtitles(self, audio_paths):
        srt_path = self.output_dir if self.output_srt else tempfile.gettempdir()
        subtitles_path = {}

        for path, audio_path in audio_paths.items():
            srt_file_path = os.path.join(srt_path, f"{filename(path)}.srt")

            if self.verbose:
                print(f"Generating subtitles for {filename(path)}... This might take a while.")

            warnings.filterwarnings("ignore")
            result = self.model.transcribe(audio_path, task=self.task, language=self.language)
            warnings.filterwarnings("default")

            with open(srt_file_path, "w", encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt)

            subtitles_path[path] = srt_file_path

        return subtitles_path

    def _add_subtitles_to_video(self, subtitles):
        for path, srt_path in subtitles.items():
            out_path = os.path.join(self.output_dir, f"{filename(path)}.mp4")
            if self.verbose:
                print(f"Adding subtitles to {filename(path)}...")

            video = ffmpeg.input(path)
            audio = video.audio

            ffmpeg.concat(
                video.filter('subtitles', srt_path, force_style="OutlineColour=&H40000000,BorderStyle=3"), audio, v=1, a=1
            ).output(out_path).run(quiet=not self.verbose, overwrite_output=True)

            if self.verbose:
                print(f"Saved subtitled video to {os.path.abspath(out_path)}.")
