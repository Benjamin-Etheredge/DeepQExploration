import streamlit as st
import mlflow
import os
#import tensorflow as tf
from pathlib import Path
import re
import ffmpeg
import uuid


def find_best_run(exp_id, metric='best_off_policy_score'):
   client = mlflow.tracking.MlflowClient()
   run_infos = client.list_run_infos(exp_id)

   best_metric = 0
   best_run_id = None
   for run_info in run_infos:
      try:
         run = client.get_run(run_info.run_id)
         #st.write(run)
         metric_value = run.data.metrics[metric]
         if metric_value > best_metric:
            best_metric = metric_value
            best_run_id = run_info.run_id
      except KeyError:
         pass

   return best_run_id

import shutil

#@st.cache(allow_output_muitation=True)
def get_recordings(exp_name):
   client = mlflow.tracking.MlflowClient()
   experiment = client.get_experiment_by_name(exp_name)
   #st.write(experiment)
   exp_id = experiment.experiment_id
   best_run_id = find_best_run(exp_id)

   shutil.rmtree('/tmp/videos', ignore_errors=True)
   os.mkdir('/tmp/videos')
   videos = client.download_artifacts('468b5809130845489da3170239aa1bcd', 'off_policy_highlights', dst_path='/tmp/videos')
   #model_path = client.download_artifacts(best_run_id, 'best_model', dst_path='/tmp/')
   return videos

   #st.write(run_infos[0])
   # TODO query for this
   #model_artifiact = client.download_artifacts(os.environ['RUN_ID'], 'best_on_policy_model.h5'], dst_path='/tmp')
   #model = tf.keras.load_model(model_artifiact)
   #return model

iteration_finder = re.compile(r'(\d+).mp4$')
def get_iteration(path):
      return iteration_finder.search(str(path)).group(1)

def get_score_history(exp_name):
   client = mlflow.tracking.MlflowClient()
   experiment = client.get_experiment_by_name(exp_name)
   #st.write(experiment)
   exp_id = experiment.experiment_id
   #best_run_id = find_best_run(exp_id)

   run_id = '468b5809130845489da3170239aa1bcd'
   data = client.get_metric_history(run_id, 'off_policy_hightlight_score')
   data = [item.value for item in data]
   return data

def create_highlights(exp_name):
   videos_path = get_recordings(exp_name)
   scores = get_score_history(exp_name)
   return combine_videos(videos_path, scores)

def combine_videos(videos, scores):
   sorted_videos = sorted(Path(videos).iterdir())
   sorted_videos = [video for video in sorted_videos if "meta" not in str(video)]

   indices = (get_iteration(v) for v in sorted_videos)
   inputs = [ffmpeg.input(str(v)) for v in sorted_videos]
   texted = [ffmpeg.drawtext(v, text=f"{idx} - {score}", x='(w-tw)/2', y='(h-th)', fontcolor='white@1.0', box=1, boxcolor='black@1.0', fontfile='ttf/Hack-Bold.ttf') for idx, v, score in zip(indices, inputs, scores)]
   #texted = [ffmpeg.drawtext(v, text=f"{idx} - {score}", x='(w-tw)/2', y='(h-th)', fontcolor='white@1.0', box=1, boxcolor='black@1.0', fontfile='ttf/Hack-Regular.ttf') for idx, v, score in zip(indices, inputs, scores)]
   joined = ffmpeg.concat(*texted)

   out_file = f"/tmp/{uuid.uuid1()}.mp4"
   out = ffmpeg.overwrite_output(ffmpeg.output(joined, out_file))
   out.run()
   return out_file



st.title("Deep-Q Experiments")
#cleaned_videos =  texted

# TODO
#audio = ffmpeg.input('https://www.youtube.com/watch?v=btPJPFnesV4')
#audio_out = ffmpeg.overwrite_output(ffmpeg.output(audio, '/tmp/test.mp4'))
#audio_out.run()
#st.video('/tmp/text.mp4')


for exp_name in ['breakout']:
   with st.beta_expander(exp_name):
      file = create_highlights(exp_name)
      st.video(file)

#test = concatentate_videoclips([str(v) for v in latest_video])
#test.write_videofile('hall_of_fame.mp4')
#st.video('hall_of_fame.mp4')
#txt_clip = ( TextClip("test", fontsize=70,color='white')
             #.set_position('bottem')
             #.set_duration(10) )