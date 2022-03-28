import os
import deeplabcut

filepath = "./output1.mp4"
video_path = os.path.abspath(filepath)
print(video_path)

project_name = 'myDLC_modelZoo'
your_name = 'teamDLC'
model2use = 'full_human'
videotype = os.path.splitext(video_path)[-1].lstrip('.') #or MOV, or avi, whatever you uploaded!
video_path = deeplabcut.DownSampleVideo(video_path, width=300)

config_path, train_config_path = deeplabcut.create_pretrained_project(
    project_name,
    your_name,
    [video_path],
    videotype=videotype,
    model=model2use,
    analyzevideo=True,
    createlabeledvideo=True,
    copy_videos=True, #must leave copy_videos=True
)