python .\rpg_e2vid\run_reconstruction.py -c .\rpg_e2vid\pretrained/E2VID_lightweight.pth.tar -i .\reconstruction\$($args[0]).feather --auto_hdr --display --show_events --output_folder .\reconstruction --dataset_name fixed_$($args[0]) --fixed_duration

mkdir .\reconstruction\fixed_$($args[0])\resampled

python .\rpg_e2vid\scripts\resample_reconstructions.py -i .\reconstruction\fixed_$($args[0]) -o .\reconstruction\fixed_$($args[0])\resampled -r 60

ffmpeg -framerate 60 -i .\reconstruction\fixed_$($args[0])\resampled\frame_%010d.png .\reconstruction\$($args[0])\60Hz_video.mp4