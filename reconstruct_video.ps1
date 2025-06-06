python .\rpg_e2vid\run_reconstruction.py -N 100000 -c .\rpg_e2vid\pretrained/E2VID_lightweight.pth.tar -i .\reconstruction\$($args[0]).feather --auto_hdr --display --show_events --output_folder .\reconstruction --dataset_name $($args[0])

mkdir .\reconstruction\$($args[0])\resampled

python .\rpg_e2vid\scripts\resample_reconstructions.py -i .\reconstruction\$($args[0]) -o .\reconstruction\$($args[0])\resampled -r 300

ffmpeg -framerate 10 -i .\reconstruction\$($args[0])\resampled\frame_%010d.png .\reconstruction\$($args[0])\240Hz_video.mp4