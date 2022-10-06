#This simple script runs the motion magnification pipeline end to end
#Usage example: .\magnify_video.ps1 -mag 20 -i .\..\demo_vid_IQS\14_25Hz_400mV_2.mp4 -s .\..\demo_vid_IQS -m ckpt\ckpt_e35.pth.tar -o 14_25Hz_400mV_2 -c

$ErrorActionPreference = "Stop"

function print_usage() {
    echo "Usage: 
  
  The following arguments must be provided:
  -mag (magnification factor): Video magnification factor (default 25)
  -i (input file): Path pointing to target video (required)
  -s (save dir): Path to a directory to store result files (required)
  -m (model chekpoint): Path to the last model checkpoint (required)
  -o (output): Output project name (required)
  -mod (mode): static(default)/dynamic (params default)
  -f (framerate): Framerate of the input video (default 60)
  -c (cuda): Activates cuda (default cpu)

  "
}

$mag_factor = '25'
$cuda_flag = 'cpu'
$framerate = '60'
#$in = ''
$mode = 'static'
#$model_ckpt = ''
#$output = ''
#$save_dir = ''

for ( $i = 0; $i -lt $args.count; $i=$i+2 ) {
    if ($args[ $i ] -eq "\mag"){ $mag_factor = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-mag"){ $mag_factor = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\c"){ $cuda_flag = "cuda" }
    elseif ($args[ $i ] -eq "-c"){ $cuda_flag = "cuda" }
    elseif ($args[ $i ] -eq "\f"){ $framerate = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-f"){ $framerate = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\i"){ $in = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-i"){ $in = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\mod"){ $mode = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-mod"){ $mode = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\m"){ $model_ckpt = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-m"){ $model_ckpt = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\o"){ $output = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-o"){ $output = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "\s"){ $save_dir = $args[ $i+1 ]}
    elseif ($args[ $i ] -eq "-s"){ $save_dir = $args[ $i+1 ]}
    else { 
        print_usage
        exit -1
    }
}

# Check if required arguments are defined
if ($in -eq $null -or $model_ckpt -eq $null -or $output -eq $null -or $save_dir -eq $null) {
    print_usage
    exit -1
}

# Convert video to frames
if (Test-Path "$save_dir\$(echo $output)_original") {
    rm "$save_dir\$(echo $output)_original" -r
}
if (-Not (Test-Path "$save_dir\$(echo $output)_original")){
    mkdir "$save_dir\$(echo $output)_original"
}

ffmpeg -i "$in" "$save_dir\$(echo $output)_original\frame_%06d.png"

# Reshape frames to be divisible by 64
$files = Get-ChildItem "$save_dir\$(echo $output)_original"
foreach ($f in $files){
    echo "padding $f"
    python3 .\utils\auto_pad.py -i "$save_dir\$(echo $output)_original\$f" -d 64 -o "$save_dir\$(echo $output)_original\$f"
}

# Count frames
$num_data = $files.Count

# Magnify frames
python3 run.py -j4 -b1 --load_ckpt "$model_ckpt" --save_dir "$save_dir\$(echo $output)_mag" --video_path "$save_dir\$(echo $output)_original\frame" --num_data "$num_data" --mode "$mode" --mag "$mag_factor" --device "$cuda_flag"

# Format magnified frames to mp4
ffmpeg -framerate "$framerate" -i "$save_dir\$(echo $output)_mag\STBVMM_$(echo $mode)_%06d.png" "$save_dir\$(echo $output)_x$(echo $mag_factor)_$(echo $mode)_output.mp4"

exit 0