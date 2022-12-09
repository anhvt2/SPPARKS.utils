#!/bin/bash
# https://askubuntu.com/questions/745732/converting-png-files-to-a-movie
# ffmpeg -framerate 1/5 -i perspective_small3d_0%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
ffmpeg -i perspective_small3d_0%03d.png -filter:v fps=5 5fps_perspective_potts_additive_small3d.mp4

