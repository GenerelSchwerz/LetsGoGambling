#!/usr/bin/bash
dnum=$(xdotool get_num_desktops)
desk=0
while [[ "$desk" -lt "$dnum" ]];
     do
          xdotool set_desktop $desk;
          sleep 2;
          maim --format png /dev/stdout > ~/capture/desktop$desk.png;
          desk=$((desk+1));
     done