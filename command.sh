# Download
scp -J hln35@gate.eng.cam.ac.uk -r hln35@air211:/scratches/dialfs/alta/hln35/tung-workspace/log/version_10-03-2024--15-54-49/checkpoints \
    /Users/tung/fun/football-vast/log/version_10-03-2024--15-54-49/checkpoints

# Forward port
ssh -L 6006:localhost:6006 hln35@gate.eng.cam.ac.uk ssh -L 6006:localhost:6006 -N hln35@air211