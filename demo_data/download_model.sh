#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-z0cfttgnd_GirdnPgVkmpms69MtWOzp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-z0cfttgnd_GirdnPgVkmpms69MtWOzp" -O PartialSC_VC.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1w4l0r33dTzy3KGoF5BfPhQLFiRFmm4iR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1w4l0r33dTzy3KGoF5BfPhQLFiRFmm4iR" -O PartialSC_CN.pth && rm -rf /tmp/cookies.txt
