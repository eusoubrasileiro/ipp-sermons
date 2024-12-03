### Igreja Presbiteriana Peregrinos Sermons AI Experiment 

Sermons transcribed by ~~whisper.cpp~~ [whisperx](https://github.com/m-bain/whisperX) ~~and to be analyzed by GPT-4o mini~~. 

Preprocessing of mp3 audio includes: `loudnorm` normalizes loudness to keep the audio level consistent. `afftdn=nf=-25` reduces noise with a frequency threshold for background noise control. `highpass=f=150` applies a high-pass filter, cutting out frequencies below 150 Hz to remove low rumbles and make speech clearer. Additionally, the audio sample rate is set to 16kHz (-ar 16000) to match the optimal frequency for speech, the audio is converted to mono (-ac 1), and the bitrate is limited to 64 kbps (-b:a 64k), which is sufficient for speech without compromising clarity. 

~~whisper.cpp using ggml-large-v3-turbo model.~~  
whisperx using large-v3-model and forced alignment split on 4 batches 

Data being analyzed at the moment

https://ipp-sermons.talvez.site/
