## Sentence fragments and punctuations chalenge

Due poor quality (specially on large-v3-turbo whisper.cpp) punctuation is missing everywhere.
Many segments are very short. Many only have one word. Most don't have any punctuation.
If present, most punctuations are ',' comma. Some segments start with upper letter and some with lower case.


### What is the best sofware for transcription?

Due problems mentioned above and looking for a best approach...
Analysing based on a low quality transcription of 36 minutes of audio on "10-10-2021 - Eclesiastes 3.2b-3a.txt".
Also found "O Primeiro Mandamento.txt" is very bad specially at the end. Many repetitive sentences. 
This initial version was run using large-v3-turbo and llama.cpp 
The simple analysis consists of counting number of '.', '?', '!' (sentence ends).
Too few sentences, or too big sentences, is a indicative of low quality.

1. Using model large-v3-turbo whispercpp    

    10-10-2021 - Eclesiastes 3.2b-3a.txt
    grep -Poz '(?<=\?|\.|!)(\n)'  10-10-2021\ -\ Eclesiastes\ 3.2b-3a.txt | wc -l 
    58 # end of phrases or sentences
    Very Bad quality an average of 1.61 phrases per minute of audio.

2. Using model large-v3 whisper.cpp (defaul f16?)
    
    10-10-2021 - Eclesiastes 3.2b-3a.txt_v2
    grep -Poz '(?<=\?|\.|!)(\n)'  10-10-2021\ -\ Eclesiastes\ 3.2b-3a.txt_v2 | wc -l 
    359 # end of phrases or sentences    
    Good average of 10 phrases per minute

3. Using model large-v3 whispercpp SYSTRAN/faster-whisper f16 with batch_size=4
    10-10-2021 - Eclesiastes 3.2b-3a.txt_v3

    It uses Silero VAD filter to identify the silences to make the splits for the batches.
    Using python text = open('10-10-2021 - Eclesiastes 3.2b-3a.txt_v3', 'r').read()
    *2 due two float point '.' on timestamp per line (\n')
    text.count('.')+text.count('!')+text.count('?')-text.count('\n')*2 
    348
    Good average of 10 phrases per minute a bit worse than not splitting.
    That's expected due the previous context being helpful to transcribe but not used with batch splits.
    Still the performance is superior, much faster. 
    Plus the vad filter removes parts withou voice like the music in the begging and in the end.
    That avoids halucinations, like multiple amens or fabricated names of transcriber.
    10 mins to run 36 minutes of audio - not fast not slow
    Unfortunatelly the merging points of segments have '...' (3 dots) still better than others but.... 
    tested also with `segments, info = batched_model.transcribe("10-10-2021 - Eclesiastes 3.2b-3a.wav", 
        beam_size=5, language='pt', batch_size=4, without_timestamps=True, vad_filter=True, chunk_length=25)`
    same problem.

4. Using large-v3 using whisperx fork of Nyralei  

    - before alignment
    10-10-2021 - Eclesiastes 3.2b-3a.txt_vx1
    text.count('.')+text.count('!')+text.count('?')-6
    294
    Code:
    ```
    batch_size = 4
    compute_type = 'float16'
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language='pt')
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    ```
    - after alignment
    10-10-2021 - Eclesiastes 3.2b-3a.txt_vx1a
    cleaned.count('.')+cleaned.count('!')+cleaned.count('?')-6
    294 
    '...' (3 dots) But just so that they are ok, they represent a break of thought and a pause moment
    Excelent. The best option of all. 
    Certainly that's what we will use despite a little lower on end of sentences. 
    Perplexiy.ai convinced me alignment will be always good.


Despite whisperX (Nyralei fork) having less 'sentences' we will stick with that.
The reason is that it performs many clean-ups and has many advanced features for high quality transcription, including also  forced alignment. Forced Alignment refers to the process by which orthographic transcriptions are aligned to audio recordings to automatically generate phone level segmentation.
That guarantees a better quality by somehow removing words not having a phonetic 'match'. We also get probability of word-sound match (maybe useful afterwards).This post here also votes in favor of whisperx https://amgadhasan.substack.com/p/sota-asr-tooling-long-form-transcription.


- Notes because whisperx is somewhat dead

    Rank of forks of WhisperX that's dying

    1. https://github.com/coqui-ai/whisperX   - company is dying but ranked with more stars on github
    2. https://github.com/federicotorrielli/BetterWhisperX --- 
    3. https://github.com/Nyralei/whisperX



### Yet another possible solution for sentence fragments (future exploration)

A Extra tree classifier to 'restore' punctuation.
Let's then mix this sentence transformer approach with the previous question to create a model that responds: should I merge or not (yes or no)? That model will be used to test whether a new fragment should be added or not. I like your idea of adding a word of the candidate fragment each time and calculate a new sentence embedding. I thought about using this to feed in the model (ExtraTreeClassifier Binary) as well but maybe that's too much. 


### Low quality audio

Some sermons are so bad tha we can't understand what's been said. Trucks and other sounds make it almost useless audio.
Maybe use the probabilty score of words from forced alignment to calculate so kind of quality score to warn the users... like : 'this is very bad quality of audio'. For example: '13-11-2020 - Igreja de Cristo no Antigo Testamento (1)'



