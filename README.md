# Unsupervised Paradigm Discovery

## Overview

This is the repository for the ACL 2020 paper "The Paradigm Discovery Problem" from Alexander Erdmann, Micha Elsner, Shijie Wu, Ryan Cotterell, and Nizar Habash. This code release represents the published system, though better results have been achieved since the paper was accepted. If you're interested in the improved code, please reach out to me (ae1541@nyu.edu).

## Prerequisites

This repository was tested using TensorFlow 1.14. I *tried* to make the code forwards compatible with TensorFlow 2.0. In theory, you *should* only have to comment out line xxx in *Scripts/xxx.py*, though I have not tested this.

## Usage

After unzipping the Data directory, first select the language and part of speech you want to run.

```
lgPOS=ara_N  # the other supported language-POS's are deu_N, eng_V, lat_N or rus_N
```

Then run the system with the following command: 

```
python Scripts/ANA.py -C Data/Corpora/corp.$lgPOS -L Data/Lexica/lex.$lgPOS -m MyModel -l $lgPOS -U Data/LexiconUDintersects/inter_lex.$lgPOS -e Data/Analogies/an.$lgPOS
```

