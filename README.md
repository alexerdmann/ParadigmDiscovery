# ParadigmDiscovery

## Overview

This is the repository for the paper "The Paradigm Discovery Problem" from Alexander Erdmann, Micha Elsner, Shijie Wu, Ryan Cotterell, and Nizar Habash, which is forthcoming at ACL 2020. We are still in the process of releasing all the data and cleaning things up. Please bear with us.

## Demo

```
lg=eng_V; python Scripts/ANA.py -C Data/Final/Intrinsic/Corpus/corp.$lg -L Data/Final/Intrinsic/Lexicon/lex.$lg -m Models -l $lg -c oracle -U Data/Final/UniMorph_intersect/inter_lex.$lg -e Data/Final/Extrinsic/analogy.$lg
```

