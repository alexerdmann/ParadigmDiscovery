# ParadigmDiscovery

Demo:

```
lg=eng_V; python Scripts/ANA.py -C Data/Final/Intrinsic/Corpus/corp.$lg -L Data/Final/Intrinsic/Lexicon/lex.$lg -m Models -l $lg -c oracle -U Data/Final/UniMorph_intersect/inter_lex.$lg -e Data/Final/Extrinsic/analogy.$lg
```

To Do:

* Clean EM code from Paradigm Cell Discovery repository
* Incorporate clean EM code with multi-task architecture predicting inflection in context
