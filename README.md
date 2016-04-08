**************************************************************************************************
**THIS IS A DEVELOPMENTAL REPOSITORY FOR M2Scorer**   
**FOR AN OFFICIAL VERSION (VERSION 3.2), visit http://www.comp.nus.edu.sg/~nlp/conll14st.html**  
**OR CHECK OUT THE RELEASES: https://github.com/nusnlp/m2scorer/releases**  
****************************************************************************************************

## M^2Scorer 

This is the scorer for evaluation of grammatical error correction systems. 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License (See [LICENSE](license.md)).

If you are using the NUS M^2 scorer in your work, please include a
citation of the following paper:

Daniel Dahlmeier and Hwee Tou Ng. 2012. Better Evaluation for
Grammatical Error Correction. In Proceedings of the 2012 Conference of
the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies (NAACL 2012).



### Contents  
1. Quickstart
2. Pre-requisites 
3. Using the scorer   
  3.1 System output format     
  3.2 Scorer's gold standard format   
4. Converting the CoNLL-2014 data format
5. Revisions  
   5.1 Alternative edits     
   5.2 F-beta measure   
   5.3 Handling of insertion edits   
   5.4 Bug fix for scoring against multiple sets of gold edits, and dealing with sequences of insertion/deletion edits


### Quickstart

```
./m2scorer [-v] SYSTEM SOURCE_GOLD 
```
SYSTEM = the system output in sentence-per-line plain text.
SOURCE_GOLD = the source sentences with gold edits.


### Pre-requisites
The following dependencies have to be installed to use the M^2 scorer.

* Python (>= 2.6.4, < 3.0, older versions might work but are not tested)
* nltk (http://www.nltk.org, needed for sentence splitting) 


### Using the scorer
```
Usage: m2scorer [OPTIONS] SYSTEM SOURCE_GOLD
```
where   
 SYSTEM          -   system output, one sentence per line   
 SOURCE_GOLD     -   source sentences with gold token edits   
```
OPTIONS
  -v    --verbose             -  print verbose output
  --very_verbose              -  print lots of verbose output
  --max_unchanged_words N     -  Maximum unchanged words when extracting edits. Default = 2.
  --ignore_whitespace_casing  -  Ignore edits that only affect whitespace and casing. Default no.
  --beta                      -  Set the ratio of recall importance against precision. Default = 0.5.

```
#### 2.1 System output format
The sentences should be in tokenized plain text, sentence-per-line
format.

Format:
```
<tokenized system output for sentence 1>
<tokenized system output for sentence 2>
 ...
```
**Examples of tokenization:**  
 Original  : He said, "We shouldn't go to the place. It'll kill one of us."   
 Tokenized : He said , " We should n't go to the place . It 'll kill one of us . "   

**Note:** Tokenization in the CoNLL-2014 shared task uses NLTK word tokenizer.  

**Sample output:**   
===> system <===
A cat sat on the mat .
The Dog .


#### Scorer's gold standard format
SOURCE_GOLD = source sentences (i.e. input to the error correction
system) and the gold annotation in TOKEN offsets (starting from zero). 

**Format:**
```
S <tokenized system output for sentence 1>
A <token start offset> <token end offset>|||<error type>|||<correction1>||<correction2||..||correctionN|||<required>|||<comment>|||<annotator id>
A <token start offset> <token end offset>|||<error type>|||<correction1>||<correction2||..||correctionN|||<required>|||<comment>|||<annotator id>

S <tokenized system output for sentence 2>
A <token start offset> <token end offset>|||<error type>|||<correction1>||<correction2||..||correctionN|||<required>|||<comment>|||<annotator id>
```

**Notes:**   
 * Each source sentence should appear on a single line starting with "S "
 * Each source sentence is followed by zero or more annotations.
 * Each annotation is on a separate line starting with "A ".
 * Sentences are separated by one or more empty lines.
 * The source sentences need to be tokenized in the same way as the system output.
 * Start and end offset for annotations are in token offsets (starting from zero).
 * The gold edits can include one or more possible correction strings. Multiple corrections should be separate by '||'.
 * The error type, required field, and comment are not used for scoring at the moment. You can put dummy values there.
 * The annotator ID is used to identify a distinct annotation set by which system edits will be evaluated.
   * Each distinct annotation set, identified by an annotator ID, is an alternative
   * If one sentence has multiple annotator IDs, score will be computed for each annotator.
   * If one of the multiple annotation alternatives is no edit at all, an edit with type 'noop' or with offsets '-1 -1' must be specified.
   * The final score for the sentence will use the set of edits by an annotation set maximizing the score.


**Example:**   

The gold annotation file can be found here: [example/source_gold](example/source_gold)
```
S The cat sat at mat .
A 3 4|||Prep|||on|||REQUIRED|||-NONE-|||0
A 4 4|||ArtOrDet|||the||a|||REQUIRED|||-NONE-|||0

S The dog .
A 1 2|||NN|||dogs|||REQUIRED|||-NONE-|||0
A -1 -1|||noop|||-NONE-|||-NONE-|||-NONE-|||1

S Giant otters is an apex predator .
A 2 3|||SVA|||are|||REQUIRED|||-NONE-|||0
A 3 4|||ArtOrDet|||-NONE-|||REQUIRED|||-NONE-|||0
A 5 6|||NN|||predators|||REQUIRED|||-NONE-|||0
A 1 2|||NN|||otter|||REQUIRED|||-NONE-|||1
```
Let the system output, [example/system](example/system) be
```
A cat sat on the mat .
The dog .
Giant otters are apex predator .
```
Run the M^2Scorer as follows:
```
./m2scorer example/system example/source_gold 
```
The evaluation output will be will be:
```
Precision   : 0.8000
Recall      : 0.8000
F_0.5       : 0.8000
````
**Explanation:**
For the first sentence, the system makes two valid edits {(at-> on),
(\epsilon -> the)} and one invalid edit (The -> A).

For the second sentence, despite missing one gold edit (dog -> dogs) according
to annotation set 0, the system misses nothing according to set 1.

For sentence #3, according to annotation set 0, the system makes two
valid edits {(is -> are), (an -> \epsilon)} and misses one edit
(predator -> predators); however according to set 1, the system makes
two unnecessary edits {(is -> are), (an -> \epsilon)} and misses one
edit (otters -> otter).

By the case above, there are four valid edits, one unnecessary edit,
and one missing edit. Therefore precision is 4/5 = 0.8. Similarly for
recall. In the above example, the beta value for the F-measure is 0.5
(the default value).


###Converting the CoNLL-2014 data format
The data format used in the M^2 scorer differs from the format used in
the CoNLL-2014 shared task (http://www.comp.nus.edu.sg/~nlp/conll14st.html)
in two aspects:
 - sentence-level edits
 - token edit offsets

To convert source files and gold edits from the CoNLL-2014 format into
the M^2 format, run the preprocessing script bundled with the CoNLL-2014
training data.

