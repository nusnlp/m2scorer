# MaxMatch (M2) scorer

### Quickstart

```
./m2scorer [-v] SYSTEM SOURCE_GOLD 
```
SYSTEM = the system output in sentence-per-line plain text.
SOURCE_GOLD = the source sentences with gold edits.


### Pre-requisites
The following dependencies have to be installed to use the M^2 scorer.

* Python (>= 2.6.4, <= 3.7, older and newer versions might work but are not tested)


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
  --max-unchanged-words N     -  Maximum unchanged words when extracting edits. Default = 2.
  --ignore-whitespace-casing  -  Ignore edits that only affect whitespace and casing. Default no.
  --beta  BETA                -  Set the ratio of recall importance against precision. Default = 0.5.
  --timeout TIMEOUT           -  Max wait time for row in M2 file
```