# Frozen reference

model.py and separation.py are unchanged upstream model files from source commit
6fdad13. demix.py contains verbatim upstream demix/helper function bodies with
only their imports isolated from the training framework. provenance.json records
original source hashes and the extracted function names. These files are used
by the tests and real-audio comparison, not by the installed stemmer package.

The tools/ directory preserves the upstream auxiliary scripts that were ignored
by the original repository's case-insensitive Scripts/ ignore rule on this Mac.
They are historical source for reference, not supported entry points of this
lean package. The supported real-audio entry point is ../compare_audio.py.
