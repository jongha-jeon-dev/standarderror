"""The small language model the later episodes measure.

Series that make claims about transformers need a transformer, and a system I
wrote and control wearing a model's name is not one. `tiny` holds a
character-level model of 816,128 parameters, trained on `tinyshakespeare` to a
validation loss of 1.573 against a uniform-guess 4.174, together with the
script that produced it and the checkpoint it produced.

The checkpoint is committed. The first version of this lived in a scratch
directory and was lost to a container reset, which cost the two episodes that
depended on it; 3.2 MB in the repository is the cheaper of the two mistakes.
"""

from standarderror.llm import tiny

__all__ = ["tiny"]
