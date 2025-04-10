"""This is a cookiecutter example - feel free to delete this module.

Documentation up here is meant for the module."""

from typing import Optional


class Foo:
  """This is class-level documentation.

  You can document `__init__` up here or within the method itself (not both
  though).

  Spotify has historically followed the Google style for documentation.
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/
  example_google.html

  Here's an example of Google documentation style:

  Attributes:
      attr1 (str): Description of `attr1` goes here.
      attr2 (:obj:`int`, optional): Description of `attr2` goes here.
  """

  def __init__(self, param1: str, param2: Optional[int]) -> None:
    """You could also document `attr1` and `attr2` from above here.

    Args:
        param1: Description of `param1`. Types are not needed as they are
            specified with the type annotations.
        param2: Description of `param2`.
                This can be multiple lines.
    """
    self.attr1 = param1
    self.attr2 = param2

  def return_state(self) -> str:
    """A simple method used for example tests."""

    return f"Current state is: attr1: {self.attr1} attr2: {self.attr2}"
