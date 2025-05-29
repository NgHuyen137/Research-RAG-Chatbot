import contractions


def clean_text(text: str) -> str:
  """
    Lowercase & Normalize text 
  """
  return contractions.fix(text).lower()
