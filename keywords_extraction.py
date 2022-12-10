from keybert import KeyBERT
doc = """
      Every smile you fake, every claim you stake, I'll be watching you
      """
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(doc)
print(keywords)