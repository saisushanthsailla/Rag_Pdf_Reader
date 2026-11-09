def format_docs(format_docs) :
  context_text = "\n\n".join(doc.page_content for doc in format_docs)
  return context_text