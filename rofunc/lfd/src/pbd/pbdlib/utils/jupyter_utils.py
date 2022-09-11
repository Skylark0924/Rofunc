from IPython.display import display, Markdown, Latex

def MK(text):
	"""
	Display markdown in Jupyter notebook
	:param text:
	:return:
	"""
	display(Markdown(text))

def LT(text):
	"""
	Display latex in Jupyter notebook
	:param text:
	:return:
	"""
	display(Latex(text))
