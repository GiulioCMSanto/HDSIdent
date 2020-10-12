.PHONY: environment
environment:
	pyenv install -s 3.7.2
	pyenv virtualenv 3.7.2 hdsident
	pyenv activate hdsident

.PHONY: requirements
requirements:
	pip install -r requirements.txt