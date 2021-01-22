# MEMORY LEAK WARNING
Please, be aware, that the CarRacing-v0 environment of OpenAI gym version 0.18.0 suffers from a severe memory leak bug
that may hinder any attempts of training/testing agents.

For details and a fix see the discussion in this PR: https://github.com/openai/gym/pull/2096

# Environment setup
To set up a minimal environment to execute the training and testing notebooks run

`conda create --name car_racing --file requirements.txt`

activate the environment

`source activate car_racing`

and add a corresponding jupyter kernel

`python -m ipykernel install --user --name car_racing --display-name "Python (car_racing)"`

(and be mindful of the memory leak bug mentioned above).
