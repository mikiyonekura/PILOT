# PILOT

PILOT is a technical debt detector built on top of a combination of different natural language processing (NLP) and machine learning (ML) techniques.

## Generate the feature matrices
- Run the Python script features-matrices-dataset-debthunter.py to generate the features-matrices related to debthunter dataset
- Run the Python script features-matrices-dataset-maldonato.py to generate the features-matrices related to Maldonado dataset

## Run the classifier engines

import Runner as runner
runner.run('DatasetD3/Round1/', [4003, 10, 2], 100, 3.0, 0.75)

runner.displayCM('DatasetD2/',5)
runner.displayCM('DatasetD3/',2)

runner.runExperiment('DatasetD3/', [4003, 10, 2], 100, 3.0, 0.75)
runner.displayROC('DatasetD3/',2)


runner.runExperiment('DatasetD1/', [5072, 10, 5], 100, 3.0, 0.75)
runner.run('DatasetD1/Round1/', [5072, 10, 5], 100, 3.0, 0.75)
runner.displayCM('DatasetD1/',5)
runner.displayROC('DatasetD1/',5)


