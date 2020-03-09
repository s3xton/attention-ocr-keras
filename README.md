# Attention OCR Keras

Keras reimplementation of https://github.com/tensorflow/models/tree/master/research/attention_ocr/python.

Work in progress.

## Downloading data
The text file listing the FSNS dataset files is taken from [here](https://github.com/tensorflow/models/tree/master/research/street/python). If you don't want to download the entire dataset to try out the model, remove some on the filenames from the list.
```
aria2c -c -j 20 -i data/fsns_urls.txt
```

## Running the code
1. Download some training data and place it in the _data_ directory.
2. Build the container (from the root of the project):
   ```
   docker build -f Dockerfile -t trainer .
   ```
3. Run the container:
   ```
   docker run -v <absolute_path_to_project>/data:/data trainer
   ```