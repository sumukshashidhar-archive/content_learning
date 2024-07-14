# content_learning
Parameter Efficient Content Learning For LLMs

## Introduction

The goal of this codebase is to facilitate the continued pre-training, and then subsequent chat finetuning of open source LLMs on brand new content. 

## Preprocessing

You must first convert your content (PDF, EPUB, etc.) into a text file. This can be done using the scripts in the `src/data_preprocessing` directory. Read the [Data Preprocessing Documentation](docs/preprocessing.md) for more information. 

## Sanitization and Chunking

Once you have a directory of text files, you must sanitize and tokenize them for pretraining, using the script in the `src/sanitization` directory. Read the [Sanitization Documentation](docs/sanitization.md) for more information. Note that this automatically assumes that you have existing chat data, to intersperse with your new content, to ensure that catostrophic forgetting of chat template data, does not occur.