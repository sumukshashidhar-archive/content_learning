# content_learning
Parameter Efficient Content Learning For LLMs

## Introduction

The goal of this codebase is to facilitate the continued pre-training, and then subsequent chat finetuning of open source LLMs on brand new content. 

## Preprocessing

You must first convert your content (PDF, EPUB, etc.) into a text file. This can be done using the scripts in the `src/data_preprocessing` directory. Read the [Data Preprocessing Documentation](docs/preprocessing.md) for more information.