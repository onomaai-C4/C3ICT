# C4: Creating the Next Story From the Current Story And the Retrieved Chapter-Intent-Chapter Triplet

This repo contains code for C4: Creating the Next Story From the Current Story And the Retrieved Chapter-Intent-Chapter Triplet.

[PDF 파일 보기](./README_data/1_Main%20Module.pdf)

### Preparation

First, you will need an OpenAPI key. Enter your key in the format 'sk-xxxx...' in the .env_sample file.
```
OPENAI_API_KEY='sk-xxxx...'
```
Then, rename the .env_sample file to .env.

## Simple start

If you want to test our pipeline through the Gradio web application, use the link that opens from
```
main_gradio.py
```
If you want to directly check all the output processes in the terminal window, use
```
main_full_pipeline.py
```