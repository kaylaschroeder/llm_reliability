# Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge
### Introduction
The repository contains supporting repository for the paper 'Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge' which can be accessed [here](TO ADD: ARXIV LINK).

### Within the Repo: Experiments

### Within the Repo: Application Experiments 

The `application_data` folder contains all the necessary components to replicate the application of our framework (as discussed in the paper) to the academics domain of the Head-to-Tail benchmark from [Sun et al. (2024)](https://aclanthology.org/2024.naacl-long.18.pdf). 

**Questions**: The files `head_to_tail_dblp.json` and `head_to_tail_mag.json` contain the academics domain Head-to-Tail questions. To obtain these questions, we followed the process outlined by the GitHub supporting Sun et al. (2024). Please see their GitHub for further detail.

**Data**: The script `application_responses.py` can be run to obtain LLM responses to the head questions from the Head-to-Tail benchmark from the academics domain. To run this script, define the model types of interest using their Hugging Face names within the `model_types` parameter, update `access_token` to include your Hugging Face access token to the models, and update `cuda` to your respective device. The results of this script that were utilized within the paper are contained within `application_responses.json`.

**Judgments**: Use the script `application_judges.py` to obtain LLM judgments of the Head-to-Tail responses. Running this script again requires specification judgment models of interest using their Hugging Face names within the `model_types` parameter, updating `access_token` to include your Hugging Face access token to the models, and updating `cuda` to your respective device. Final judgment results from all judgment models are contained within the `judgment_results` folder.

**Reliability**: Calculate the reliability of your judgment models for Head-to-Tail judgments using `application_reliab.py`.
