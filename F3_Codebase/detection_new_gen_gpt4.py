import os
import pandas as pd
import json

from dotenv import load_dotenv
load_dotenv('../.env')

pd.set_option('display.max_columns', 100)

models = [
    'gpt3_5',
    'gpt4o',
    'llama2',
]
prompt_types = [
    'Critical',
    'Major',
    'Minor',
]
dfs = []
for model in models:
    for prompt_type in prompt_types:
        try:
            df_real = pd.read_csv(
                f'X_GenPost_{model}_Real_Post_Completed_Data/{prompt_type}_results.csv'
            )
            df_fake = pd.read_csv(
                f'X_GenPost_{model}_Fake_Post_Completed_Data/{prompt_type}_results.csv'
            )

            df_real.loc[:,'Prompt_type'] = prompt_type
            df_fake.loc[:,'Prompt_type'] = prompt_type
            
            dfs.append(df_real)
            dfs.append(df_fake)
        except FileNotFoundError:
            print(f'model: {model}; prompt_type {prompt_type} not found')
df = pd.concat(dfs)
print(df.shape)
df[:3]

# Create version with just the pre-scoring columns
cols_shared = [
    'uuid', 'label', 'article_type', 'source_type', 'pre_post_GPT',
    'dataset_source', 'Prompt_type',
]
df_gen_wide = df.rename(columns={
    'ai_generated_label': 'label',
})[cols_shared + [
    'human_written_content', 'aigenerated_content',  # -> content
    'num_original_token', 'num_completion_token',  # -> text_length
]]

# Human text
df_gen_human = df_gen_wide.rename(columns={
    'human_written_content': 'content',
    'num_original_token': 'text_length',
})[cols_shared + ['content', 'text_length']]
df_gen_human.loc[:,'source_type'] = 'human'

# LLM/AI text
df_gen_llm = df_gen_wide.rename(columns={
    'aigenerated_content': 'content',
    'num_completion_token': 'text_length',
})[cols_shared + ['content', 'text_length']]
df_gen_llm.loc[:,'source_type'] = 'LLM'

df_gen = pd.concat([df_gen_human, df_gen_llm])
df_gen.info()

# Create subsetted version for replication

# Create version with just the pre-scoring columns
cols_shared = [
    'uuid', 'label', 'article_type', 'source_type', 'pre_post_GPT',
    'dataset_source', 'Prompt_type',
]
df_gen_wide = df.rename(columns={
    'ai_generated_label': 'label',
})[cols_shared + [
    'human_written_content', 'aigenerated_content',  # -> content
    'num_original_token', 'num_completion_token',  # -> text_length
]]

df_gen_wide_ss = df_gen_wide.sample(frac=.1, random_state=0)

# Human text
df_gen_human_ss = df_gen_wide_ss.rename(columns={
    'human_written_content': 'content',
    'num_original_token': 'text_length',
})[cols_shared + ['content', 'text_length']]
df_gen_human_ss.loc[:,'source_type'] = 'human'

# LLM/AI text
df_gen_llm_ss = df_gen_wide_ss.rename(columns={
    'aigenerated_content': 'content',
    'num_completion_token': 'text_length',
})[cols_shared + ['content', 'text_length']]
df_gen_llm_ss.loc[:,'source_type'] = 'LLM'

df_gen_ss = pd.concat([df_gen_human_ss, df_gen_llm_ss])
df_gen_ss.info()

df_gen['source_type'].value_counts()

df_gen_ss['source_type'].value_counts()

"""Generate a prompt using template.

  Args:
    F3 prompt template (θ) has three parameters: (1) content (C) embeds data to be analyzed. The content parameter in the the function [ai_text_generator] (2) Impersonator (R)
    establishes context, guides LLMs’ generation and detection, and overrides alignment-tuning. (3) Instructor (I)
    provides directives to guide LLM.

  Returns:
    A generated prompt.
"""
def get_prompt(prompt_type):

  impersonator = ""
  instructor = ""
  cloze_prompt = "Is this real or fake news? {_____}"

  if prompt_type == "VaN":# Technique
    impersonator = "You are an AI assistant trained to detect fake news."
    instructor = "Analyze the given text and determine if it is real or fake news."

  elif prompt_type == "A_CoT":# Technique  # Z_CoT
    impersonator = "You are an AI assistant trained to detect fake news."
    instructor = "Analyze the given text, think step-by-step, and determine if it is real or fake news."

  elif prompt_type == "X_CoT":# Technique
    impersonator = "You are an AI assistant trained to detect fake news."
    instructor = "Analyze the given text, explain your reasoning step-by-step, and determine if it is real or fake news."

  elif prompt_type == "A_Con":# Technique
    impersonator = "You are an AI assistant trained to detect fake news with confidence estimates."
    instructor = "Analyze the given text, provide a confidence score between 0-100%, and determine if it is real or fake news."

  elif prompt_type == "MSReN":# Technique
    impersonator = "You are an AI fact checker trained to detect fake news."
    instructor = "Analyze the text in detail as a fact checker would. Check for misleading info, false claims, biased language. If real, respond 'True', if fake, respond 'False'."
  
  # From paper
  elif prompt_type == "MSReN_paper":# Technique
    impersonator = "You are an AI fact checker trained to detect fake news."
    instructor = "Analyze the text in detail as a fact checker would solve it by splitting your reasoning into steps. Check for misleading info, false claims, biased language. If real, respond ’True’, if fake, respond ’False’."
  
  elif prompt_type == "MSReN_CoT":# Technique
    impersonator = "You are an AI fact checker trained to detect fake news."
    instructor = "Analyze the text in detail as a fact checker would. Explain your reasoning step-by-step. If real, respond 'True', if fake, respond 'False'."

  elif prompt_type == "DeF_Gen":# Technique
    impersonator = "You are an AI assistant trained to detect fake news."
    instructor = "Determine if the text contains factual information supported by evidence (real) or misleading/inaccurate information (fake). Respond with 'True' or 'False'."

  elif prompt_type == "DeF_SpeC":# Technique
    impersonator = "You are an AI assistant trained to detect fake news."
    instructor = "Analyze the tone, language, sources to determine if the text is real (supported by facts) or fake (misleading info). Respond with 'True' or 'False'."

  elif prompt_type == "Analyze_Cld2":# Technique
    impersonator = "You are an AI assistant specially trained to detect fake news with high accuracy."
    instructor = """Analyze the given news article in depth. Check for the following indicators that it may be fake news:
    - Inaccurate or misleading information
    - Lack of evidence for claims
    - Emotionally charged language with intent to provoke outrage or shock
    - Biased portrayal of events or people
    - Unverified sources or 'experts'
    - Logical fallacies or conspiracy theories without proof

    If the article exhibits multiple indicators of fake news, respond 'False'. If the article is factual and supported by credible evidence, respond 'True'. Your judgement should be highly accurate.
    """

  elif prompt_type == "Analyze_AI_GPT":# Technique
    impersonator = "You are an AI trained with extensive knowledge up to 2022 on various news articles, both real and fake."
    instructor = """Analyze the given text for potential indicators of fake news, such as:
    - Sensationalist or emotionally charged language.
    - Absence of specific details or dates.
    - Over-generalizations or sweeping statements.
    - Statements that are too good to be true or overly dramatic.
    - Lack of logical flow in arguments or jumping to conclusions without evidence.

    It's essential to understand that without real-time verification capabilities, your judgment will be based on patterns and knowledge up to your last training. Using these textual cues and your training, determine the credibility of the given text. If it seems factual and consistent with your training, respond 'True'. If it exhibits patterns typical of fake news, respond 'False'."""

  else:
    raise ValueError('Unexpected prompt_type:', prompt_type)
    
  prompt = f"{impersonator} {instructor} {cloze_prompt}"

  return prompt

# %% [markdown]
# ### GPT3.5 Turbo

# %%
from time import sleep
import os
import pandas as pd
import openai
from tqdm import tqdm
import concurrent.futures

# %%
OPENAI_TOKEN = os.getenv('OPENAI_KEY')

# %%
import multiprocessing

cores = multiprocessing.cpu_count() # Count the number of cores in a computer
cores

# %%
import os
import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import concurrent.futures

client = OpenAI(api_key=OPENAI_TOKEN)

# Function for AI text classification
def ai_text_classification(prompt, content):
    # openai.api_key = OPENAI_TOKEN

    while True:
        try:
            time.sleep(11)  # Sleep for 600 milliseconds
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content[:4097]},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            time.sleep(30)  # Wait for 60 seconds before retrying

# Load the prompt pattern
prompt_types = [
    "VaN", "A_CoT", "X_CoT", "A_Con",
    "MSReN", "MSReN_paper", "MSReN_CoT",
    "DeF_Gen", "DeF_SpeC",
    "Analyze_Cld2", "Analyze_AI_GPT"
]

# No seeds in GPT, but we can do multiple classifications
seeds = [0]

for seed in seeds:
    # Ensure backup directory exists
    backup_folder = f'bkp_new_gen_gpt4o_seed_{seed}'
    os.makedirs(backup_folder, exist_ok=True)

    # Load previous progress if exists
    backup_path = os.path.join(backup_folder, "progress.csv")
    last_processed_uuid = None
    last_processed_prompt = None

    try:
        progress_df = pd.read_csv(backup_path)
        last_processed_uuid = progress_df.iloc[-1]['uuid']
        last_processed_prompt = progress_df.iloc[-1]['Prompt_type']
    except FileNotFoundError:
        progress_df = pd.DataFrame(columns=[
            'uuid', 'GPT3_5T-label', 'content', 'label',
            'Prompt_type', 'pre_post_GPT', 'article_type', 'dataset_source', 'source_type',
        ])

    # Create a list of all combinations of rows and prompt types
    if last_processed_uuid and last_processed_prompt:
        start_idx = df_gen_ss[df_gen_ss['uuid'] == last_processed_uuid].index[0]
        start_prompt_idx = prompt_types.index(last_processed_prompt) + 1
        if start_prompt_idx >= len(prompt_types):
            start_idx += 1
            start_prompt_idx = 0

        all_data = [
            (row, prompt)
            for _, row in df_gen_ss.iloc[start_idx:].iterrows()
            for prompt in (prompt_types[start_prompt_idx:]
            if row['uuid'] == last_processed_uuid else prompt_types)
        ]
    else:
        all_data = [
            (row, prompt)
            for _, row in df_gen_ss.iterrows()
            for prompt in prompt_types
        ]

    # Function to process each row for a given prompt type
    def process_row_for_prompt(data):
        row, prompt_type = data
        content = row['content']
        cloze_prompt = get_prompt(prompt_type)  # assuming this function exists in your full code
        prompt_content = f"Content: '{content}' prompt: '{cloze_prompt}. Please strictly state a single probable answer if you are uncertain, lack evidence or unverified, but please strictly answer with your single word: 'True' or 'False'.'"

        try:
            model_output = ai_text_classification(prompt_content, content)
            return (
                row['uuid'], model_output, content, row['label'], prompt_type,
                row['pre_post_GPT'], row['article_type'], row['dataset_source'],
                row['source_type']
            )
        except Exception as e:
            print(f"Error for uuid {row['uuid']} with prompt {prompt_type}: {str(e)}")
            return (
                row['uuid'], "Error", content, row['label'], prompt_type,
                row['pre_post_GPT'], row['article_type'], row['dataset_source'],
                row['source_type']
            )

    # Parallel execution using ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for data in tqdm(executor.map(process_row_for_prompt, all_data), total=len(all_data), desc=f'Seed {seed}'):
            results.append(data)

            # Adjust the save condition (you might want to modify this based on the frequency of saving)
            if len(results) % 100 == 0 or len(results) == len(all_data):
                print("Saving progress.")
                columns = [
                    'uuid', 'GPT3_5T-label', 'content', 'label',
                    'Prompt_type', 'pre_post_GPT', 'article_type', 'dataset_source', 'source_type'
                ]
                temp_df = pd.DataFrame(results, columns=columns)
                progress_df = pd.concat([progress_df, temp_df], axis=0, ignore_index=True)
                progress_df.to_csv(backup_path, mode='w', header=True, index=False)
                results = []
