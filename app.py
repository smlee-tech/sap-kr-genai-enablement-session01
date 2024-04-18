# from langchain.prompts import PromptTemplate

import os
import json
import gradio as gr

import hana_ml
from hana_ml import dataframe
from hana_ml.dataframe import ConnectionContext
from hana_ml.algorithms.pal.utility import DataSets, Settings
import pandas as pd

# print(hana_ml.__version__)

### Connect to HANA 
# with open(os.path.join(os.getcwd(), '/Users/i551982/Desktop/Github/demo-block-1/HANA-configure-chul.json')) as f:
with open(os.path.join(os.getcwd(), './HANA-configure-chul.json')) as f:
    hana_env_c = json.load(f)
    port_c  = hana_env_c['port']
    user_c  = hana_env_c['user']
    url_c  = hana_env_c['url']
    pwd_c  = hana_env_c['pwd']

cc = ConnectionContext(address=url_c, port=port_c, user= user_c, password = pwd_c, encrypt=True)
# cc = dbapi.connect(address=url_c, port=port_c, user= user_c, password = pwd_c, encrypt=True)

print(cc.hana_version())

cursor = cc.connection.cursor()
# cursor = cc.cursor()

cursor.execute("""SET SCHEMA GEN_AI""")

print(cc.get_current_schema())

print(cc.sql('''SELECT TOP 10 *  FROM DEMO_BLOCK_3''').collect().head(3))

from gen_ai_hub.proxy.native.openai import embeddings

def get_embedding(input, model="dc872f9eef04c31a") -> str:
    response = embeddings.create(
      deployment_id=model,
      input=input
    )
    return response.data[0].embedding

# Wrapping HANA vector search in a function
def run_vector_search_en(query: str, metric="COSINE_SIMILARITY", k=5):
    if metric == 'L2DISTANCE':
        sort = 'ASC'
        col = 'L2D_SIM'
    else:
        sort = 'DESC'
        col = 'COS_SIM'
    query_vector = get_embedding(query)

    sql = '''SELECT TOP {k} "BlockID","MaterialNumber", "MaterialName","MaterialDescription","PurchasingGroupDescription",
            "Blocked_stock","Block_reason","Blocked_date", "Solution", "Solution_date",
            "{metric}"("VECTOR_EN", TO_REAL_VECTOR('{qv}'))
            AS "{col}"
            FROM "DEMO_BLOCK_3"
            ORDER BY "{col}"'''.format(k=k, metric = metric, qv=query_vector, sort=sort, col = col)
            
    hdf = cc.sql(sql)
    df_context = hdf.collect()#hdf.head(k).collect()
    return df_context

# Wrapping HANA vector search in a function
def run_vector_search_kr(query: str, metric="COSINE_SIMILARITY", k=5):
    if metric == 'L2DISTANCE':
        sort = 'ASC'
        col = 'L2D_SIM'
    else:
        sort = 'DESC'
        col = 'COS_SIM'
    query_vector = get_embedding(query)
    sql = '''SELECT TOP {k} "BlockID","MaterialNumber", "MaterialName","MaterialDescription_KR","PurchasingGroupDescription_KR",
            "Blocked_stock","Block_reason_KR","Blocked_date", "Solution_KR", "Solution_date",
            "{metric}"("VECTOR_EN", TO_REAL_VECTOR('{qv}'))
            AS "{col}"
            FROM "DEMO_BLOCK_3"
            ORDER BY "{col}"'''.format(k=k, metric=metric, qv=query_vector, sort=sort, col = col)
    hdf = cc.sql(sql)
    df_context = hdf.collect() 
    return df_context

# Function to detect query language
def detect_language(query):
    # Simplified example, you would use a more sophisticated method
    if any(ord(char) > 128 for char in query):  # Basic check for non-ASCII characters
        return 'kr'
    else:
        return 'en'
    

from langchain import PromptTemplate
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
proxy_client = get_proxy_client('gen-ai-hub')

#  prompt template
promptTemplate_fstring = """
You are provided multiple context items that are related to the prompt you have to answer.
Use the following pieces of context to answer the question at the end. 
Context:
{context}

Question:
{query}

Answer: 
To answer questions, Please provides solutions with detail instructions with steps and ResponsibleDepartment. 
You must format your output as a plain text that adheres in english. 
This is example formate: 
1. Solution: 
- Responsible department: 
- Instructions:
1) , 2)  , 3), ...
2. Solution: 
- Responsible department: 
- Instructions:
1) , 2)  , 3), ...
...
"""
promptTemplate_fstring_KR = """
You are provided multiple context items that are related to the prompt you have to answer.
Use the following pieces of context to answer the question at the end. 
Context:
{context}

Question:
{query}

Answer: 
To answer questions, Please provides Solution_KR with detail instructions with steps and ResponsibleDepartment_KR for each step. 
You must format your output as a plain text adheres in korean. 
This is example format: 
1. 해결방안: 
- 담당부서: 
- 해결철자:
1) , 2)  , 3), ...
2. 해결방안: 
- 담당부서: 
- 해결철자:
1) , 2)  , 3), ...
...
"""


def ask_llm(query: str, retrieval_augmented_generation = True, metric='COSINE_SIMILARITY', k = 5) -> str:
    
    language = detect_language(query)
    assistant_description = "You are a helpful assistant in warehouse management." if language == 'en' else "당신은 창고 관리에 도움이 되는 담당자입니다."
    
    # Use the appropriate vector search function based on the language
    if language == 'en':
        df_context = run_vector_search_en(query, metric, k)

        # Prepare the prompt using the appropriate language
        prompt = PromptTemplate.from_template(promptTemplate_fstring).format(
            assistant_description=assistant_description,
            query=query,
            context=' '.join(df_context[context_columns_based_on_language(language)].astype('string'))
        )
    else:
        df_context = run_vector_search_kr(query, metric, k)
        # promptTemplate = PromptTemplate.from_template(promptTemplate_fstring)
        
        # Prepare the prompt using the appropriate language
        prompt = PromptTemplate.from_template(promptTemplate_fstring_KR).format(
            assistant_description=assistant_description,
            query=query,
            context=' '.join(df_context[context_columns_based_on_language(language)].astype('string'))
        )
    print(df_context.head())
    
    # Adjust temperature based on language or other criteria
    temperature = 0.2 #if language == 'kr' else 0.3
    
    llm = ChatOpenAI(deployment_id="d31f88c3e2e7eda5", temperature=temperature)
    
    # response = llm.predict(prompt)
    response = llm.invoke(prompt)
    # print(response)
    return response.content, df_context

def context_columns_based_on_language(language):
    if language == 'en':
        return ["BlockID","MaterialNumber", "MaterialName","MaterialDescription","PurchasingGroupDescription",
           "Block_reason","Blocked_date", "Solution", "Solution_date"]

    else:
        return ["BlockID","MaterialNumber", "MaterialName","MaterialDescription_KR","PurchasingGroupDescription_KR",
            "Block_reason_KR","Blocked_date", "Solution_KR", "Solution_date"]

# Example usage
# query = "가구 포장 불량 배송에 대한 해결방안은 무엇인가요?"
# query = "가구 품질불량에 대한 해결방안은 무엇인가요?"
# query = "What can be solution for the delivery for packing defects for furniture?"
# response = ask_llm(query=query, retrieval_augmented_generation=True, k=4)

def main():     
    iface = gr.Interface(
                     inputs=gr.Textbox(lines=7, placeholder="Enter your question here"),
                     fn=ask_llm,
                     outputs=[gr.Textbox(),
                              gr.Dataframe()],
                     title="Frost AI ChatBot: Your Knowledge Companion Powered-by ChatGPT",
                     description="Ask any question about Block reason & solutions - Dataset: IKEA")

    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)
    
if __name__ == "__main__":
    main()
