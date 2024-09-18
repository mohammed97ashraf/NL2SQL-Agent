import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate
from typing import Annotated, Sequence, TypedDict,List, Dict, Any, Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import base64
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import psycopg2
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
# importing module
import logging




load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ERD_info : BaseModel

# Create and configure logger
logging.basicConfig(filename="SQLagent.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


def load_image(image_path: str) -> str:
    """Load image from file and encode it as base64."""
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return image_base64

image_url = "pagila-schema-diagram.png"
image_data = load_image(image_url)


class text_to_sql_agent:
    def __init__(self, state : AgentState , image_url : str ) -> None:
        self.state = self.state
        self.image_data = load_image(image_url)
        pass

def get_related_table_details(state):
    """
    This function takes in the state and image data and returns the related table details.
    """
    # Get the related table details
    
    query = state["messages"][-1].content
    logging.warning(f"Sql agent started working {query}")
    class QueryParse(BaseModel):
        is_relevant: bool = Field(description="Is the query relevant to the database")
        relevant_tables_and_columns: Optional[dict] = Field(description="Relevant tables and columns, including foreign keys and primary keys details.")
    
    model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(QueryParse)

    prompt = [
        SystemMessage(content="""
            You are a highly skilled data analyst with expertise in analyzing SQL databases and understanding their structure. 
            You are provided with an ER diagram (ERD) for a PostgreSQL database and a natural language user query referred to as `query`.

            Your task is as follows:
            1. Use the ER diagram to carefully examine the structure of the PostgreSQL database, including tables, columns, relationships, foreign keys, and primary keys.
            2. Identify and list all relevant tables and columns that could be used to answer the user's query. This includes analyzing relationships between tables, such as foreign key constraints and how different tables connect to one another.
            3. **Do not generate any SQL queries.** Your job is to strictly identify and return only the relevant tables and columns, which will assist in creating the SQL query later.
            4. If the query does not align with the structure of the database or if there is insufficient information, mark `is_relevant` as `false`. Otherwise, set `is_relevant` to `true` and provide a detailed list of relevant tables and columns in JSON format, including primary and foreign key details.

            Make sure the response is concise and structured correctly, without any unnecessary details or explanations.
        """),
        HumanMessage(
            content=[
                {"type": "text", "text": f"query: {query}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
    ]

    response = model.invoke(prompt)
    state["ERD_info"] = response
    return state


def generate_sql_query(state):
    prompt = ChatPromptTemplate(
        [
            ("system", """
                You are an expert SQL query generator with deep knowledge of PostgreSQL syntax and best practices. 
                You are given a PostgreSQL database schema which contains relevant tables and columns, referred to as `relevant_tables_and_columns`. 
                Additionally, you will receive a natural language user query, referred to as `query`, that needs to be transformed into a valid, efficient, 
                and correct PostgreSQL SQL query. 

                Your task is to:
                1. Analyze the given `relevant_tables_and_columns` structure to identify the appropriate tables and columns needed for the query.
                2. Interpret the user's query and map it to the correct SQL operations (e.g., SELECT, JOIN, WHERE, GROUP BY, etc.).
                3. Ensure that the SQL query is optimized, adheres to PostgreSQL standards, and accurately represents the intent of the user's query.
                4. Handle edge cases such as ambiguous column names, missing conditions, or complex queries by making reasonable assumptions or asking for clarification when needed.

                Your response should consist solely of the generated SQL query in correct syntax, without any additional explanation.
            """),
            ("human", """
                relevant_tables_and_columns: {relevant_tables_and_columns}
                query: {query}
            """)
        ]
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm 
    response = chain.invoke({
        "relevant_tables_and_columns": state["ERD_info"].relevant_tables_and_columns, 
        "query": state["messages"][0].content
    })
    logging.info(f"generatin query for {state['messages'][0].content}\
                 with relevant_tables_and_columns {state['ERD_info'].relevant_tables_and_columns}")
    print(response)
    return {"messages": [response]}


@tool
def fatch_data_from_db(query: str) -> list:
    """Fetches data from a PostgreSQL database using the provided SQL query.
    This tool connects to a PostgreSQL database and executes the provided SQL query. 
    It retrieves all the resulting data and returns it as a list of tuples. Each tuple represents a row of data fetched from the database."""
    logging.info(f"Using the tool to fetch data from db {query}")
    conn = psycopg2.connect(database="postgres",
                        host="localhost",
                        user="postgres",
                        password="123456",
                        port="5432")
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    return result


def agent_with_tools(state):
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0).bind_tools([fatch_data_from_db])
    messages = state["messages"][-1].content
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def html_generater(state):
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0,max_tokens=16000, model_kwargs={"response_format": { "type": "json_object" }})
    messages = state["messages"][-1].content
    query = state["messages"][0].content

    prompt2 = [
        SystemMessage(content="""
            You are a skilled data formatter and visualization expert. 
            Given a PostgreSQL SQL query and the resulting data from the query, your task is to transform the output into a visually appealing HTML format.
            
            Here’s what you need to do:
            1. **Analyze the result data**: Review the structure of the result and determine the best way to present it, whether in a table, bar chart, line chart, or pie chart.
            2. **Charts**: You may use only 'bar', 'line', or 'pie' charts for visualizing the data. Choose a chart type based on the data distribution and what best represents the information.
               - Provide the chart details in the following format:
                 ```json
                 {
                     'format': 'chart',
                      'chart_type': '<A suitable chart format.',
                     'chart_title': '<A suitable title for the chart>',
                     'labels': <list of labels>,
                     'data': <list of corresponding data points>
                 }
                 ```
               - Example: 
                 ```json
                 {
                     'format': 'chart',
                      'chart_type': 'bar',
                     'chart_title': 'Monthly Sales Data',
                     'labels': ['January', 'February', 'March'],
                     'data': [500, 600, 550]
                 }
                 ```
            3. **Tables**: If a chart is not suitable for the data, present it in table format:
               - Provide the table details in this format:
                 ```json
                 {
                     'format': 'table',
                     'headers': <list of column headers>,
                     'rows': <list of rows with corresponding data>
                 }
                 ```
               - Example:
                 ```json
                 {
                     'format': 'table',
                     'headers': ['Name', 'Age', 'City'],
                     'rows': [
                         ['John', 28, 'New York'],
                         ['Jane', 22, 'Los Angeles'],
                         ['Doe', 35, 'Chicago']
                     ]
                 }
                 ```
            4. **No Visual Needed**: If the data doesn't require a chart or table (e.g., simple text), respond with a plain message in the following format:
               ```json
               {
                   'format': 'message',
                   'content': 'Provide the text message here'
               }
               ```
            Your goal is to select the most appropriate visual (chart, table, or message) based on the query results and format the response as structured JSON.
        """),
        HumanMessage(content=f"""query: {query}, result: {messages}""")
    ]

    response = llm.invoke(prompt2)
    # print(response.content)
    return {"messages": [response]}

def sanitize_sql_query_with_llm(state) -> Literal[ END , "agent_with_tools"]:
    """
    This function uses OpenAI's LLM to sanitize a given SQL query, allowing only 'SELECT' operations.
    Any query that contains 'UPDATE', 'INSERT', 'DELETE', or other modifying operations will be flagged as unsafe.

    Parameters:
    query (str): The SQL query to be sanitized.

    Returns:
    bool: True if the query is valid (SELECT-only), False otherwise.
    """
    class sanitizing_queries(BaseModel):
        is_safe: bool
        reason: str
    
    query = state["messages"][-1].content

    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(sanitizing_queries)
    
    # Define the system prompt to instruct the model to analyze the SQL query for safety
    prompt = [
        SystemMessage(content="""
            You are an expert in SQL security. Your task is to analyze a given SQL query and determine whether it is a safe read-only query. 
            Only queries that retrieve data (i.e., SELECT queries) are allowed. 
            If the query contains any modifying operations like INSERT, UPDATE, DELETE, DROP, ALTER, or TRUNCATE, mark it as unsafe.
            Your response should be in the following format:
            
            {
                "is_safe": true/false,
                "reason": "<Explain why the query is safe or unsafe>"
            }
        """),
        HumanMessage(content=f"""SQL query: {query}""")
    ]
    
    # Invoke the model with the provided query
    response = llm.invoke(prompt)
    
    if response.is_safe == True:
        return "agent_with_tools"
    else:
        globals()['error_message'] = {
                "format": "message",
                "content": response.reason
            }
        
    
        return END

def check_query_relavency(state) -> Literal[ END , "generate_sql_query"]:
    messgae = state["ERD_info"]
    if messgae.is_relevant == True:
        return "generate_sql_query"
    else:
        globals()['error_message'] = {
            "format": "message",
            "content": "The query is not relevant to this table."
        }
        return END
    

def get_langgraph_agent():
    # Define a new graph
    workflow = StateGraph(AgentState)

    #workflow.add_node("get_related_table_deatils", get_related_table_deatils)
    # Define the nodes we will cycle between
    workflow.add_node("get_related_table_details", get_related_table_details)
    # Define the nodes we will cycle between
    workflow.add_node("generate_sql_query", generate_sql_query)
    workflow.add_node("agent_with_tools", agent_with_tools)
    workflow.add_node("html_data_generater", html_generater)
    #workflow.add_node("sanitize_sql_query", sanitize_sql_query_with_llm)
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "get_related_table_details")
    #workflow.add_node("grade_documents", grade_documents)
    sql_tool = ToolNode([fatch_data_from_db])
    workflow.add_node("sql_tool", sql_tool)

    ## adding the condition edges
    workflow.add_conditional_edges(
        "get_related_table_details",
        # Assess agent decision
        check_query_relavency,
    )

    # Decide whether to contiue
    workflow.add_conditional_edges(
        "agent_with_tools",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "sql_tool",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "generate_sql_query",
        #
        sanitize_sql_query_with_llm

    )

    #workflow.add_edge("rewrite","agent_with_tools")
    workflow.add_edge("sql_tool", "html_data_generater")
    workflow.add_edge("html_data_generater", END)
    # Compile
    graph = workflow.compile()
    return graph
