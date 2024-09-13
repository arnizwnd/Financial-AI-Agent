import os
import requests
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)

@tool
def get_company_overview(stock: str) -> str:
    """
    Get company overview, such as phone, email, website, market cap.
    """
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"

    return retrieve_from_endpoint(url)


@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """
    Get daily transaction for a stock from a range of start date and end date.
    """
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    return retrieve_from_endpoint(url)

@tool
def get_performance_since_ipo(stock: str) -> str:
    """
    Get stock performance since initial public offering (IPO) listing. Returns price change over the last 7 days (chg_7d), 30 days (chg_30d), 90 days (chg_90d), and 365 days (chg_365d).
    """
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"

    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(
    start_date: str, end_date: str, top_n: int = 5
) -> str:
    """
    Get top companies by transaction volume. This tool will return the top n companies 
    with the highest transaction volumes within a given date range, sorted in descending 
    order of total volume.
    """
    def aggregate_volumes(data_dict, top_n):
        data_dict = json.loads(data_dict)
        company_volumes = {}

        # Aggregate volumes for each company
        for companies in data_dict.values():
            for company in companies:
                name = company['company_name']
                volume = company['volume']

                if name not in company_volumes:
                    company_volumes[name] = {'symbol': company['symbol'], 'total_volume': 0}

                company_volumes[name]['total_volume'] += volume

        # Sort by total_volume in descending order and return the top n companies
        sorted_companies = sorted(company_volumes.items(), key=lambda item: item[1]['total_volume'], reverse=True)
        return dict(sorted_companies[:top_n])

    # Fetch data from the endpoint
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    aggregated_data = aggregate_volumes(retrieve_from_endpoint(url), top_n)

    # Create a formatted string to display the sorted data as a table
    result = "Top Companies by Transaction Volume\n"
    result += "--------------------------------\n"
    result += f"{'Company Name':<20}{'Symbol':<10}{'Total Volume':<15}\n"
    result += "--------------------------------\n"
    
    for company, data in aggregated_data.items():
        result += f"{company:<20}{data['symbol']:<10}{data['total_volume']:<15}\n"

    return result


tools = [
    get_company_overview,
    # we created this in the earlier code chunk under Exercise 1
    # (so make sure you've run that cell),
    get_top_companies_by_tx_volume,
    get_daily_tx,
    get_performance_since_ipo
]

llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        prompt_input = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""
                    Answer the following queries, being as factual and analytical as you can.
                    If you need the start and end dates but they are not explicitly provided,
                    infer from the query. Whenever you return a list of names, return also the
                    corresponding values for each name. If the volume was about a single day,
                    the start and end parameter should be the same. Note that the endpoint for
                    performance since IPO has only one required parameter, which is the stock.
                    If a comparison needed between two stock or company, invoke queries for both stock.
                    For each query, select one of the available and suitable tool.
                    if the date is not specified just give the top company from last week to
                    {datetime.today()}. Today is {datetime.today()}.

                    If the data for the start_date is unavailable due to it being a non-trading day, you must:
                    1. Return the message: "The data is unavailable for because it is a non-trading day." and then
                    2. Invoke again with new start_date + 1 day so valid data is found.
                    3. Once valid stock data is found, display the results for that actual day and return the top N companies based on transaction volume.
                    4. Always order by total volume descending as a table, do not try to order by company name.
                    """
                ),
                ("human", prompt),
                # msg containing previous agent tool invocations and corresponding tool outputs
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt_input)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input":prompt})
        # response = agent.run(prompt, callbacks=[st_callback])
        st.write(result["output"])