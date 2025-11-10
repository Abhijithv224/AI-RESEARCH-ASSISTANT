from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import requests
import json
from tqdm import tqdm
import time



load_dotenv()
GROQ_API=os.getenv('GROQ_API')

def payload( keyword, page=1,):

    headers = {
        "Connection": "keep-alive",
        "sec-ch-ua": '"Google Chrome";v="95", "Chromium";v="95", ";Not A Brand";v="99"',
        "Cache-Control": "no-cache,no-store,must-revalidate,max-age=-1",
        "Content-Type": "application/json",
        "sec-ch-ua-mobile": "?1",
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Mobile Safari/537.36",
        "X-S2-UI-Version": "20166f1745c44b856b4f85865c96d8406e69e24f",
        "sec-ch-ua-platform": '"Android"',
        "Accept": "*/*",
        "Origin": "https://www.semanticscholar.org",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://www.semanticscholar.org/search?year%5B0%5D=2018&year%5B1%5D=2022&q=multi%20label%20text%20classification&sort=relevance",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    }

    data = json.dumps(
        {
            "queryString": f"{keyword.lower()}",
            "page": page,
            "pageSize": 10,
            "sort": "relevance",
            "authors": [],
            "coAuthors": [],
            "venues": [],
            "requireViewablePdf": True,
            "fieldsOfStudy": [],
            "hydrateWithDdb": True,
            "includeTldrs": True,
            "performTitleMatch": True,
            "includeBadges": True,
            "getQuerySuggestions": False,
        }
    )

    response = requests.post(
        "https://www.semanticscholar.org/api/1/search", headers=headers, data=data
    )
    return response

def soup_html( output):
    """Semantic Scholar Result"""

    final_result = []
    output = output.json()["results"]

    for paper in output:

        result = {}

        result["title"] = paper.get("title", {}).get("text", "No title found")

        result["abstract"] = paper.get("tldr", {}).get("text", "No abstract/TLDR found")

        authors_list = paper.get("authors", [])
        if authors_list:

            author_names = [author_group[0].get("name", "N/A") for author_group in authors_list if author_group]
            result["authors"] = ", ".join(author_names)
        else:
            result["authors"] = "No authors found"

        if paper.get("primaryPaperLink") and paper["primaryPaperLink"].get("url"):
            result["link"] = paper["primaryPaperLink"]["url"]
        elif paper.get("alternatePaperLinks"):
            result["link"] = paper["alternatePaperLinks"][0].get("url", "no_link_found")
        else:
            result["link"] = "no_link_found"

        final_result.append(result)

    return final_result

def retrive_paper(
    keyword,
    max_pages=5,
    full_page_result=False,
    api_wait=5,
):
    "ss function call"

    all_pages = []
    for page in tqdm(range(1, max_pages + 1)):
        ss_soup = payload(
            keyword, page=page,
        )

        ss_result = soup_html(ss_soup)
        all_pages.extend(ss_result)
        time.sleep(api_wait)

    return all_pages
        
def create_vectorstore(url,embedding):
    loader=PyPDFLoader(url)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    docs=text_splitter.split_documents(documents)
    db=FAISS.from_documents(docs,embedding=embedding)
    return db

def get_answer(llm,db,query):
    retriever=db.as_retriever()
    prompt = ChatPromptTemplate.from_template("""
You are research assisitant. 
Answer the user's question clearly and factually using the given context, but do not mention or reference the context explicitly. 
If you don't know the answer, just say that you don't know.
                                              
<context>
{context}
</context>
Question: {input}""")
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({"input":query})
    return response['answer']
