import os
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from crewai_tools import ScrapeWebsiteTool, FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun,DuckDuckGoSearchResults, tool, YouTubeSearchTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import streamlit as st
from datetime import datetime

headers = {'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}

file_tool = FileReadTool('/news.txt')
wrapper = DuckDuckGoSearchAPIWrapper(region='br-pt', backend='api', time='w', source='text')
search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
search_results = DuckDuckGoSearchResults(num_results=5, api_wrapper=wrapper)
web_scrap_tool = ScrapeWebsiteTool()

@tool
def image_scraping_tool(url: str):
  '''  Use essa ferramenta para extrair somente as urls das imagens do site.

    *Args -> input: url do site alvo   
'''
  try:
    site = requests.get(url, headers=headers)
    soup = BeautifulSoup(site.content, 'html.parser')
    tag = soup.find_all('img')
    links_imagens= [l['src'] for l in tag if l['src']]
    return links_imagens
  
  except requests.exceptions.RequestException as e:
    print(f"Erro ao fazer a requisição: {e}")
    return []

os.environ['GROQ_API_KEY']=st.secrets['GRO_API_KEY']
llm_manager = ChatGroq(model='gemma2-9b-it', temperature=0.3)
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=os.environ.get('GROQ_API_KEY'))

research_agent = Agent(
  role='Web Researcher News',
  goal='Find relevant news',
  backstory="""You're a researcher at a large company.
  You're responsible for research relevant news insights.""",
  verbose=True,
  llm=llm,
  allow_delegation=False,
  tools=[search_results]
)
research_task = Task(
  description=f'Find the current news about the inputs user. The current date is {datetime.now()}.',
  expected_output='''Uma lista com as 2 notícias mais recentes sobre o tema: {assunto}.
  Forneça os links de onde pesquisou.
  ''',
  agent=research_agent,
  async_execution=True
)

research_agent_two = Agent(
  role='Web Researcher News',
  goal='Find relevant news',
  backstory="""You're a researcher at a large company.
  You're responsible for research relevant news insights.""",
  verbose=True,
  llm=llm,
  allow_delegation=False,
  tools=[search_results]
)
research_task_two = Task(
  description=f'Find the current news about the user input. The current date is {datetime.now()}.',
  expected_output='''Uma lista com as 2 notícias mais recentes sobre o tema: {assunto2}.
  Forneça os links de onde encontrou as 2 notícias''',
  agent=research_agent_two,
  async_execution=True
)

pesquisador_clubes = Agent(
  role='Pesquisador Esportivo',
  goal='Encontrar notícias relevantes sobre o time de futebol: {time}',
  backstory="""Você é um jornalista esportivo de uma grande site esportivo.
  Você é responsável por pesquisar notícias atuais sobre times de futebol.""",
  verbose=True,
  llm=llm,
  allow_delegation=False,
  tools=[search_results]
)
pesquisa_clubes_task = Task(
  description=f'Pesquisar pelos últimos acontecimentos sobre o time de futebol: {{time}}. A data atual é {datetime.now()}.',
  expected_output='''Uma lista com as 2 notícias mais recentes sobre o time de futebol: {time}.
  forneça também os links''',
  agent=pesquisador_clubes
)

image_scraper_agent = Agent(
  role='Web Urls Images Scraper',
  goal='Find Images Urls from a web site',
  backstory="""You're a Scraper Agent at a large company.
  You're responsible for research and scraping images urls from specified web site using your best tool for do that.""",
  verbose=True,
  allow_delegation=False,
  llm=llm,
  tools=[image_scraping_tool]
)
image_scraping_task = Task(
  description='''Encontre as urls das respectivas imagens das notícias que recebeu dos agentes,
  usando a sua ferramenta que faz a extração do links de imagens de um site especifico''',
  expected_output='''Uma lista com as respectivas urls das imagens extraídas''',
  agent=image_scraper_agent,
  context=[research_task, research_task_two, pesquisa_clubes_task]
  #async_execution=True
)

escritor = Agent(
    role='Escritor Sênior',
    goal='Escrever resumos chamativos e envolventes sobre os assuntos solicitados',
    backstory='Escritor sênior com 30 anos de experiência em escrever resumos de notícias criativos, trabalha em diversos jornais influentes',
    llm=llm,
    verbose=True,
    allow_delegation=True,
    tools = [file_tool],
    max_iter=15
)
tarefa_escritor = Task(
    agent = escritor,
    description = '''Elaborar uma Newsletter de fácil leitura no idioma Português das notícias recebidas dos agentes pesquisadores
    sobre os temas: {assunto} | {assunto2} | {time}

    Use a ferramenta para ler o arquivo modelo para se inspirar no mesmo formato de saída.
    ''',
    expected_output = '''
    siga as instruções definidas no <template>

    <template>

    *obervação -> Utilize exatamente o seguinte texto para começar a Newsletter:

        "Olá, {nome}! ✌️

        Seja bem-vindo à sua Newsletter!\n
        Reunimos alguns artigos relevantes para você se manter atualizado"

    
    *task -> Elabore uma Newsletter no idioma Português, de fácil leitura, com as notícias recebidas.

    - o título da noticia com a data dela ao lado\n
    - breve resumo da noticia com no máximo 2 parágrafos\n
    - link da notícia(Leia mais)
    
    </template>
''',
    context = [research_task, research_task_two ,pesquisa_clubes_task],
    #output_file='/news4.txt' 
)

conversor = Agent( # opção para enviar a newsletter por email
    role='Desenvolvedor HTML',
    goal='Formatar textos para a linguagem de programação HTML',
    backstory='Desenvolvedor HTML nível Sênior, trabalha transformando qualquer conteúdo recebido para o formato HTML',
    llm=llm,
    verbose=True,
    allow_delegation=False,
)
converter = Task(
    agent = conversor,
    description = '''Transformar o texto recebido do agente Escritor em formato HTML''',
    #Use a ferramenta para ler o arquivo modelo para se inspirar e usar o mesmo formato de saída.Garantindo que as notícias estejam recentes de acordo com a data atual: {datetime.now()}
    expected_output = '''<!DOCTYPE html>''',
    context = [tarefa_escritor],
    #formatada em <!DOCTYPE html>
)

# manager = Agent( --> opção para usar um gerente de projeto
#     role="Project Manager",
#     goal="Efficiently manage the crew and ensure high-quality task completion",
#     backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
#     allow_delegation=True,
# )

crew = Crew(
    agents=[research_agent,research_agent_two,pesquisador_clubes,escritor],
    tasks=[research_task,research_task_two,pesquisa_clubes_task,tarefa_escritor],
    verbose=2,
    process=Process.sequential,
    manager_llm=llm_manager
)


with st.sidebar:
    st.header('Preencha os campos para criar sua Newsletter:')

    with st.form(key='research_form'):
        name = st.text_input("Digite seu nome")
        topic = st.text_input("Digite um assunto")
        topic2 = st.text_input("Escolha mais um assunto")
        time = st.text_input("Digite um time de futebol")
        submit_button = st.form_submit_button(label = "Escrever a Newsletter!")
if submit_button:
    if not [topic,topic2,time,name]:
        st.error("Please fill the empty field")
    else:
        st.write('Aguarde alguns minutos, estamos gerando sua newsletter...')
        results = crew.kickoff(inputs={'nome':name,'assunto': topic, 'assunto2':topic2, 'time':time})
        st.subheader("Resultado:")
        #st.write(results)
        st.markdown(results)
        

# result = crew.kickoff({'assunto':'IA llms',
#                       'assunto2':'Marvel',
#                       'time':'Corinthians'})
# print(result)
