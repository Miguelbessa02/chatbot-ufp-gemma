import requests
from bs4 import BeautifulSoup
import re
import os

cursos_urls = {
#    "engenharia_informatica": "https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/engenharia-informatica/",
#    "mestrado_informatica": "https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/engenharia-informatica/",
#    "arquitetura": "https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/arquitetura/",
#   "ciencia_politica_relaoes_internacionais": "https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/ciencia-politica-e-relacoes-internacionais/",
#   "ciencias_da_comunicacao":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/ciencias-da-comunicacao/",
#    "ciencias-da_nutricao": "https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/ciencias-da-nutricao/",
#    "ciencias_empresariais":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/ciencias-empresariais/",
#    "ciencias-farmaceuticas":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/ciencias-farmaceuticas/",
#    "criminologia":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/criminologia/",
#    "medicina":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/medicina/",
#    "medicina_dentaria":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/medicinadentaria/",
#    "psicologia":"https://www.ufp.pt/inicio/estudar-e-investigar/licenciaturas/psicologia/",
#    "mestrado-acao-humanitaria-cooperacao-e-desenvolvimento": "https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/acao-humanitaria-cooperacao-e-desenvolvimento-2/",
#    "mestrado-biomedicina":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/biomedicina/",
#    "mestrado-ciencias-da-comunicacao":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/ciencias-da-comunicacao/",
# "mestrado-ciencias-da-educacao-educacao-especial":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/ciencias-da-educacao-educacao-especial-2-2/",
#  "mestrado-criminologia":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/criminologia/",
#  "mestrado-psicologia-clinica-e-da-saude":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/psicologia-clinica-e-da-saude/",
#   "mestrado-psicologia-da-justica-vitimas-de-violencia-e-de-crime":"https://www.ufp.pt/inicio/estudar-e-investigar/mestrados/psicologia-da-justica-vitimas-de-violencia-e-de-crime/"
"Sobre_a_ufp":"https://www.ufp.pt/inicio/conhecer-a-ufp/"
}

# Função para extrair texto
def extrair_texto_generico(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    candidates = [
        soup.find("main"),
        soup.find("article"),
        soup.find("div", class_="elementor"),
        soup.find("div", id="content"),
        soup.find("div", class_="site-content"),
        soup.body
    ]

    for candidate in candidates:
        if candidate:
            content = candidate
            break
    else:
        return "Não foi possível encontrar conteúdo significativo."

    for tag in content(["script", "style", "noscript", "iframe"]):
        tag.decompose()

    text = content.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# Guardar ficheiros
os.makedirs("cursos_ufp_txt", exist_ok=True)

for curso, url in cursos_urls.items():
    print(f"Extraindo: {curso}")
    texto = extrair_texto_generico(url)
    with open(f"cursos_ufp_txt/{curso}.txt", "w", encoding="utf-8") as f:
        f.write(texto)
print("Todos os ficheiros foram guardados na pasta 'cursos_ufp_txt'.")