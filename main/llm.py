

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


sentences = ["   Analista de Dados Contato (11) 94793-9258  (Mobile) dscarneiro03@gmail.com www.linkedin.com/in/ds-carneiro (LinkedIn) Principais competências Pensamento crítico Mapeamento de dados Resolução de problemas Languages Inglês  (Limited Working) Espanhol  (Elementary) Português  (Native or Bilingual) Certifications Certificado de Qualificação Profissional em Desenvolvimento e Designer Web 2.0 Certificado de Qualificação Profissional em Estratégia e Inovação Tecnológica com aplicações em IA e IoT Certificado de Qualificação Profissional em Análise de Sistemas e Prototipação Web Lista completa dos certificados Alura Certificado de Qualificação Profissional em Desenvolvimento de Aplicações MóveisGustavo Carneiro Alves Analista de Dados | Cientista de Dados São Paulo, São Paulo, Brasil Resumo Sou um estudante de ciência de dados com mais de 200 horas em cursos extracurriculares em dados e apaixonado por resolver problemas complexos. Minhas principais habilidades são focadas no pensamento crítico, storytelling com dados, codificação, machine learning, gestão de banco de dados, Excel, PowerBI, SQL e Python. Participei de diversos projetos acadêmicos, como análise de dados de saúde, previsão de custos empresariais e classificação de imagens. Meu objetivo profissional é ingressar na área de dados como analista de dados e posteriormente virar um cientista de dados. Tenho interesse em me especializar em uma área de negócios que me desafie e me permita aplicar meus conhecimentos e habilidades. Formação acadêmica FIAP Curso Superior de Tecnologia (CST), Análise e Desenvolvimento de Software  · (janeiro de 2020 - dezembro de 2022) Instituto Infnet Curso Superior de Tecnologia (CST), Ciência de Dados Analista de Dados · (2024)   Page 1 of 1", 
             "Vaga: Analista de dados Atribuições:Coleta e análise de dados e informações: elaboração de análises e relatórios de desempenho de performance;Identificação de tendências: avaliar demanda e desempenho da transportadora mapeando e analisando despesas com frete e performance da equipe de transportes;Planejamento organizacional: análise e criação de sistemas de controle e de organização, mapear os resultados de rotinas de transporte e movimentação de cargas;Gestão de indicadores das equipes: garantir visibilidade dos indicadores das equipes, criando painéis de gestão (DashBoards);Governança de dados e suporte: dimensiona recursos e presta suporte no acompanhamento das demandas administrativas;Controle de fluxos de trabalho com objetivo de aprimorá-los.Requisitos obrigatórios:Superior em ciência de dados ou áreas correlatas (cursando)Excel avançadoPower BISalário: R$ 3.520,00Benefícios: Vale transportes + vale alimentação no valor de R$ 31,00/dia + bônus por indicador de desempenhoHorário: 08:00 ás 17:48h - segunda a sexta-feiraFormato híbrido: 3x presencial e 2x home officeNecessário disponibilidade para viagensLocal: O candidato pode ser do Rio de Janeiro ou de São PauloTipo de vaga: Tempo integral, Efetivo CLTPagamento: R$3.520,00 por mêsBenefícios:Vale-alimentaçãoVale-transporteHorário de trabalho:De segunda à sexta-feiraTurno de 8 horasPagamento adicional:BônusPergunta(s) de seleção:Qual seu nível de conhecimento de Power Bi?Qual seu nível de excel?Tem disponibilidade para viagens?"]
sentences2 = ["   Analista de Dados Contato (11) 94793-9258  (Mobile) dscarneiro03@gmail.com www.linkedin.com/in/ds-carneiro (LinkedIn) Principais competências Pensamento crítico Mapeamento de dados Resolução de problemas Languages Inglês  (Limited Working) Espanhol  (Elementary) Português  (Native or Bilingual) Certifications Certificado de Qualificação Profissional em Desenvolvimento e Designer Web 2.0 Certificado de Qualificação Profissional em Estratégia e Inovação Tecnológica com aplicações em IA e IoT Certificado de Qualificação Profissional em Análise de Sistemas e Prototipação Web Lista completa dos certificados Alura Certificado de Qualificação Profissional em Desenvolvimento de Aplicações MóveisGustavo Carneiro Alves Analista de Dados | Cientista de Dados São Paulo, São Paulo, Brasil Resumo Sou um estudante de ciência de dados com mais de 200 horas em cursos extracurriculares em dados e apaixonado por resolver problemas complexos. Minhas principais habilidades são focadas no pensamento crítico, storytelling com dados, codificação, machine learning, gestão de banco de dados, Excel, PowerBI, SQL e Python. Participei de diversos projetos acadêmicos, como análise de dados de saúde, previsão de custos empresariais e classificação de imagens. Meu objetivo profissional é ingressar na área de dados como analista de dados e posteriormente virar um cientista de dados. Tenho interesse em me especializar em uma área de negócios que me desafie e me permita aplicar meus conhecimentos e habilidades. Formação acadêmica FIAP Curso Superior de Tecnologia (CST), Análise e Desenvolvimento de Software  · (janeiro de 2020 - dezembro de 2022) Instituto Infnet Curso Superior de Tecnologia (CST), Ciência de Dados Analista de Dados · (2024)   Page 1 of 1", 
             "Atividades: Projetar e executar de forma criativa e científica soluções para os espaços interiores, visando através da oferta de projetos com estética, eficiência, segurança e conforto, melhores vendas.Local de trabalho: Avendida Teodoro Sampaio, Pinheiros - São PauloHorário de trabalho: Segunda a sábado, comercial.Formação necessária: Designer de Interiores ou de Ambientes, Arquitetura.Necessário: experiência nas atividades, conhecimento do Promob.Oferecemos: Vale transporte, comissão, convênio médico e odontológico."]


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
#print(embeddings)

# transforma o embedding em um valor numérico
cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(cos_sim)

embeddings = model.encode(sentences2)
cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(cos_sim)






from transformers import pipeline

context = r"""

Vaga: Analista de dados Atribuições:Coleta e análise de dados e informações: elaboração de análises e relatórios de desempenho de performance;Identificação de tendências: avaliar demanda e desempenho da transportadora mapeando e analisando despesas com frete e performance da equipe de transportes;Planejamento organizacional: análise e criação de sistemas de controle e de organização, mapear os resultados de rotinas de transporte e movimentação de cargas;Gestão de indicadores das equipes: garantir visibilidade dos indicadores das equipes, criando painéis de gestão (DashBoards);Governança de dados e suporte: dimensiona recursos e presta suporte no acompanhamento das demandas administrativas;Controle de fluxos de trabalho com objetivo de aprimorá-los.Requisitos obrigatórios:Superior em ciência de dados ou áreas correlatas (cursando)Excel avançadoPower BISalário: R$ 3.520,00Benefícios: Vale transportes + vale alimentação no valor de R$ 31,00/dia + bônus por indicador de desempenhoHorário: 08:00 ás 17:48h - segunda a sexta-feiraFormato híbrido: 3x presencial e 2x home officeNecessário disponibilidade para viagensLocal: O candidato pode ser do Rio de Janeiro ou de São PauloTipo de vaga: Tempo integral, Efetivo CLTPagamento: R$3.520,00 por mêsBenefícios:Vale-alimentaçãoVale-transporteHorário de trabalho:De segunda à sexta-feiraTurno de 8 horasPagamento adicional:BônusPergunta(s) de seleção:Qual seu nível de conhecimento de Power Bi?Qual seu nível de excel?Tem disponibilidade para viagens?

"""

model_name = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
nlp = pipeline("question-answering", model=model_name)

question = "A vaga possui qualquer benefício?"

result = nlp(question=question, context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

